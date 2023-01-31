/// \file PromoteCSVs.cpp
/// \brief

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/STLExtras.h"

#include "revng/ADT/GenericGraph.h"
#include "revng/FunctionIsolation/IsolationFunctionKind.h"
#include "revng/FunctionIsolation/PromoteCSVs.h"
#include "revng/FunctionIsolation/StructInitializers.h"
#include "revng/MFP/MFP.h"
#include "revng/MFP/SetLattices.h"
#include "revng/Pipeline/AllRegistries.h"
#include "revng/Pipeline/Contract.h"
#include "revng/Pipes/RootKind.h"
#include "revng/Support/IRHelpers.h"

using namespace llvm;
using namespace MFP;

char PromoteCSVsPass::ID = 0;
using Register = RegisterPass<PromoteCSVsPass>;
static Register X("promote-csvs", "Promote CSVs Pass", true, true);

struct PromoteCSVsPipe {
  static constexpr auto Name = "promote-csvs";

  std::vector<pipeline::ContractGroup> getContract() const {
    using namespace pipeline;
    using namespace ::revng::kinds;
    return { ContractGroup::transformOnlyArgument(ABIEnforced,
                                                  CSVsPromoted,
                                                  InputPreservation::Erase) };
  }

  void registerPasses(llvm::legacy::PassManager &Manager) {
    Manager.add(new PromoteCSVsPass());
  }
};

static pipeline::RegisterLLVMPass<PromoteCSVsPipe> Y;

// TODO: switch from CallInst to CallBase

struct CSVsUsageMap {
  std::map<Function *, CSVsUsage> Functions;
  std::map<CallInst *, CSVsUsage> Calls;

  CSVsUsage &get(CallInst *Call) {
    auto It = Calls.find(Call);
    if (It != Calls.end()) {
      return It->second;
    } else {
      return Functions.at(getCallee(Call));
    }
  }
};

class PromoteCSVs {
private:
  struct WrapperKey {
  public:
    Function *Helper;
    std::set<GlobalVariable *> Read;
    std::set<GlobalVariable *> Written;

  private:
    auto tie() const { return std::tie(Helper, Read, Written); }

  public:
    bool operator<(const WrapperKey &Other) const {
      return tie() < Other.tie();
    }
  };

private:
  Module *M;
  StructInitializers Initializers;
  OpaqueFunctionsPool<StringRef> CSVInitializers;
  std::map<WrapperKey, Function *> Wrappers;
  std::set<GlobalVariable *> CSVs;

public:
  PromoteCSVs(Module *M, GeneratedCodeBasicInfo &GCBI);

public:
  void run();

private:
  void wrap(CallInst *Call,
            ArrayRef<GlobalVariable *> Read,
            ArrayRef<GlobalVariable *> Written);
  void promoteCSVs(Function *F);
  Function *createWrapper(const WrapperKey &Key);
  CSVsUsageMap getUsedCSVs(ArrayRef<CallInst *> CallsRange);
  void wrapCallsToHelpers(Function *F);
};

PromoteCSVs::PromoteCSVs(Module *M, GeneratedCodeBasicInfo &GCBI) :
  M(M), Initializers(M), CSVInitializers(M, false) {

  CSVInitializers.addFnAttribute(Attribute::ReadOnly);
  CSVInitializers.addFnAttribute(Attribute::NoUnwind);
  CSVInitializers.addFnAttribute(Attribute::WillReturn);
  CSVInitializers.setTags({ &FunctionTags::OpaqueCSVValue });

  // Record existing initializers
  const auto &PCCSVs = GCBI.programCounterHandler()->pcCSVs();
  for (GlobalVariable *CSV :
       llvm::concat<GlobalVariable *const>(GCBI.csvs(), PCCSVs)) {
    if (GCBI.isSPReg(CSV))
      continue;

    CSVs.insert(CSV);
    if (auto *F = M->getFunction((Twine("init_") + CSV->getName()).str()))
      if (FunctionTags::OpaqueCSVValue.isTagOf(F))
        CSVInitializers.record(CSV->getName(), F);
  }
}

// TODO: assign alias information
Function *PromoteCSVs::createWrapper(const WrapperKey &Key) {
  auto &[Helper, Read, Written] = Key;

  LLVMContext &Context = Helper->getParent()->getContext();
  auto *PointeeTy = Helper->getType()->getPointerElementType();
  auto *HelperType = cast<FunctionType>(PointeeTy);

  //
  // Create new argument list
  //
  SmallVector<Type *, 16> NewArguments;

  // Initialize with base arguments
  std::copy(HelperType->param_begin(),
            HelperType->param_end(),
            std::back_inserter(NewArguments));

  // Add type of read registers
  for (GlobalVariable *CSV : Read)
    NewArguments.push_back(CSV->getType()->getPointerElementType());

  // Add out arguments for written registers
  const unsigned FirstOutArgument = NewArguments.size();
  for (GlobalVariable *CSV : Written)
    NewArguments.push_back(CSV->getType());

  //
  // Create new helper wrapper function
  //
  auto *NewHelperType = FunctionType::get(HelperType->getReturnType(),
                                          NewArguments,
                                          false);
  auto *HelperWrapper = Function::Create(NewHelperType,
                                         Helper->getLinkage(),
                                         Twine(Helper->getName()) + "_wrapper",
                                         Helper->getParent());
  HelperWrapper->setSection(Helper->getSection());

  // Copy and extend tags
  auto Tags = FunctionTags::TagsSet::from(Helper);
  Tags.insert(FunctionTags::CSVsAsArgumentsWrapper);
  Tags.set(HelperWrapper);

  auto *Entry = BasicBlock::Create(Context, "", HelperWrapper);

  //
  // Populate the helper wrapper function
  //
  IRBuilder<> Builder(Entry);

  // Serialize read CSV
  auto It = HelperWrapper->arg_begin();
  for (unsigned I = 0; I < HelperType->getNumParams(); I++, It++) {
    // Do nothing
    revng_assert(It != HelperWrapper->arg_end());
  }

  for (GlobalVariable *CSV : Read) {
    revng_assert(It != HelperWrapper->arg_end());
    Builder.CreateStore(&*It, CSV);
    It++;
  }

  // Prepare the arguments
  SmallVector<Value *, 16> HelperArguments;
  It = HelperWrapper->arg_begin();
  for (unsigned I = 0; I < HelperType->getNumParams(); I++, It++) {
    revng_assert(It != HelperWrapper->arg_end());
    HelperArguments.push_back(&*It);
  }

  // Create the function call
  auto *HelperResult = Builder.CreateCall(Helper, HelperArguments);

  // Update values of the out arguments
  unsigned OutArgument = FirstOutArgument;
  for (GlobalVariable *CSV : Written) {
    Builder.CreateStore(Builder.CreateLoad(CSV),
                        HelperWrapper->getArg(OutArgument));
    ++OutArgument;
  }

  if (HelperResult->getType()->isVoidTy()) {
    Builder.CreateRetVoid();
  } else {
    Builder.CreateRet(HelperResult);
  }

  return HelperWrapper;
}

template<typename T>
std::set<T> toSet(ArrayRef<T> AR) {
  std::set<T> Result;
  copy(AR, std::inserter(Result, Result.begin()));
  return Result;
}

void PromoteCSVs::wrap(CallInst *Call,
                       ArrayRef<GlobalVariable *> Read,
                       ArrayRef<GlobalVariable *> Written) {

  if (Read.size() == 0 and Written.size() == 0)
    return;

  Function *Helper = getCallee(Call);
  revng_assert(Helper != nullptr);

  WrapperKey Key{ Helper, toSet(Read), toSet(Written) };

  // Fetch or create the wrapper
  Function *&HelperWrapper = Wrappers[Key];
  if (HelperWrapper == nullptr)
    HelperWrapper = createWrapper(Key);

  auto *PointeeTy = Helper->getType()->getPointerElementType();
  auto *HelperType = cast<FunctionType>(PointeeTy);

  //
  // Emit call to the helper wrapper
  //
  auto EntryIt = Call->getParent()->getParent()->getEntryBlock().begin();
  IRBuilder<> AllocaBuilder(&*EntryIt);
  IRBuilder<> Builder(Call);

  // Initialize the new set of arguments with the old ones
  SmallVector<Value *, 16> NewArguments;
  for (auto [Argument, Type] : zip(Call->args(), HelperType->params()))
    NewArguments.push_back(Builder.CreateBitOrPointerCast(Argument, Type));

  // Add arguments read
  for (GlobalVariable *CSV : Read)
    NewArguments.push_back(Builder.CreateLoad(CSV));

  SmallVector<AllocaInst *, 16> WrittenCSVAllocas;
  for (GlobalVariable *CSV : Written) {
    Type *AllocaType = CSV->getType()->getPointerElementType();
    auto *OutArgument = AllocaBuilder.CreateAlloca(AllocaType);
    WrittenCSVAllocas.push_back(OutArgument);
    NewArguments.push_back(OutArgument);
  }

  // Emit the actual call
  Instruction *Result = Builder.CreateCall(HelperWrapper, NewArguments);
  Result->setDebugLoc(Call->getDebugLoc());
  Call->replaceAllUsesWith(Result);

  // Restore into CSV the written registers
  for (const auto &[CSV, Alloca] : zip(Written, WrittenCSVAllocas))
    Builder.CreateStore(Builder.CreateLoad(Alloca), CSV);

  // Erase the old call
  eraseFromParent(Call);
}

static Instruction *findFirstNonAlloca(BasicBlock *BB) {
  for (Instruction &I : *BB)
    if (not isa<AllocaInst>(&I))
      return &I;
  return nullptr;
}

void PromoteCSVs::promoteCSVs(Function *F) {
  // Create an alloca for each CSV and replace all uses of CSVs with the
  // corresponding allocas
  BasicBlock &Entry = F->getEntryBlock();

  // Get/create initializers
  std::map<Function *, GlobalVariable *> CSVForInitializer;
  std::map<GlobalVariable *, Function *> InitializerForCSV;
  for (GlobalVariable *CSV : CSVs) {
    // Initialize all allocas with opaque, CSV-specific values
    Type *CSVType = CSV->getType()->getPointerElementType();
    auto *Initializer = CSVInitializers.get(CSV->getName(),
                                            CSVType,
                                            {},
                                            Twine("init_") + CSV->getName());
    CSVForInitializer[Initializer] = CSV;
    InitializerForCSV[CSV] = Initializer;
  }

  // Collect existing CSV allocas
  std::map<GlobalVariable *, AllocaInst *> CSVAllocas;
  for (Instruction &I : Entry) {
    if (auto *Call = dyn_cast<CallInst>(&I)) {
      auto It = CSVForInitializer.find(Call->getCalledFunction());
      if (It != CSVForInitializer.end()) {
        auto *Initializer = cast<StoreInst>(getUniqueUser(Call));
        auto *Alloca = cast<AllocaInst>(Initializer->getPointerOperand());
        CSVAllocas[It->second] = Alloca;
      }
    }
  }

  Instruction *NonAlloca = findFirstNonAlloca(&Entry);
  revng_assert(NonAlloca != nullptr);

  IRBuilder<> InitializersBuilder(NonAlloca);
  auto *Separator = InitializersBuilder.CreateUnreachable();
  IRBuilder<> AllocaBuilder(&Entry, Entry.begin());

  // For each GlobalVariable representing a CSV used in F, create a dedicated
  // alloca and save it in CSVMaps.
  for (GlobalVariable *CSV : CSVs) {
    AllocaInst *Alloca = nullptr;

    auto It = CSVAllocas.find(CSV);
    if (It != CSVAllocas.end()) {
      Alloca = It->second;
    } else {
      // Create the alloca
      Type *CSVType = CSV->getType()->getPointerElementType();
      Alloca = AllocaBuilder.CreateAlloca(CSVType, nullptr, CSV->getName());

      // Check if already have an initializer
      Function *Initializer = InitializerForCSV.at(CSV);
      auto *InitializerCall = InitializersBuilder.CreateCall(Initializer);

      // Initialize the alloca
      InitializersBuilder.CreateStore(InitializerCall, Alloca);
    }

    // Replace users
    replaceAllUsesInFunctionWith(F, CSV, Alloca);
  }

  FunctionTags::CSVsPromoted.addTo(F);

  // Drop separators
  eraseFromParent(Separator);

#ifndef NDEBUG
  auto It = findFirstNonAlloca(&Entry)->getIterator();
  for (Instruction &I : make_range(It, Entry.end()))
    revng_assert(not isa<AllocaInst>(&I));
#endif
}

struct FunctionNodeData {
  Function *F;
  using UsedCSVSet = std::set<std::pair<bool, GlobalVariable *>>;
  UsedCSVSet UsedCSVs;
};

using FunctionNode = ForwardNode<FunctionNodeData>;
using GenericCallGraph = GenericGraph<FunctionNode>;

static FunctionNode *getNode(std::map<Function *, FunctionNode *> &NodeMap,
                             GenericCallGraph &Graph,
                             Function *F) {
  FunctionNode *Result = nullptr;

  auto It = NodeMap.find(F);
  if (It == NodeMap.end()) {
    Result = Graph.addNode();
    Result->F = F;
    NodeMap[F] = Result;
  } else {
    Result = It->second;
  }

  return Result;
}

static void addEdge(FunctionNode *Source, FunctionNode *Destination) {

  for (auto *Successor : Source->successors())
    if (Successor == Destination)
      return;

  Source->addSuccessor(Destination);
}

static bool needsWrapper(Function *F) {
  // Ignore lifted functions and functions that have already been wrapped
  {
    using namespace FunctionTags;
    auto Tags = TagsSet::from(F);
    if (Tags.contains(Isolated) or Tags.contains(CSVsAsArgumentsWrapper)
        or Tags.contains(Marker) or Tags.contains(Exceptional))
      return false;
  }

  if (F->isIntrinsic())
    return false;

  auto IsPointer = [](Type *T) { return T->isPointerTy(); };

  return any_of(F->getFunctionType()->params(), IsPointer);
}

struct UsedRegistersMFI : public SetUnionLattice<FunctionNodeData::UsedCSVSet> {
  using Label = FunctionNode *;
  using GraphType = GenericCallGraph *;

  static LatticeElement
  applyTransferFunction(Label L, const LatticeElement &Value) {
    return combineValues(L->UsedCSVs, Value);
  }
};

CSVsUsageMap PromoteCSVs::getUsedCSVs(ArrayRef<CallInst *> CallsRange) {
  CSVsUsageMap Result;

  // Note: this graph goes from callee to callers
  GenericCallGraph CallGraph;

  std::map<Function *, FunctionNode *> NodeMap;

  // Inspect the calls we need to analyze
  //
  // There are two type of calls: calls to helpers tagged by CSAA and calls to
  // regular functions. For the former, we ask GCBI to extract the information
  // from metadata. For the latter, we use a monotone framework to compute the
  // set of read/written registers by the callee.  Note that the former is more
  // accurate thanks to CSAA being call-site sensitive.
  std::queue<Function *> Queue;
  for (CallInst *Call : CallsRange) {
    Function *Callee = getCallee(Call);
    if (FunctionTags::Helper.isTagOf(Callee)) {
      CSVsUsage &Usage = Result.Calls[Call];
      auto UsedCSVs = getCSVUsedByHelperCall(Call);
      Usage.Read = UsedCSVs.Read;
      Usage.Written = UsedCSVs.Written;
    } else {
      Queue.push(Callee);
    }
  }

  while (not Queue.empty()) {
    Function *F = Queue.front();
    Queue.pop();

    auto *CallerNode = getNode(NodeMap, CallGraph, F);

    for (BasicBlock &BB : *F) {

      for (Instruction &I : BB) {
        bool Write = false;
        GlobalVariable *CSV = nullptr;

        if (auto *Store = dyn_cast<StoreInst>(&I)) {

          // Record store
          Write = true;
          CSV = dyn_cast<GlobalVariable>(skipCasts(Store->getPointerOperand()));

        } else if (auto *Load = dyn_cast<StoreInst>(&I)) {

          // Record load
          CSV = dyn_cast<GlobalVariable>(skipCasts(Store->getPointerOperand()));

        } else if (auto *Call = dyn_cast<CallInst>(&I)) {
          Function *Callee = getCallee(Call);
          revng_assert(Callee != nullptr);

          // In case we meet an `abort` skip this block
          if (Callee->getName() == "abort")
            break;

          // TODO: use forwardTaintAnalysis
          if (not needsWrapper(Callee))
            continue;

          // Ensure callee is visited
          if (NodeMap.count(Callee) == 0)
            Queue.push(Callee);

          // Insert an edge in the call graph
          auto *CalleeNode = getNode(NodeMap, CallGraph, Callee);
          addEdge(CalleeNode, CallerNode);
        }

        // If there was a memory access targeting a CSV, record it
        if (CSVs.count(CSV) != 0) {
          CallerNode->UsedCSVs.insert({ Write, CSV });
        }
      }
    }
  }

  auto AnalysisResult = getMaximalFixedPoint<UsedRegistersMFI>({},
                                                               &CallGraph,
                                                               {},
                                                               {},
                                                               {},
                                                               {});

  // Populate results set
  for (auto &[Label, Value] : AnalysisResult) {
    auto &FunctionDescriptor = Result.Functions[Label->F];
    for (auto [IsWrite, CSV] : Value.OutValue) {
      if (IsWrite)
        FunctionDescriptor.Written.push_back(CSV);
      else
        FunctionDescriptor.Read.push_back(CSV);
    }
  }

  return Result;
}

template<typename T>
ArrayRef<T> oneElement(T &Element) {
  return ArrayRef(&Element, 1);
}

void PromoteCSVs::wrapCallsToHelpers(Function *F) {
  std::vector<CallInst *> ToWrap;
  for (BasicBlock &BB : *F) {
    for (Instruction &I : BB) {
      if (auto *Call = dyn_cast<CallInst>(&I)) {
        Function *Callee = getCallee(Call);

        // Ignore calls to isolated functions
        if (Callee == nullptr or not needsWrapper(Callee))
          continue;

        ToWrap.emplace_back(Call);
      }
    }
  }

  auto UsedCSVs = getUsedCSVs(ToWrap);

  for (CallInst *Call : ToWrap) {
    CSVsUsage &CSVsUsage = UsedCSVs.get(Call);

    // Sort to ensure compatibility between caller and callee
    CSVsUsage.sort();

    wrap(Call, CSVsUsage.Read, CSVsUsage.Written);
  }
}

void PromoteCSVs::run() {
  for (Function &F : FunctionTags::ABIEnforced.functions(M)) {
    // Wrap calls to wrappers
    wrapCallsToHelpers(&F);

    // (Re-)promote CSVs
    promoteCSVs(&F);
  }
}

bool PromoteCSVsPass::runOnModule(Module &M) {
  auto &GCBI = getAnalysis<GeneratedCodeBasicInfoWrapperPass>().getGCBI();
  PromoteCSVs HW(&M, GCBI);
  HW.run();
  return true;
}
