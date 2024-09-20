/// \file PromoteCSVs.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"

#include "revng/ADT/GenericGraph.h"
#include "revng/BasicAnalyses/GeneratedCodeBasicInfo.h"
#include "revng/FunctionIsolation/PromoteCSVs.h"
#include "revng/FunctionIsolation/StructInitializers.h"
#include "revng/MFP/MFP.h"
#include "revng/MFP/SetLattices.h"
#include "revng/Pipeline/AllRegistries.h"
#include "revng/Pipeline/Contract.h"
#include "revng/Pipes/FunctionPass.h"
#include "revng/Pipes/Kinds.h"
#include "revng/Pipes/RootKind.h"
#include "revng/Pipes/TaggedFunctionKind.h"
#include "revng/Support/IRHelpers.h"

using namespace llvm;
using namespace MFP;

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

struct WrapperKey {
public:
  Function *Helper = nullptr;

  /// GlobalVariables representing read CPU State Variables sorted by name.
  std::vector<GlobalVariable *> Read;

  /// GlobalVariables representing written CPU State Variables sorted by name.
  std::vector<GlobalVariable *> Written;

private:
  auto tie() const { return std::tie(Helper, Read, Written); }

public:
  bool operator<(const WrapperKey &Other) const { return tie() < Other.tie(); }
};

class PromoteCSVs final : public pipeline::FunctionPassImpl {
private:
  StructInitializers Initializers;
  OpaqueFunctionsPool<StringRef> CSVInitializers;
  std::map<WrapperKey, Function *> Wrappers;
  SetVector<GlobalVariable *> CSVs;
  model::Architecture::Values Architecture;

public:
  PromoteCSVs(ModulePass &Pass, const model::Binary &Binary, Module &M);

public:
  bool prologue() final { return false; }

  bool runOnFunction(const model::Function &ModelFunction,
                     llvm::Function &Function) final;

  bool epilogue() final { return false; }

  static void getAnalysisUsage(llvm::AnalysisUsage &AU) {
    AU.addRequired<GeneratedCodeBasicInfoWrapperPass>();
  }

private:
  void wrap(CallInst *Call,
            const std::vector<GlobalVariable *> &Read,
            const std::vector<GlobalVariable *> &Written);

  void promoteCSVs(Function *F);

  Function *createWrapper(const WrapperKey &Key);

  CSVsUsageMap getUsedCSVs(ArrayRef<CallInst *> CallsRange);

  void wrapCallsToHelpers(Function *F);
};

PromoteCSVs::PromoteCSVs(ModulePass &Pass,
                         const model::Binary &Binary,
                         Module &M) :
  pipeline::FunctionPassImpl(Pass),
  Initializers(&M),
  CSVInitializers(&M, false),
  Architecture(Binary.Architecture()) {

  CSVInitializers.setMemoryEffects(MemoryEffects::readOnly());
  CSVInitializers.addFnAttribute(Attribute::NoUnwind);
  CSVInitializers.addFnAttribute(Attribute::WillReturn);
  CSVInitializers.setTags({ &FunctionTags::OpaqueCSVValue });

  // Record existing initializers
  auto &GCBI = getAnalysis<GeneratedCodeBasicInfoWrapperPass>().getGCBI();
  const auto &PCCSVs = GCBI.programCounterHandler()->pcCSVs();
  const auto &R = llvm::concat<GlobalVariable *const>(GCBI.csvs(), PCCSVs);
  SmallVector<GlobalVariable *> CSVsToSort{ R.begin(), R.end() };
  llvm::sort(CSVsToSort, CompareByName);
  for (GlobalVariable *CSV : CSVsToSort) {
    if (GCBI.isSPReg(CSV))
      continue;

    CSVs.insert(CSV);
    if (auto *F = M.getFunction((Twine("_init_") + CSV->getName()).str()))
      if (FunctionTags::OpaqueCSVValue.isTagOf(F))
        CSVInitializers.record(CSV->getName(), F);
  }
}

// TODO: assign alias information
Function *PromoteCSVs::createWrapper(const WrapperKey &Key) {
  auto &[Helper, Read, Written] = Key;

  LLVMContext &Context = Helper->getParent()->getContext();
  auto *PointeeTy = Helper->getValueType();
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
    NewArguments.push_back(CSV->getValueType());

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
    Builder.CreateStore(createLoad(Builder, CSV),
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

void PromoteCSVs::wrap(CallInst *Call,
                       const std::vector<GlobalVariable *> &Read,
                       const std::vector<GlobalVariable *> &Written) {

  if (Read.size() == 0 and Written.size() == 0)
    return;

  Function *Helper = getCallee(Call);
  revng_assert(Helper != nullptr);

  WrapperKey Key{ Helper, Read, Written };

  // Fetch or create the wrapper
  Function *&HelperWrapper = Wrappers[Key];
  if (HelperWrapper == nullptr)
    HelperWrapper = createWrapper(Key);

  auto *PointeeTy = Helper->getValueType();
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
    NewArguments.push_back(createLoad(Builder, CSV));

  SmallVector<AllocaInst *, 16> WrittenCSVAllocas;
  for (GlobalVariable *CSV : Written) {
    Type *AllocaType = CSV->getValueType();
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
    Builder.CreateStore(createLoad(Builder, Alloca), CSV);

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
  QuickMetadata QMD(F->getParent()->getContext());

  // Get/create initializers
  std::map<Function *, GlobalVariable *> CSVForInitializer;
  std::map<GlobalVariable *, Function *> InitializerForCSV;
  for (GlobalVariable *CSV : CSVs) {
    // Initialize all allocas with opaque, CSV-specific values
    Type *CSVType = CSV->getValueType();
    llvm::StringRef CSVName = CSV->getName();
    using namespace model::Register;
    Values Register = fromCSVName(CSVName, Architecture);
    if (Register != Invalid) {
      auto *Initializer = CSVInitializers.get(CSVName,
                                              CSVType,
                                              {},
                                              Twine("_init_") + CSVName);

      if (not Initializer->hasMetadata("revng.abi_register")) {
        Initializer->setMetadata("revng.abi_register",
                                 QMD.tuple(getName(Register)));
      }

      CSVForInitializer[Initializer] = CSV;
      InitializerForCSV[CSV] = Initializer;
    }
  }

  // Collect existing CSV allocas

  Instruction *NonAlloca = findFirstNonAlloca(&Entry);
  revng_assert(NonAlloca != nullptr);

  IRBuilder<> InitializersBuilder(NonAlloca);
  auto *Separator = InitializersBuilder.CreateUnreachable();
  IRBuilder<> AllocaBuilder(&Entry, Entry.begin());

  // For each GlobalVariable representing a CSV used in F, create a dedicated
  // alloca and save it in CSVMaps.
  std::map<GlobalVariable *, AllocaInst *> CSVAllocas;
  for (GlobalVariable *CSV : CSVs) {
    AllocaInst *Alloca = nullptr;

    auto It = CSVAllocas.find(CSV);
    if (It != CSVAllocas.end()) {
      Alloca = It->second;
    } else {
      // Create the alloca
      Type *CSVType = CSV->getValueType();
      Alloca = AllocaBuilder.CreateAlloca(CSVType, nullptr, CSV->getName());

      // Check if already have an initializer
      Value *Initializer = nullptr;
      auto It = InitializerForCSV.find(CSV);
      if (It != InitializerForCSV.end()) {
        Function *InitializerFunction = InitializerForCSV.at(CSV);
        Initializer = InitializersBuilder.CreateCall(InitializerFunction);
      } else {
        Initializer = CSV->getInitializer();
      }

      // Initialize the alloca
      InitializersBuilder.CreateStore(Initializer, Alloca);
    }

    // Replace users
    replaceAllUsesInFunctionWith(F, CSV, Alloca);
  }

  // Drop separators
  eraseFromParent(Separator);

#ifndef NDEBUG
  auto It = findFirstNonAlloca(&Entry)->getIterator();
  for (Instruction &I : make_range(It, Entry.end()))
    revng_assert(not isa<AllocaInst>(&I));
#endif
}

struct FunctionNodeData {
  Function *F = nullptr;
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

  static LatticeElement applyTransferFunction(Label L,
                                              const LatticeElement &Value) {
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
  // There are three types of calls: calls to helpers tagged by CSAA, calls to
  // isolated functions and other calls that do not touch CPU state. For the
  // former, we ask GCBI to extract the information from metadata. For the
  // latter, we use a monotone framework to compute the set of read/written
  // registers by the callee.  Note that the former is more accurate thanks to
  // CSAA being call-site sensitive.
  std::queue<Function *> Queue;
  for (CallInst *Call : CallsRange) {
    Function *Callee = getCallee(Call);
    if (FunctionTags::Isolated.isTagOf(Callee)) {
      Queue.push(Callee);
    } else if (FunctionTags::Helper.isTagOf(Callee)) {
      CSVsUsage &Usage = Result.Calls[Call];
      auto UsedCSVs = getCSVUsedByHelperCall(Call);
      Usage.Read = UsedCSVs.Read;
      Usage.Written = UsedCSVs.Written;
    } else {
      // Just create the entry
      Result.Calls[Call];
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
          if (!NodeMap.contains(Callee))
            Queue.push(Callee);

          // Insert an edge in the call graph
          auto *CalleeNode = getNode(NodeMap, CallGraph, Callee);
          addEdge(CalleeNode, CallerNode);
        }

        // If there was a memory access targeting a CSV, record it
        if (CSVs.contains(CSV)) {
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
    CSVsUsage.sortByName();

    wrap(Call, CSVsUsage.Read, CSVsUsage.Written);
  }
}

bool PromoteCSVs::runOnFunction(const model::Function &ModelFunction,
                                llvm::Function &Function) {
  // Add tag
  FunctionTags::CSVsPromoted.addTo(&Function);

  if (not Function.isDeclaration()) {
    // Wrap calls to wrappers
    wrapCallsToHelpers(&Function);

    // (Re-)promote CSVs
    promoteCSVs(&Function);
  }

  return true;
}

template<>
char pipeline::FunctionPass<PromoteCSVs>::ID = 0;
using Register = RegisterPass<pipeline::FunctionPass<PromoteCSVs>>;
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
    Manager.add(new pipeline::FunctionPass<PromoteCSVs>());
  }
};

static pipeline::RegisterLLVMPass<PromoteCSVsPipe> Y;
