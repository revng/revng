//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <memory>

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/InstIterator.h"

#include "revng/ADT/GenericGraph.h"
#include "revng/ADT/Queue.h"
#include "revng/BasicAnalyses/CSVGlobals.h"
#include "revng/FunctionIsolation/PromoteCSVs.h"
#include "revng/MFP/MFP.h"
#include "revng/MFP/SetLattices.h"
#include "revng/Model/ProgramCounterHandler.h"
#include "revng/Support/EmitAbort.h"
#include "revng/Support/IRBuilder.h"
#include "revng/Support/IRHelpers.h"

using namespace llvm;
using namespace mfp;

static Logger Log("promote-csvs");

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

class PromoteCSVs {
private:
  llvm::Module &Module;
  Function &LLVMFunction;
  OpaqueFunctionsPool<StringRef> CSVInitializers;
  CSVGlobals Globals;
  std::unique_ptr<ProgramCounterHandler> PCH;
  model::Architecture::Values Architecture;
  const model::NamingConfiguration &Configuration;

  std::map<WrapperKey, llvm::Function *> Wrappers;
  SetVector<GlobalVariable *> CSVs;

public:
  PromoteCSVs(const model::Binary &Binary, llvm::Function &LLVMFunction) :
    Module(*LLVMFunction.getParent()),
    LLVMFunction(LLVMFunction),
    CSVInitializers(&Module, false),
    Globals(Binary, Module),
    PCH(ProgramCounterHandler::fromModule(Binary.Architecture(), &Module)),
    Architecture(Binary.Architecture()),
    Configuration(Binary.Configuration().Naming()) {}

public:
  void run();

private:
  void wrap(CallInst *Call,
            const DenseSet<GlobalVariable *> &Alive,
            const std::vector<GlobalVariable *> &Read,
            const std::vector<GlobalVariable *> &Written);

  void promoteCSVs(Function *F);

  Function *createWrapper(const WrapperKey &Key);

  CSVsUsageMap getUsedCSVs(ArrayRef<CallInst *> CallsRange);

  void wrapCallsToHelpers(Function *F);

  /// CSVs accessed by an instruction of \p F or of a function whose body is
  /// reachable (and thus inlinable) from \p F, ignoring other isolated
  /// functions.
  DenseSet<GlobalVariable *> computeAliveCSVs(Function *F);

  /// A CSV that is neither an ABI register nor alive within \p F can only ever
  /// hold its opaque default value, so it needs no alloca/load/store.
  bool isDeadCSV(GlobalVariable *CSV, const DenseSet<GlobalVariable *> &Alive) {
    return CSVs.contains(CSV) and not Globals.isABIRegister(CSV)
           and not Alive.contains(CSV);
  }
};

void PromoteCSVs::run() {
  CSVInitializers.setMemoryEffects(MemoryEffects::readOnly());
  CSVInitializers.addFnAttribute(Attribute::NoUnwind);
  CSVInitializers.addFnAttribute(Attribute::WillReturn);
  CSVInitializers.setTags({ &FunctionTags::OpaqueCSVValue });

  // Record existing initializers
  const auto &PCCSVs = PCH->pcCSVs();
  const auto &R = llvm::concat<GlobalVariable *const>(Globals.csvs(), PCCSVs);
  SmallVector<GlobalVariable *> CSVsToSort{ R.begin(), R.end() };
  llvm::sort(CSVsToSort, CompareByName);
  for (GlobalVariable *CSV : CSVsToSort) {
    if (Globals.isSPReg(CSV))
      continue;

    CSVs.insert(CSV);
    if (auto *F = Module.getFunction(Configuration.OpaqueCSVValuePrefix()
                                     + CSV->getName().str()))
      if (FunctionTags::OpaqueCSVValue.isTagOf(F))
        CSVInitializers.record(CSV->getName(), F);
  }

  // Add tag
  FunctionTags::CSVsPromoted.addTo(&LLVMFunction);

  if (not LLVMFunction.isDeclaration()) {
    // Wrap calls to wrappers
    wrapCallsToHelpers(&LLVMFunction);

    // (Re-)promote CSVs
    promoteCSVs(&LLVMFunction);
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

  // Dead CSVs are written through a null out-argument: mark null as valid so
  // the optimizer does not treat that store as undefined behavior.
  HelperWrapper->addFnAttr(Attribute::NullPointerIsValid);

  // Copy and extend tags
  auto Tags = FunctionTags::TagsSet::from(Helper);
  Tags.insert(FunctionTags::CSVsAsArgumentsWrapper);
  Tags.set(HelperWrapper);

  auto *Entry = BasicBlock::Create(Context, "", HelperWrapper);

  //
  // Populate the helper wrapper function
  //

  // TODO: is there any useful debug information we could attach to the helper
  //       wrapper?
  revng::IRBuilder Builder(Entry);

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
    Builder.CreateStore(Builder.createLoad(CSV),
                        HelperWrapper->getArg(OutArgument));
    ++OutArgument;
  }

  if (HelperResult->getType()->isVoidTy()) {
    Builder.CreateRetVoid();
  } else {
    Builder.CreateRet(HelperResult);
  }

  // The wrapper inherits the `revng_inline` section from the underlying
  // helper, so a downstream `inline-helpers` invocation will treat it as an
  // inline candidate and consult its `!revng.inline.policy` metadata. By
  // construction a wrapper body always inlinable.
  if (HelperWrapper->getSection() == InlineHelpersSection)
    serializeInliningPolicy(*HelperWrapper,
                            llvm::BitVector(HelperWrapper->arg_size(), false));

  return HelperWrapper;
}

// The wrapper keeps the full CSAA-reported signature (one wrapper per helper),
// but dead CSVs get no per-call alloca/load/store: we pass `undef` for reads
// and a null out-argument for writes, and skip the restore store.
void PromoteCSVs::wrap(CallInst *Call,
                       const DenseSet<GlobalVariable *> &Alive,
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
  revng::IRBuilder AllocaBuilder(&*EntryIt);
  revng::IRBuilder Builder(Call);

  // Initialize the new set of arguments with the old ones
  SmallVector<Value *, 16> NewArguments;
  for (auto &&[Argument, Type] : zip(Call->args(), HelperType->params()))
    NewArguments.push_back(Builder.CreateBitOrPointerCast(Argument, Type));

  // Add arguments read
  for (GlobalVariable *CSV : Read) {
    if (isDeadCSV(CSV, Alive))
      NewArguments.push_back(UndefValue::get(CSV->getValueType()));
    else
      NewArguments.push_back(Builder.createLoad(CSV));
  }

  SmallVector<std::pair<GlobalVariable *, AllocaInst *>, 16> WrittenCSVAllocas;
  for (GlobalVariable *CSV : Written) {
    if (isDeadCSV(CSV, Alive)) {
      auto *Null = ConstantPointerNull::get(cast<PointerType>(CSV->getType()));
      NewArguments.push_back(Null);
    } else {
      auto *OutArgument = AllocaBuilder.CreateAlloca(CSV->getValueType());
      WrittenCSVAllocas.push_back({ CSV, OutArgument });
      NewArguments.push_back(OutArgument);
    }
  }

  // Emit the actual call
  Instruction *Result = Builder.CreateCall(HelperWrapper, NewArguments);
  Result->setDebugLoc(Call->getDebugLoc());
  Call->replaceAllUsesWith(Result);

  // Restore into CSV the live written registers
  for (const auto &[CSV, Alloca] : WrittenCSVAllocas)
    Builder.CreateStore(Builder.createLoad(Alloca), CSV);

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
      llvm::StringRef Prefix = Configuration.OpaqueCSVValuePrefix();
      auto *Initializer = CSVInitializers.get(CSVName,
                                              CSVType,
                                              {},
                                              Prefix + CSVName);

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

  revng::IRBuilder InitializersBuilder(NonAlloca);
  auto *Separator = InitializersBuilder.CreateUnreachable();
  revng::IRBuilder AllocaBuilder(&Entry, Entry.begin());

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
  using ExtraStateType = mfp::NoExtraState;

  static LatticeElement applyTransferFunction(Label L,
                                              const LatticeElement &Value,
                                              mfp::NoExtraState &) {
    return combineValues(L->UsedCSVs, Value);
  }
};

CSVsUsageMap PromoteCSVs::getUsedCSVs(ArrayRef<CallInst *> CallsRange) {
  CSVsUsageMap Result;

  revng_log(Log, "getUsedCSVs");

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
    } else if (FunctionTags::Helper.isTagOf(Callee)
               and AbortHelper.getCall(Call) == std::nullopt) {
      CSVsUsage &Usage = Result.Calls[Call];
      auto UsedCSVs = getCSVUsedByHelperCall(Call);

      if (Log.isEnabled()) {
        Log << "Call " << getName(Call) << " to " << Callee->getName() << ":\n";
        Log << "  Reads:\n";
        for (auto *CSV : UsedCSVs.Read)
          Log << "    " << CSV->getName() << "\n";

        Log << "  Written:\n";
        for (auto *CSV : UsedCSVs.Written)
          Log << "    " << CSV->getName() << "\n";
        Log << DoLog;
      }

      revng_log(Log, "Call " << getName(Call));
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
          if (AbortHelper.getCall(Call).has_value())
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

  auto GetMaximalFixedPoint = getMaximalFixedPoint<UsedRegistersMFI>;
  auto AnalysisResult = GetMaximalFixedPoint({ .Flow = &CallGraph });

  // Populate results set
  for (auto &[Label, Value] : AnalysisResult) {
    auto &FunctionDescriptor = Result.Functions[Label->F];
    for (auto &&[IsWrite, CSV] : Value.OutValue) {
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

DenseSet<GlobalVariable *> PromoteCSVs::computeAliveCSVs(Function *F) {
  // Functions whose body is reachable from F, stopping at declarations and at
  // other isolated functions (which are not inlined into F).
  OnceQueue<Function *> Queue;
  Queue.insert(F);
  while (not Queue.empty()) {
    for (Instruction &I : instructions(Queue.pop())) {
      Function *Callee = getCallee(&I);
      if (Callee != nullptr and not Callee->isDeclaration()
          and not FunctionTags::Isolated.isTagOf(Callee))
        Queue.insert(Callee);
    }
  }
  std::set<Function *> Reachable = Queue.visited();

  // A CSV is alive if one of its users, followed through constant expressions,
  // is an instruction living in a reachable function.
  DenseSet<GlobalVariable *> Alive;
  for (GlobalVariable *CSV : CSVs) {
    OnceQueue<User *> Users;
    for (User *U : CSV->users())
      Users.insert(U);

    while (not Users.empty()) {
      User *U = Users.pop();
      if (auto *I = dyn_cast<Instruction>(U)) {
        if (Reachable.contains(I->getFunction())) {
          Alive.insert(CSV);
          break;
        }
      } else if (isa<Constant>(U)) {
        for (User *TransitiveUser : U->users())
          Users.insert(TransitiveUser);
      }
    }
  }

  return Alive;
}

void PromoteCSVs::wrapCallsToHelpers(Function *F) {
  revng_log(Log, "wrapCallsToHelpers: " << F->getName().str());
  std::vector<CallInst *> ToWrap;

  {
    LoggerIndent Indent(Log);
    for (BasicBlock &BB : *F) {
      for (Instruction &I : BB) {
        if (auto *Call = dyn_cast<CallInst>(&I)) {
          Function *Callee = getCallee(Call);

          // Ignore calls to isolated functions
          if (Callee == nullptr or not needsWrapper(Callee))
            continue;

          revng_log(Log,
                    "Call to " << Callee->getName().str()
                               << " needs a wrapper");
          ToWrap.emplace_back(Call);
        }
      }
    }
  }

  auto UsedCSVs = getUsedCSVs(ToWrap);

  // Compute this before wrapping: wrap() introduces new CSV loads/stores that
  // would otherwise pollute the set of CSVs alive within F.
  DenseSet<GlobalVariable *> Alive = computeAliveCSVs(F);

  for (CallInst *Call : ToWrap) {
    CSVsUsage &Usage = UsedCSVs.get(Call);

    // Sort to ensure compatibility between caller and callee
    Usage.sortByName();

    wrap(Call, Alive, Usage.Read, Usage.Written);
  }
}

namespace revng::pypeline::piperuns {

void PromoteCSVs::runOnLLVMFunction(const model::Function &Function,
                                    llvm::Function &LLVMFunction) {
  ::PromoteCSVs Impl(Binary, LLVMFunction);
  Impl.run();
}

} // namespace revng::pypeline::piperuns
