//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <optional>

#include "llvm/IR/Constants.h"

#include "revng/ABI/FunctionType/Layout.h"
#include "revng/Model/IRHelpers.h"
#include "revng/Model/LoadModelPass.h"
#include "revng/Model/VerifyHelper.h"
#include "revng/Pipeline/Context.h"
#include "revng/Pipeline/LLVMContainer.h"
#include "revng/Pipeline/RegisterAnalysis.h"
#include "revng/Pipes/Kinds.h"
#include "revng/Pipes/ModelGlobal.h"
#include "revng/Support/Debug.h"
#include "revng/Support/FunctionTags.h"

#include "revng-c/PromoteStackPointer/DetectStackSizePass.h"
#include "revng-c/PromoteStackPointer/InstrumentStackAccessesPass.h"

#include "Helpers.h"

using namespace llvm;
using model::RawFunctionDefinition;

static Logger<> Log("detect-stack-size");

static bool isValidStackSize(uint64_t Size) {
  return 0 < Size and Size < 10 * 1024 * 1024;
}

/// \note The bound collection is performed using signed comparisons
template<bool IsUpperBound>
class BoundCollector {
private:
  std::optional<APInt> Value;

public:
  bool hasValue() const { return Value.has_value(); }

  const APInt &value() const {
    revng_assert(hasValue());
    return *Value;
  }

public:
  void record(const APInt &NewValue) {
    if (not Value.has_value()) {
      Value = NewValue;
    } else {
      if constexpr (IsUpperBound) {
        if (NewValue.sgt(*Value))
          Value = NewValue;
      } else {
        if (NewValue.slt(*Value))
          Value = NewValue;
      }
    }
  }
};

using UpperBoundCollector = BoundCollector<true>;
using LowerBoundCollector = BoundCollector<false>;

template<bool IsUpper>
static void setBound(BoundCollector<IsUpper> &BoundCollector, Value *V) {
  if (isa<UndefValue>(V))
    return;

  const APInt &Bound = cast<ConstantInt>(V)->getValue();
  if (Bound.isMaxSignedValue() or Bound.isMinSignedValue())
    return;

  BoundCollector.record(Bound);
}

struct CallSite {
  std::optional<uint64_t> StackSize;
  model::TypeDefinition::Key CallType{};
};

class FunctionStackInfo {
public:
  model::Function &Function;
  std::optional<uint64_t> MaxStackSize;
  std::vector<CallSite> CallSites;

public:
  FunctionStackInfo(model::Function &Function) : Function(Function) {}
};

namespace Architecture = model::Architecture;

class DetectStackSize {
private:
  TupleTree<model::Binary> &Binary;
  std::vector<FunctionStackInfo> FunctionsStackInfo;
  std::map<RawFunctionDefinition *, UpperBoundCollector>
    FunctionTypeStackArguments;
  const size_t CallInstructionPushSize = 0;
  /// Helper for fast model::TypeDefinition size computation
  model::VerifyHelper VH;

public:
  DetectStackSize(TupleTree<model::Binary> &B) :
    Binary(B),
    CallInstructionPushSize(Architecture::getCallPushSize(B->Architecture())) {}

public:
  void run(Module &M) {
    // Collect information about the stack of each function
    for (llvm::Function &F : FunctionTags::Isolated.functions(&M))
      collectStackBounds(F);

    // At this point we have populated two data structures:
    //
    // * FunctionTypeStackArguments: we can use it to elect stack arguments size
    // * FunctionsStackInfo: we can use it to elect stack frame size

    // Elect stack arguments size for prototypes
    for (auto [Prototype, UpperBound] : FunctionTypeStackArguments)
      electStackArgumentsSize(*Prototype, UpperBound);

    // Now all prototypes have a definitive stack arguments size, we can elect
    // stack frame size
    for (FunctionStackInfo &FSI : FunctionsStackInfo)
      electFunctionStackFrameSize(FSI);
  }

private:
  void collectStackBounds(Function &F);
  void electStackArgumentsSize(RawFunctionDefinition &Prototype,
                               const UpperBoundCollector &Bound) const;
  void electFunctionStackFrameSize(FunctionStackInfo &FSI);
  std::optional<uint64_t> handleCallSite(const CallSite &CallSite);
};

void DetectStackSize::collectStackBounds(Function &F) {

  // Obtain model::Function corresponding to this llvm::Function
  MetaAddress Entry = getMetaAddressMetadata(&F, "revng.function.entry");
  model::Function &ModelFunction = Binary->Functions().at(Entry);
  revng_log(Log, "Collecting stack bounds for " << ModelFunction.name().str());
  LoggerIndent<> Indent(Log);

  // Check if this function already has information about stack
  // frame/arguments
  bool NeedsStackFrame = ModelFunction.StackFrameType().isEmpty();
  bool NeedsStackArguments = false;
  auto &Prototype = *Binary->prototypeOrDefault(ModelFunction.prototype());

  // We only upgrade the stack size of RawFunctionDefinition
  RawFunctionDefinition *RawPrototype = nullptr;
  if ((RawPrototype = llvm::dyn_cast<RawFunctionDefinition>(&Prototype)))
    NeedsStackArguments = RawPrototype->StackArgumentsType().isEmpty();

  revng_log(Log, "NeedsStackFrame: " << NeedsStackFrame);
  revng_log(Log, "NeedsStackArguments: " << NeedsStackArguments);

  if (not NeedsStackFrame and not NeedsStackArguments)
    return;

  FunctionStackInfo FSI(ModelFunction);

  // Go over all stack accesses and record the extremes
  UpperBoundCollector UpperBound;
  LowerBoundCollector LowerBound;
  for (llvm::BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      if (auto *Call = dyn_cast<CallInst>(&I)) {
        auto *CalledValue = skipCasts(Call->getCalledOperand());
        if (auto *CalledFunction = dyn_cast<llvm::Function>(CalledValue)) {
          if (FunctionTags::StackOffsetMarker.isTagOf(CalledFunction)) {
            revng_log(Log, "Considering stack offset marker " << getName(Call));
            // This is a call to a stack_offset function, let's record the
            // offset
            setBound(LowerBound, Call->getArgOperand(1));
            setBound(UpperBound, Call->getArgOperand(2));
          } else if (CalledFunction->getName() == "stack_size_at_call_site") {
            revng_log(Log, "Considering call site " << getName(Call));
            auto &NewCallSite = FSI.CallSites.emplace_back();

            // Try to get the stack offset
            Value *StackOffsetArgument = Call->getArgOperand(0);
            if (auto *Offset = dyn_cast<ConstantInt>(StackOffsetArgument))
              NewCallSite.StackSize = Offset->getLimitedValue();

            // Get the prototype
            const auto &Proto = *getCallSitePrototype(*Binary.get(),
                                                      findAssociatedCall(Call));
            NewCallSite.CallType = Proto.key();
          }
        }
      }
    }
  }

  if (NeedsStackFrame) {
    if (LowerBound.hasValue()) {
      int64_t Size = -LowerBound.value().getLimitedValue();
      if (Size > 0)
        FSI.MaxStackSize = Size;
    }

    // Record FSI for later processing
    revng_log(Log, "Registering function");
    FunctionsStackInfo.push_back(std::move(FSI));
  }

  if (NeedsStackArguments and UpperBound.hasValue()) {
    // For stack arguments, we reason prototype-wise, not function-wise.
    // Record for processing later.
    FunctionTypeStackArguments[RawPrototype].record(UpperBound.value());
  }
}

using DSSI = DetectStackSize;
void DSSI::electStackArgumentsSize(RawFunctionDefinition &Prototype,
                                   const UpperBoundCollector &Bound) const {
  revng_assert(Prototype.StackArgumentsType().isEmpty());
  revng_assert(Bound.hasValue());

  APInt Value = Bound.value();

  // Upper bound is excluded
  Value -= 1;

  // The return address is not a stack argument
  Value -= CallInstructionPushSize;

  auto Size = Value.getLimitedValue();
  if (Value.sgt(0) and isValidStackSize(Size)) {
    revng_log(Log,
              "electStackArgumentsSize for " << Prototype.ID() << ": " << Size);

    auto EmptyStruct = Binary->makeStructDefinition(Size).second;
    Prototype.StackArgumentsType() = std::move(EmptyStruct);
  }
}

void DetectStackSize::electFunctionStackFrameSize(FunctionStackInfo &FSI) {
  model::Function &ModelFunction = FSI.Function;
  revng_log(Log, "electFunctionStackFrameSize: " << ModelFunction.name().str());
  LoggerIndent<> Indent(Log);

  if (FSI.MaxStackSize)
    revng_log(Log, "MaxStackSize: " << *FSI.MaxStackSize);

  std::optional<uint64_t> StackSize;

  // If we have call site, the stack size is the highest value of the
  // following expression:
  //
  //    StackSizeAtCallSite - CallSiteStackArgumentsSize
  //
  for (const CallSite &CallSite : FSI.CallSites) {
    auto MaybeNewCandidate = handleCallSite(CallSite);
    if (MaybeNewCandidate) {
      uint64_t NewCandidate = *MaybeNewCandidate;
      revng_log(Log, "Considering new candidate: " << NewCandidate);
      StackSize = std::max(StackSize.value_or(NewCandidate), NewCandidate);
    }
  }

  if (not StackSize and FSI.MaxStackSize) {
    // No call sites, let's just use the extreme memory access
    StackSize = *FSI.MaxStackSize;
  }

  if (StackSize and isValidStackSize(*StackSize)) {
    revng_log(Log, "Final StackSize: " << *StackSize);

    auto EmptyStruct = Binary->makeStructDefinition(*StackSize).second;
    ModelFunction.StackFrameType() = std::move(EmptyStruct);
  }
}

std::optional<uint64_t>
DetectStackSize::handleCallSite(const CallSite &CallSite) {
  revng_log(Log, "CallSite");
  LoggerIndent<> Indent2(Log);
  revng_log(Log, "CallSite.StackSize: " << CallSite.StackSize.value_or(0));
  revng_log(Log, "StackArgumentsSize: " << std::get<0>(CallSite.CallType));

  if (not CallSite.StackSize)
    return std::nullopt;

  using namespace abi::FunctionType;
  uint64_t StackArgumentSize = 0;
  for (auto &Prototype = *Binary->TypeDefinitions().at(CallSite.CallType).get();
       Layout::Argument & Argument : Layout::make(Prototype).Arguments) {
    if (Argument.Stack.has_value()) {
      StackArgumentSize = std::max(StackArgumentSize,
                                   Argument.Stack->Offset
                                     + Argument.Stack->Size);
    }
  }
  revng_log(Log, "StackArgumentSize: " << StackArgumentSize);

  int64_t Result = (*CallSite.StackSize - StackArgumentSize
                    - CallInstructionPushSize);

  revng_log(Log, "Result: " << Result);

  if (Result >= 0)
    return static_cast<uint64_t>(Result);
  else
    return std::nullopt;
}

bool DetectStackSizePass::runOnModule(Module &M) {
  //
  // Overview:
  //
  // * Collect stack argument boundaries on a per-prototype basis
  // * Collect stack frame boundaries for each function
  // * Collect stack frame size at each call site and record it along with the
  //   prototype
  // * Elect stack arguments size for each prototype
  // * Elect stack frame size for each function as the maximum stack size,
  //   considering stack sizes at each call site *minus* the size of stack
  //   arguments for that call site
  //
  auto &ModelWrapper = getAnalysis<LoadModelWrapperPass>().get();
  TupleTree<model::Binary> &Binary = ModelWrapper.getWriteableModel();

  DetectStackSize StackSizeDetector(Binary);
  StackSizeDetector.run(M);

  return false;
}

void DetectStackSizePass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<LoadModelWrapperPass>();
  AU.setPreservesCFG();
}

char DetectStackSizePass::ID = 0;

using RegisterDSS = RegisterPass<DetectStackSizePass>;
static RegisterDSS R("detect-stack-size", "Detect Stack Size Pass");

class DetectStackSizeAnalysis {
public:
  static constexpr auto Name = "detect-stack-size";

  std::vector<std::vector<pipeline::Kind *>> AcceptedKinds = {
    { &revng::kinds::StackPointerPromoted }
  };

  llvm::Error run(pipeline::ExecutionContext &EC,
                  pipeline::LLVMContainer &Module) {
    using namespace revng;

    llvm::legacy::PassManager Manager;
    auto &Global = getWritableModelFromContext(EC);

    if (Global->Architecture() == model::Architecture::Invalid) {
      return createStringError(inconvertibleErrorCode(),
                               "DetectStackSize analysis require a valid"
                               " Architecture");
    }

    Manager.add(new LoadModelWrapperPass(ModelWrapper(Global)));
    Manager.add(new DetectStackSizePass());
    Manager.run(Module.getModule());

    return Error::success();
  }
};

static pipeline::RegisterAnalysis<DetectStackSizeAnalysis> RegisterAnalysis;
