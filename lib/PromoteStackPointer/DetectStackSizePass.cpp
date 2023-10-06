//
// Copyright (c) rev.ng Labs Srl. See LICENSE.md for details.
//

#include <optional>

#include "llvm/IR/Constants.h"

#include "revng/EarlyFunctionAnalysis/FunctionMetadataCache.h"
#include "revng/Model/IRHelpers.h"
#include "revng/Model/LoadModelPass.h"
#include "revng/Model/VerifyHelper.h"
#include "revng/Pipeline/Context.h"
#include "revng/Pipeline/LLVMContainer.h"
#include "revng/Pipeline/RegisterAnalysis.h"
#include "revng/Pipes/Kinds.h"
#include "revng/Pipes/ModelGlobal.h"
#include "revng/Support/Debug.h"

#include "revng-c/Pipes/Kinds.h"
#include "revng-c/PromoteStackPointer/DetectStackSizePass.h"
#include "revng-c/PromoteStackPointer/InstrumentStackAccessesPass.h"
#include "revng-c/Support/FunctionTags.h"

#include "Helpers.h"

using namespace llvm;
using model::RawFunctionType;

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
  RawFunctionType *Prototype;
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
  std::map<RawFunctionType *, UpperBoundCollector> FunctionTypeStackArguments;
  const size_t CallInstructionPushSize = 0;
  /// Helper for fast model::Type size computation
  model::VerifyHelper VH;

public:
  DetectStackSize(TupleTree<model::Binary> &B) :
    Binary(B),
    CallInstructionPushSize(Architecture::getCallPushSize(B->Architecture())) {}

public:
  void run(FunctionMetadataCache &Cache, Module &M) {
    // Collect information about the stack of each function
    for (llvm::Function &F : FunctionTags::Isolated.functions(&M))
      collectStackBounds(Cache, F);

    // At this point we have populated two data structures:
    //
    // * FunctionTypeStackArguments: we can use it to elect stack arguments size
    // * FunctionsStackInfo: we can use it to elect stack frame size

    // Elect stack arguments size for prototypes
    for (auto &[Prototype, UpperBound] : FunctionTypeStackArguments)
      electStackArgumentsSize(Prototype, UpperBound);

    // Now all prototypes have a definitive stack arguments size, we can elect
    // stack frame size
    for (FunctionStackInfo &FSI : FunctionsStackInfo)
      electFunctionStackFrameSize(FSI);
  }

private:
  void collectStackBounds(FunctionMetadataCache &Cache, Function &F);
  void electStackArgumentsSize(RawFunctionType *Prototype,
                               const UpperBoundCollector &Bound) const;
  void electFunctionStackFrameSize(FunctionStackInfo &FSI);
  std::optional<uint64_t> handleCallSite(const CallSite &CallSite);
};

void DetectStackSize::collectStackBounds(FunctionMetadataCache &Cache,
                                         Function &F) {

  // Obtain model::Function corresponding to this llvm::Function
  MetaAddress Entry = getMetaAddressMetadata(&F, "revng.function.entry");
  model::Function &ModelFunction = Binary->Functions().at(Entry);
  revng_log(Log, "Collecting stack bounds for " << ModelFunction.name().str());
  LoggerIndent<> Indent(Log);

  // Check if this function already has information about stack
  // frame/arguments
  bool NeedsStackFrame = ModelFunction.StackFrameType().empty();
  bool NeedsStackArguments = false;
  model::Type *Prototype = ModelFunction.prototype(*Binary).get();
  RawFunctionType *RawPrototype = nullptr;
  if ((RawPrototype = dyn_cast<RawFunctionType>(Prototype))) {
    revng_assert(RawPrototype->StackArgumentsType().Qualifiers().empty());
    NeedsStackArguments = RawPrototype->StackArgumentsType()
                            .UnqualifiedType()
                            .empty();
  }

  revng_log(Log, "NeedsStackFrame: " << NeedsStackArguments);
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
            auto *Proto = Cache
                            .getCallSitePrototype(*Binary.get(),
                                                  findAssociatedCall(Call),
                                                  &ModelFunction)
                            .get();

            NewCallSite.Prototype = nullptr;
            if (auto *FType = dyn_cast<RawFunctionType>(Proto))
              NewCallSite.Prototype = FType;
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
    FunctionsStackInfo.push_back(std::move(FSI));
  }

  if (NeedsStackArguments and UpperBound.hasValue()) {
    // For stack arguments, we reason prototype-wise, not function-wise.
    // Record for processing later.
    FunctionTypeStackArguments[RawPrototype].record(UpperBound.value());
  }
}

using DSSI = DetectStackSize;
void DSSI::electStackArgumentsSize(RawFunctionType *Prototype,
                                   const UpperBoundCollector &Bound) const {
  revng_assert(Prototype->StackArgumentsType().Qualifiers().empty());
  revng_assert(Prototype->StackArgumentsType().UnqualifiedType().empty());
  revng_assert(Bound.hasValue());

  APInt Value = Bound.value();

  // Upper bound is excluded
  Value -= 1;

  // The return address is not a stack argument
  Value -= CallInstructionPushSize;

  auto Size = Value.getLimitedValue();
  if (Value.sgt(0) and isValidStackSize(Size)) {
    revng_log(Log,
              "electStackArgumentsSize for " << Prototype->ID() << ": "
                                             << Size);
    Prototype->StackArgumentsType() = { createEmptyStruct(*Binary.get(), Size),
                                        {} };
  }
}

void DetectStackSize::electFunctionStackFrameSize(FunctionStackInfo &FSI) {
  model::Function &ModelFunction = FSI.Function;
  revng_log(Log,
            "electFunctionStackFrameSize: "
              << ModelFunction.Entry().toString());
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
      revng_log(Log, "Considering new candidate" << NewCandidate);
      StackSize = std::max(StackSize.value_or(NewCandidate), NewCandidate);
    }
  }

  if (not StackSize and FSI.MaxStackSize) {
    // No call sites, let's just use the extreme memory access
    StackSize = *FSI.MaxStackSize;
  }

  if (StackSize and isValidStackSize(*StackSize)) {
    revng_log(Log, "Final StackSize: " << *StackSize);
    ModelFunction.StackFrameType() = createEmptyStruct(*Binary.get(),
                                                       *StackSize);
  }
}

std::optional<uint64_t>
DetectStackSize::handleCallSite(const CallSite &CallSite) {
  revng_log(Log, "CallSite");
  LoggerIndent<> Indent2(Log);
  if (Log.isEnabled()) {
    if (CallSite.StackSize)
      Log << "StackSize: " << *CallSite.StackSize << DoLog;
    Log << "ID: " << CallSite.Prototype->ID() << DoLog;
  }

  if (not CallSite.StackSize)
    return {};

  const RawFunctionType *Prototype = CallSite.Prototype;
  // TODO: handle CABIFunctionType
  if (Prototype == nullptr)
    return {};

  uint64_t StackArgumentsSize = 0;

  const model::QualifiedType &StackArguments = Prototype->StackArgumentsType();
  revng_assert(StackArguments.Qualifiers().empty());
  if (not StackArguments.UnqualifiedType().empty()) {
    using std::optional;
    optional<uint64_t> MaybeStackArgumentsSize = StackArguments.size(VH);
    revng_assert(MaybeStackArgumentsSize);
    StackArgumentsSize = *MaybeStackArgumentsSize;
  } else {
    revng_log(Log, "No stack arguments");
  }

  revng_log(Log, "StackArgumentsSize: " << StackArgumentsSize);

  int64_t Result = (*CallSite.StackSize - StackArgumentsSize
                    - CallInstructionPushSize);
  if (Result >= 0)
    return static_cast<uint64_t>(Result);
  else
    return {};
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
  StackSizeDetector.run(getAnalysis<FunctionMetadataCachePass>().get(), M);

  return false;
}

void DetectStackSizePass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<LoadModelWrapperPass>();
  AU.addRequired<FunctionMetadataCachePass>();
  AU.setPreservesCFG();
}

char DetectStackSizePass::ID = 0;

using RegisterDSS = RegisterPass<DetectStackSizePass>;
static RegisterDSS R("detect-stack-size", "Detect Stack Size Pass");

class DetectStackSizeAnalysis {
public:
  static constexpr auto Name = "detect-stack-size";

  std::vector<std::vector<pipeline::Kind *>> AcceptedKinds = {
    { &revng::kinds::LiftingArtifactsRemoved }
  };

  llvm::Error run(pipeline::Context &Ctx, pipeline::LLVMContainer &Module) {
    using namespace revng;

    llvm::legacy::PassManager Manager;
    auto Global = llvm::cantFail(Ctx.getGlobal<ModelGlobal>(ModelGlobalName));

    const TupleTree<model::Binary> &Model = Global->get();
    if (Model->Architecture() == model::Architecture::Invalid) {
      return createStringError(inconvertibleErrorCode(),
                               "DetectStackSize analysis require a valid"
                               " Architecture");
    }

    Manager.add(new LoadModelWrapperPass(ModelWrapper(Global->get())));
    Manager.add(new DetectStackSizePass());
    Manager.run(Module.getModule());

    return Error::success();
  }
};

static pipeline::RegisterAnalysis<DetectStackSizeAnalysis> RegisterAnalysis;
