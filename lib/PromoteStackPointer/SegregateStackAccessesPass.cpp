//
// Copyright (c) rev.ng Srls. See LICENSE.md for details.
//

#include <optional>
#include <set>

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"

#include "revng/BasicAnalyses/GeneratedCodeBasicInfo.h"
#include "revng/MFP/MFP.h"
#include "revng/MFP/SetLattices.h"
#include "revng/Model/IRHelpers.h"
#include "revng/Model/LoadModelPass.h"
#include "revng/Model/VerifyHelper.h"

#include "revng-c/PromoteStackPointer/InstrumentStackAccessesPass.h"
#include "revng-c/PromoteStackPointer/SegregateStackAccessesPass.h"
#include "revng-c/Support/FunctionTags.h"
#include "revng-c/Support/IRHelpers.h"

using namespace llvm;

static Logger<> Log("segregate-stack-accesses");

static unsigned getCallPushSize(const model::Binary &Binary) {
  return model::Architecture::getCallPushSize(Binary.Architecture);
}

static MetaAddress getCallerBlockAddress(Instruction *I) {
  return getMetaAddressMetadata(I, "revng.callerblock.start");
}

static bool isCallToIsolatedFunction(Instruction *I) {
  return FunctionTags::CallToLifted.isTagOf(I);
}

static CallInst *findCallTo(Function *F, Function *ToSearch) {
  CallInst *Call = nullptr;
  for (BasicBlock &BB : *F)
    for (Instruction &I : BB)
      if ((Call = getCallTo(&I, ToSearch)))
        return Call;
  return nullptr;
}

template<typename... Types>
static CallInst *
createCall(IRBuilder<> &B, FunctionCallee Callee, Types... Arguments) {

  SmallVector<Value *> ArgumentsValues;
  FunctionType *CalleeType = Callee.getFunctionType();

  unsigned Index = 0;
  auto AddArgument = [&](auto Argument) {
    using ArgumentType = decltype(Argument);
    Value *ArgumentValue = nullptr;
    if constexpr (std::is_same_v<ArgumentType, uint64_t>) {
      auto *ArgumentType = cast<IntegerType>(CalleeType->getParamType(Index));
      ArgumentValue = ConstantInt::get(ArgumentType, Argument);
    } else {
      ArgumentValue = Argument;
    }

    ArgumentsValues.push_back(ArgumentValue);
    ++Index;
  };

  (AddArgument(Arguments), ...);

  return B.CreateCall(Callee, ArgumentsValues);
}

static std::optional<uint64_t>
getStackArgumentsSize(const model::Type *Prototype, model::VerifyHelper &VH) {
  using namespace model;

  if (isa<CABIFunctionType>(Prototype)) {
    return {};
  } else if (auto *RFT = dyn_cast<RawFunctionType>(Prototype)) {
    if (const model::Type *StackStruct = RFT->StackArgumentsType.get()) {
      return StackStruct->size(VH);
    } else {
      return {};
    }
  } else {
    revng_abort("Not a function type");
  }
}

static std::optional<int64_t> getStackOffset(Value *Pointer) {
  auto *PointerInstruction = dyn_cast<Instruction>(skipCasts(Pointer));
  if (PointerInstruction == nullptr)
    return {};

  if (auto *Call = dyn_cast<CallInst>(PointerInstruction)) {
    if (auto *Callee = getCallee(Call)) {
      if (StackOffsetMarker.isTagOf(Callee)) {
        // Check if this is a stack access, i.e., targets an exact range
        unsigned AccessSize = getPointeeSize(Pointer);
        auto MaybeStart = getSignedConstantArg(Call, 1);
        auto MaybeEnd = getSignedConstantArg(Call, 2);

        revng_log(Log, "AccessSize: " << AccessSize);
        revng_log(Log, "MaybeStart: " << (MaybeStart ? *MaybeStart : -1));
        revng_log(Log, "MaybeEnd: " << (MaybeEnd ? *MaybeEnd : -1));

        if (MaybeStart and MaybeEnd
            and *MaybeEnd == *MaybeStart + AccessSize + 1) {
          revng_log(Log, "StackOffset found: " << *MaybeStart);
          return MaybeStart;
        }
      }
    }
  }

  return {};
}

struct StoredByte {
  int64_t StackOffset = 0;
  llvm::StoreInst *Store = nullptr;
  unsigned StoreOffset = 0;

  bool operator<(const StoredByte &Other) const {
    auto ThisTuple = std::tie(StackOffset, Store, StoreOffset);
    auto OtherTuple = std::tie(Other.StackOffset,
                               Other.Store,
                               Other.StoreOffset);
    return ThisTuple < OtherTuple;
  }
};

using Lattice = std::set<StoredByte>;

struct SegregateStackAccessesMFI : public SetUnionLattice<Lattice> {
  using Label = llvm::BasicBlock *;
  using GraphType = llvm::Function *;

  static LatticeElement
  applyTransferFunction(llvm::BasicBlock *BB, const LatticeElement &Value) {
    using namespace llvm;
    revng_log(Log, "Analzying block " << getName(BB));
    LoggerIndent<> Indent(Log);

    LatticeElement StackBytes = Value;

    for (Instruction &I : *BB) {
      if (isCallToIsolatedFunction(&I)) {
        StackBytes.clear();
        continue;
      }

      // Get pointer
      llvm::Value *Pointer = getPointer(&I);

      // If it's not a load/store, pointer is nullptr
      if (Pointer == nullptr)
        continue;

      revng_log(Log, "Analzying instruction " << getName(&I));
      LoggerIndent<> Indent(Log);

      // Get stack offset, if available
      auto MaybeStartStackOffset = getStackOffset(Pointer);
      if (not MaybeStartStackOffset)
        continue;

      int64_t StartStackOffset = *MaybeStartStackOffset;
      unsigned AccessSize = getMemoryAccessSize(&I);
      int64_t EndStackOffset = StartStackOffset + AccessSize;

      // Erase all the existing entries
      // TODO: use lower_bound instead of scanning everything
      StackBytes.erase(StackBytes.lower_bound(StoredByte{ StartStackOffset }),
                       StackBytes.upper_bound(StoredByte{ EndStackOffset }));

      // If it's a store, record all of its bytes
      if (auto *Store = dyn_cast<StoreInst>(&I))
        for (unsigned I = 0; I < AccessSize; ++I)
          StackBytes.insert({ StartStackOffset + I, Store, I });
    }

    return StackBytes;
  }
};

class SegregateStackAccesses {
private:
  using MFIResult = std::map<BasicBlock *,
                             MFP::MFPResult<std::set<StoredByte>>>;

private:
  bool Changed = false;
  const model::Binary &Binary;
  Module &M;
  Function *SSACS = nullptr;
  Function *InitLocalSP = nullptr;
  Function *StackFrameAllocator = nullptr;
  Function *CallStackArgumentsAllocator = nullptr;
  std::set<Instruction *> ToPurge;
  /// Builder for StackArgumentsAllocator calls
  IRBuilder<> SABuilder;
  /// Builder for new stores
  IRBuilder<> B;
  // MFIResult Result;
  model::VerifyHelper VH;
  const size_t CallInstructionPushSize = 0;
  Value *StackPointer = nullptr;
  Type *SPType = nullptr;
  Function *RootFunction = nullptr;
  std::map<Function *, Function *> OldToNew;
  std::set<Function *> FunctionsWithStackArguments;

public:
  SegregateStackAccesses(const model::Binary &Binary,
                         Module &M,
                         Value *StackPointer,
                         Function *RootFunction) :
    Binary(Binary),
    M(M),
    SSACS(M.getFunction("stack_size_at_call_site")),
    InitLocalSP(M.getFunction("revng_init_local_sp")),
    SABuilder(M.getContext()),
    B(M.getContext()),
    CallInstructionPushSize(getCallPushSize(Binary)),
    StackPointer(StackPointer),
    SPType(StackPointer->getType()->getPointerElementType()),
    RootFunction(RootFunction) {

    revng_assert(SSACS != nullptr);
    revng_assert(InitLocalSP != nullptr);

    auto StackAllocatorType = FunctionType::get(SPType, { SPType }, false);
    auto Create = [&StackAllocatorType, &M](StringRef Name) {
      auto *Result = Function::Create(StackAllocatorType,
                                      GlobalValue::ExternalLinkage,
                                      Name,
                                      &M);
      Result->addFnAttr(Attribute::NoUnwind);
      Result->addFnAttr(Attribute::InaccessibleMemOnly);
      Result->addFnAttr(Attribute::WillReturn);
      FunctionTags::AllocatesLocalVariable.addTo(Result);
      FunctionTags::MallocLike.addTo(Result);
      return Result;
    };

    StackFrameAllocator = Create("revng_stack_frame");
    CallStackArgumentsAllocator = Create("revng_call_stack_arguments");
  }

public:
  bool run() {
    addStackArguments(M);

    for (Function &F : FunctionTags::Lifted.functions(&M))
      segregateStackAccesses(F);

    // Purge stores that have been used at least once
    for (Instruction *I : ToPurge)
      eraseFromParent(I);

    // Erase original functions
    for (auto [OldFunction, NewFunction] : OldToNew)
      eraseFromParent(OldFunction);

    // Drop InitLocalSP if it's not used anymore
    if (InitLocalSP->getNumUses() == 0)
      eraseFromParent(InitLocalSP);

    return Changed;
  }

private:
  /// Add a new argument of type SPType to all functions that have stack
  /// arguments
  void addStackArguments(Module &M) {
    // Identify all functions that have stack arguments
    for (Function &F : FunctionTags::Lifted.functions(&M)) {
      MetaAddress Entry = getMetaAddressMetadata(&F, "revng.function.entry");
      const model::Function &ModelFunction = Binary.Functions.at(Entry);
      const model::Type *StackArguments = ModelFunction.Prototype.get();

      std::optional<uint64_t> MaybeStackArgumentsSize;
      if (StackArguments != nullptr)
        MaybeStackArgumentsSize = getStackArgumentsSize(StackArguments, VH);

      if (MaybeStackArgumentsSize)
        FunctionsWithStackArguments.insert(&F);
    }

    // Recreate with extra argument
    for (Function *OldFunction : FunctionsWithStackArguments) {
      Function *NewFunction = changeFunctionType(*OldFunction,
                                                 nullptr,
                                                 { SPType });
      OldToNew[OldFunction] = NewFunction;

      // Drop all tags so we don't go over this again
      OldFunction->clearMetadata();

      // Update the invoke in the root function
      updateCallsInRoot(OldFunction, NewFunction);
    }
  }

  void updateCallsInRoot(Function *OldFunction, Function *NewFunction) {
    // Fix call in root function
    for (CallBase *Caller : callers(OldFunction)) {
      if (Caller->getCaller() == RootFunction) {
        auto *Old = cast<InvokeInst>(Caller);
        B.SetInsertPoint(Old);

        // Prepare new invoke's arguments
        SmallVector<Value *> NewArguments;
        llvm::copy(Old->args(), std::back_inserter(NewArguments));
        auto *PushSize = getSPConstant(CallInstructionPushSize);
        auto *CallStackArguments = B.CreateAdd(B.CreateLoad(StackPointer),
                                               PushSize);
        NewArguments.push_back(CallStackArguments);

        // Create new invoke
        auto *New = B.CreateInvoke(NewFunction,
                                   Old->getNormalDest(),
                                   Old->getUnwindDest(),
                                   NewArguments);
        New->copyMetadata(*Old);
        New->setAttributes(Old->getAttributes());

        // Replace and drop old invoke
        Old->replaceAllUsesWith(New);
        eraseFromParent(Old);
      }
    }
  }

  void segregateStackAccesses(Function &F) {
    std::set<Instruction *> ToPushALAP;

    setInsertPointToFirstNonAlloca(SABuilder, F);

    // Get model::Function
    MetaAddress Entry = getMetaAddressMetadata(&F, "revng.function.entry");
    const model::Function &ModelFunction = Binary.Functions.at(Entry);

    revng_log(Log, "Segregating " << ModelFunction.name().str());
    LoggerIndent<> Indent(Log);

    // Get the last argument, i.e., the previously created stack argument
    bool HasStackArguments = FunctionsWithStackArguments.count(&F) != 0;
    Value *FunctionStackArguments = nullptr;
    if (HasStackArguments) {
      unsigned LastArgumentIndex = F.getFunctionType()->getNumParams() - 1;
      FunctionStackArguments = F.getArg(LastArgumentIndex);
      revng_assert(FunctionStackArguments->getType() == SPType);
    }

    //
    // Analyze stack usage
    //

    // Analysis preparation: split basic blocks at call sites
    {
      std::set<Instruction *> SplitPoints;
      for (BasicBlock &BB : F)
        for (Instruction &I : BB)
          if (isCallToIsolatedFunction(&I))
            SplitPoints.insert(&I);
      for (Instruction *I : SplitPoints)
        I->getParent()->splitBasicBlock(I);
    }

    // Run the analysis
    MFIResult AnalysisResult;
    {
      revng_log(Log, "Running SegregateStackAccessesMFI");
      LoggerIndent<> Indent(Log);
      using SSAMFI = SegregateStackAccessesMFI;
      BasicBlock *Entry = &F.getEntryBlock();
      AnalysisResult = MFP::getMaximalFixedPoint<SSAMFI>({},
                                                         &F,
                                                         {},
                                                         {},
                                                         { Entry });
    }

    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        if (CallInst *SSACSCall = getCallTo(&I, SSACS)) {
          auto *StackArguments = handleCallSite(ModelFunction,
                                                AnalysisResult,
                                                SSACSCall);
          if (StackArguments != nullptr)
            ToPushALAP.insert(StackArguments);
        } else if (isa<LoadInst>(&I) or isa<StoreInst>(&I)) {
          handleMemoryAccess(FunctionStackArguments, &I);
        }
      }
    }

    //
    // Fix stack frame
    //
    adjustStackFrame(ModelFunction, F);

    // Push ALAP all stack arguments allocations
    if (ToPushALAP.size()) {
      DominatorTree DT(F);
      for (Instruction *I : ToPushALAP)
        pushALAP(DT, I);
    }
  }

  /// \returns allocator for this call sites' stack arguments
  CallInst *handleCallSite(const model::Function &ModelFunction,
                           MFIResult &AnalysisResult,
                           CallInst *SSACSCall) {
    revng_log(Log, "Handling call site " << getName(SSACSCall));
    LoggerIndent<> Indent(Log);

    // Get stack size at call site
    auto MaybeStackSize = getSignedConstantArg(SSACSCall, 0);

    // Obtain RawFunctionType
    auto *MD = SSACSCall->getMetadata("revng.callerblock.start");
    revng_assert(MD != nullptr);
    auto *RawPrototype = getCallSitePrototype(Binary,
                                              SSACSCall,
                                              &ModelFunction);
    // TODO: handle CABIFunctionType
    if (RawPrototype == nullptr)
      return nullptr;

    auto MaybeStackArgumentsSize = getStackArgumentsSize(RawPrototype, VH);
    uint64_t StackArgumentsSize = MaybeStackArgumentsSize.value_or(0);
    revng_log(Log, "StackArgumentsSize: " << StackArgumentsSize);

    CallInst *CallStackArguments = nullptr;

    if (StackArgumentsSize != 0) {
      Changed = true;

      // Allocate memory for stack arguments
      CallStackArguments = createCall(SABuilder,
                                      CallStackArgumentsAllocator,
                                      StackArgumentsSize);
      CallStackArguments->setMetadata("revng.callerblock.start", MD);

      // Recreate call with an extra argument
      CallInst *OldCall = findAssociatedCall(SSACSCall);
      revng_assert(OldCall != nullptr);
      B.SetInsertPoint(OldCall);
      SmallVector<Value *> Arguments;
      llvm::copy(OldCall->args(), std::back_inserter(Arguments));
      Arguments.push_back(CallStackArguments);
      auto *NewCall = B.CreateCall(OldToNew.at(OldCall->getCalledFunction()),
                                   Arguments);
      OldCall->replaceAllUsesWith(NewCall);
      eraseFromParent(OldCall);
    }

    if (not MaybeStackSize)
      return CallStackArguments;

    int64_t StackSizeAtCallSite = *MaybeStackSize;

    // Compute stack arguments boundary.
    // Any access below this threshold targets the stack arguments of this
    // call site.
    int64_t Boundary = (-StackSizeAtCallSite + StackArgumentsSize
                        + CallInstructionPushSize);
    revng_log(Log, "Boundary: " << Boundary);

    // Identify all the StoredBytes targeting this call sites' stack
    // arguments
    struct StoreInfo {
      unsigned Count = 0;
      int64_t Offset = 0;
    };
    std::map<StoreInst *, StoreInfo> MarkedStores;
    BasicBlock *BB = SSACSCall->getParent();
    const std::set<StoredByte> &BlockFinalResult = AnalysisResult.at(BB)
                                                     .OutValue;
    for (const StoredByte &Byte : BlockFinalResult) {
      // Mark this store
      if (Byte.StackOffset < Boundary) {
        revng_log(Log,
                  "Byte " << Byte.StoreOffset << " of " << getName(Byte.Store)
                          << " is within Boundary, recording it");
        StoreInfo &Info = MarkedStores[Byte.Store];
        Info.Count += 1;
        Info.Offset = Byte.StackOffset - Byte.StoreOffset;
      }
    }

    // Process MarkedStores
    for (const auto &[Store, Info] : MarkedStores) {
      auto Size = getMemoryAccessSize(Store);
      int64_t StackArgumentsOffset = (Info.Offset + StackSizeAtCallSite
                                      - CallInstructionPushSize);

      revng_log(Log, "Considering " << getName(Store));
      LoggerIndent<> Indent(Log);
      revng_log(Log, "Size: " << Size);
      revng_log(Log, "Info.Count: " << Info.Count);
      revng_log(Log, "Info.Offset: " << Info.Count);
      revng_log(Log, "StackSizeAtCallSite: " << StackSizeAtCallSite);
      revng_log(Log, "StackArgumentsOffset: " << StackArgumentsOffset);

      if (Size == Info.Count) {
        int64_t NegativePushSize = -CallInstructionPushSize;
        if (StackArgumentsOffset == NegativePushSize
            and Size == CallInstructionPushSize) {
          Changed = true;

          // This store targets the saved return address slot, drop it
          revng_log(Log,
                    "This store is saving the return address: we'll drop it");
          ToPurge.insert(Store);
        } else if (CallStackArguments != nullptr) {
          Changed = true;

          replace(Store, CallStackArguments, StackArgumentsOffset);
        }
      } else {
        revng_log(Log,
                  "Warning: " << getName(Store) << " has size " << Size
                              << " but only " << Info.Count << " bytes target "
                              << getName(SSACSCall)
                              << " stack arguments. Ignoring.");
      }
    }

    return CallStackArguments;
  }

  void handleMemoryAccess(Value *FunctionStackArguments, Instruction *I) {
    revng_log(Log, "Handling memory access " << getName(I));
    LoggerIndent<> Indent(Log);

    auto *Pointer = getPointer(I);
    revng_assert(Pointer != nullptr);

    auto MaybeStackOffset = getStackOffset(Pointer);
    if (not MaybeStackOffset)
      return;
    int64_t StackOffset = *MaybeStackOffset;
    revng_log(Log, "StackOffset: " << StackOffset);

    unsigned AccessSize = getMemoryAccessSize(I);
    if (StackOffset >= 0) {
      // We're accessing a stack argumnent
      if (FunctionStackArguments != nullptr) {
        replace(I,
                FunctionStackArguments,
                StackOffset - CallInstructionPushSize);
      }
    } else if (StackOffset + AccessSize > 0) {
      // The access is crossing the boundary
      revng_log(Log,
                "Warning: the memory access "
                  << getName(I) << " has size " << AccessSize
                  << " and stack offset " << StackOffset << "."
                  << " and therefore it partially crosses stack boundaries."
                  << " Ignoring.");
    }
  }

  void adjustStackFrame(const model::Function &ModelFunction, Function &F) {
    //
    // Find call to revng_init_local_sp
    //
    CallInst *Call = findCallTo(&F, InitLocalSP);
    if (Call == nullptr)
      return;

    //
    // Get stack frame size
    //
    std::optional<uint64_t> MaybeStackFrameSize;
    if (const model::Type *T = ModelFunction.StackFrameType.get())
      MaybeStackFrameSize = T->size(VH);

    uint64_t StackFrameSize = MaybeStackFrameSize.value_or(0);

    //
    // Create call and rebase SP0, if StackFrameSize is not zero
    //
    if (StackFrameSize != 0) {
      IRBuilder<> Builder(Call);
      auto *StackFrame = createCall(Builder,
                                    StackFrameAllocator,
                                    StackFrameSize);
      auto *SP0 = Builder.CreateAdd(StackFrame, getSPConstant(StackFrameSize));
      Call->replaceAllUsesWith(SP0);

      // Cleanup revng_init_local_sp
      eraseFromParent(Call);
    }
  }

private:
  /// \name Support functions
  /// \{

  CallInst *findAssociatedCall(CallInst *SSACSCall) const {
    // Look for the actual call in the same block or the next one
    Instruction *I = SSACSCall->getNextNode();
    while (I != SSACSCall) {
      if (isCallToIsolatedFunction(I)) {
        MetaAddress SSACSBlockAddress = getCallerBlockAddress(SSACSCall);
        revng_assert(getCallerBlockAddress(I) == SSACSBlockAddress);
        return cast<CallInst>(I);
      } else if (I->isTerminator()) {
        if (I->getNumSuccessors() != 1)
          return nullptr;
        I = I->getSuccessor(0)->getFirstNonPHI();
      } else {
        I = I->getNextNode();
      }
    }

    return nullptr;
  }

  Constant *getSPConstant(uint64_t Value) const {
    return ConstantInt::get(SPType, Value);
  }

  void replace(Instruction *I, Value *Base, int64_t Offset) {
    ToPurge.insert(I);

    B.SetInsertPoint(I);
    Type *PointerType = getPointer(I)->getType();
    auto *NewOffset = ConstantInt::get(Base->getType(), Offset);
    auto *NewAddress = B.CreateIntToPtr(B.CreateAdd(Base, NewOffset),
                                        PointerType);

    Instruction *NewInstruction = nullptr;
    if (auto *Store = dyn_cast<StoreInst>(I)) {
      NewInstruction = B.CreateStore(Store->getValueOperand(), NewAddress);
    } else if (auto *Load = dyn_cast<LoadInst>(I)) {
      NewInstruction = B.CreateLoad(NewAddress);
    }

    I->replaceAllUsesWith(NewInstruction);
    NewInstruction->copyMetadata(*I);
  }

  /// \}
};

bool SegregateStackAccessesPass::runOnModule(Module &M) {
  // Get model::Binary
  auto &ModelWrapper = getAnalysis<LoadModelWrapperPass>().get();
  const model::Binary &Binary = *ModelWrapper.getReadOnlyModel();

  // Get the stack pointer type
  auto &GCBI = getAnalysis<GeneratedCodeBasicInfoWrapperPass>().getGCBI();

  SegregateStackAccesses SSA(Binary, M, GCBI.spReg(), GCBI.root());
  return SSA.run();
}

void SegregateStackAccessesPass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesCFG();
  AU.addRequired<LoadModelWrapperPass>();
  AU.addRequired<GeneratedCodeBasicInfoWrapperPass>();
}

char SegregateStackAccessesPass::ID = 0;

using RegisterSSA = RegisterPass<SegregateStackAccessesPass>;
static RegisterSSA
  R("segregate-stack-accesses", "Segregate Stack Accesses Pass");
