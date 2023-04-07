//
// Copyright (c) rev.ng Labs Srl. See LICENSE.md for details.
//

#include <optional>
#include <set>

#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"

#include "revng/ABI/FunctionType/Layout.h"
#include "revng/BasicAnalyses/GeneratedCodeBasicInfo.h"
#include "revng/EarlyFunctionAnalysis/FunctionMetadataCache.h"
#include "revng/MFP/MFP.h"
#include "revng/MFP/SetLattices.h"
#include "revng/Model/IRHelpers.h"
#include "revng/Model/LoadModelPass.h"
#include "revng/Model/VerifyHelper.h"
#include "revng/Pipeline/RegisterLLVMPass.h"
#include "revng/Support/IRHelpers.h"
#include "revng/Support/OverflowSafeInt.h"

#include "revng-c/Pipes/Kinds.h"
#include "revng-c/PromoteStackPointer/InstrumentStackAccessesPass.h"
#include "revng-c/PromoteStackPointer/SegregateStackAccessesPass.h"
#include "revng-c/Support/FunctionTags.h"
#include "revng-c/Support/IRHelpers.h"
#include "revng-c/Support/ModelHelpers.h"

using namespace llvm;

static Logger<> Log("segregate-stack-accesses");

static StringRef stripPrefix(StringRef Prefix, StringRef String) {
  revng_assert(String.startswith(Prefix));
  return String.substr(Prefix.size());
}

static unsigned getCallPushSize(const model::Binary &Binary) {
  return model::Architecture::getCallPushSize(Binary.Architecture());
}

static MetaAddress getCallerBlockAddress(Instruction *I) {
  return getMetaAddressMetadata(I, "revng.callerblock.start");
}

static CallInst *findCallTo(Function *F, Function *ToSearch) {
  CallInst *Call = nullptr;
  for (BasicBlock &BB : *F)
    for (Instruction &I : BB)
      if ((Call = getCallTo(&I, ToSearch)))
        return Call;
  return nullptr;
}

static std::optional<int64_t> getStackOffset(Value *Pointer) {
  auto *PointerInstruction = dyn_cast<Instruction>(skipCasts(Pointer));
  if (PointerInstruction == nullptr)
    return {};

  if (auto *Call = dyn_cast<CallInst>(PointerInstruction)) {
    if (auto *Callee = getCallee(Call)) {
      if (FunctionTags::StackOffsetMarker.isTagOf(Callee)) {
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

class StackAccessRedirector {
private:
  using Span = abi::FunctionType::Layout::Argument::StackSpan;

private:
  int64_t BaseOffset;
  std::map<int64_t, std::pair<uint64_t, Value *>> Map;

public:
  StackAccessRedirector(int64_t BaseOffset) : BaseOffset(BaseOffset) {}

  void recordSpan(const Span &Span, Value *BaseAddress) {
    auto Offset = BaseOffset + Span.Offset;
    revng_assert(Map.count(Offset) == 0);
    Map[Offset] = { Span.Size, BaseAddress };

    revng_assert(verify());
  }

public:
  std::optional<std::pair<uint64_t, Value *>>
  computeNewBase(int64_t Offset, uint64_t Size) const {

    revng_log(Log, "Searching for " << Offset << " of size " << Size);

    auto It = Map.upper_bound(Offset);
    if (It == Map.begin()) {
      revng_log(Log, "Not found");
      return std::nullopt;
    }

    --It;

    int64_t SpanStart = It->first;
    uint64_t SpanSize = It->second.first;
    Value *BaseAddress = It->second.second;

    using OSI = OverflowSafeInt<int64_t>;
    auto MaybeSpanEnd = (OSI(SpanStart) + SpanSize).value();
    auto MaybeEnd = (OSI(Offset) + Size).value();
    if (not MaybeSpanEnd or not MaybeEnd or Offset >= *MaybeSpanEnd
        or *MaybeEnd > *MaybeSpanEnd) {
      revng_log(Log, "Not found");
      return std::nullopt;
    }

    revng_log(Log, "Found");
    return { { Offset - SpanStart, BaseAddress } };
  }

public:
  bool verify() const debug_function {
    if (Map.size() >= 2) {
      auto FirstToSemiLast = llvm::make_range(Map.begin(), --Map.end());
      auto SecondToLast = llvm::make_range(++Map.begin(), Map.end());
      for (auto [Current, Next] : llvm::zip(FirstToSemiLast, SecondToLast)) {
        auto CurrentEnd = Current.first
                          + static_cast<int64_t>(Current.second.first);
        auto NextStart = Next.first;
        if (CurrentEnd > NextStart)
          return false;
      }
    }
    return true;
  }

  template<typename T>
  void dump(T &Stream) const {
    for (auto [K, V] : Map) {
      Stream << K << ": [" << V.first << ", " << getName(V.second) << "]\n";
    }
  }

  void dump() const debug_function { dump(dbg); }
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

struct SortByFunction {
  bool operator()(const Instruction *LHS, const Instruction *RHS) const {
    using std::make_pair;
    return make_pair(LHS->getParent(), LHS) < make_pair(RHS->getParent(), RHS);
  }
};

class SegregateStackAccesses {
private:
  using MFIResult = std::map<BasicBlock *,
                             MFP::MFPResult<std::set<StoredByte>>>;

private:
  const model::Binary &Binary;
  Module &M;
  Function *SSACS = nullptr;
  Function *InitLocalSP = nullptr;
  Function *StackFrameAllocator = nullptr;
  Function *CallStackArgumentsAllocator = nullptr;
  std::set<Instruction *> ToPurge;
  /// Builder for StackArgumentsAllocator calls
  IRBuilder<> SABuilder;
  model::VerifyHelper VH;
  const size_t CallInstructionPushSize = 0;
  Type *StackPointerType = nullptr;
  std::map<Function *, Function *> OldToNew;
  std::set<Function *> FunctionsWithStackArguments;
  std::map<Function *, StackAccessRedirector> StackArgumentsRedirectors;
  std::vector<Instruction *> ToPushALAP;

  llvm::Type *PtrSizedInteger;
  OpaqueFunctionsPool<TypePair> AddressOfPool;
  OpaqueFunctionsPool<llvm::Type *> AssignPool;
  OpaqueFunctionsPool<llvm::Type *> LocalVarPool;
  FunctionMetadataCache *Cache;

public:
  SegregateStackAccesses(FunctionMetadataCache &Cache,
                         const model::Binary &Binary,
                         Module &M,
                         GlobalValue *StackPointer) :
    Binary(Binary),
    M(M),
    SSACS(M.getFunction("stack_size_at_call_site")),
    InitLocalSP(M.getFunction("revng_init_local_sp")),
    SABuilder(M.getContext()),
    CallInstructionPushSize(getCallPushSize(Binary)),
    StackPointerType(StackPointer->getValueType()),
    PtrSizedInteger(getPointerSizedInteger(M.getContext(), Binary)),
    AddressOfPool(&M, false),
    AssignPool(&M, false),
    LocalVarPool(&M, false),
    Cache(&Cache) {

    revng_assert(SSACS != nullptr);

    initAddressOfPool(AddressOfPool, &M);
    initAssignPool(AssignPool);
    initLocalVarPool(LocalVarPool);

    auto Create = [&M](StringRef Name, llvm::FunctionType *FType) {
      auto *Result = Function::Create(FType,
                                      GlobalValue::ExternalLinkage,
                                      Name,
                                      &M);
      Result->addFnAttr(Attribute::NoUnwind);
      Result->addFnAttr(Attribute::WillReturn);
      Result->setMemoryEffects(MemoryEffects::readOnly());
      Result->setOnlyAccessesInaccessibleMemory();
      FunctionTags::AllocatesLocalVariable.addTo(Result);
      FunctionTags::MallocLike.addTo(Result);
      FunctionTags::IsRef.addTo(Result);

      return Result;
    };

    StackFrameAllocator = Create("revng_stack_frame",
                                 FunctionType::get(StackPointerType,
                                                   { StackPointerType },
                                                   false));
    llvm::Type *StringPtrType = getStringPtrType(M.getContext());
    CallStackArgumentsAllocator = Create("revng_call_stack_arguments",
                                         FunctionType::get(StackPointerType,
                                                           { StringPtrType,
                                                             StackPointerType },
                                                           false));
  }

public:
  bool run() {
    upgradeDynamicFunctions();
    upgradeLocalFunctions();

    for (Function &F : FunctionTags::StackPointerPromoted.functions(&M)) {
      segregateStackAccesses(*Cache, F);
      FunctionTags::StackAccessesSegregated.addTo(&F);
    }

    pushALAP();

    // Purge stores that have been used at least once
    for (Instruction *I : ToPurge)
      eraseFromParent(I);

    // Erase original functions
    for (auto [OldFunction, NewFunction] : OldToNew)
      eraseFromParent(OldFunction);

    // Drop InitLocalSP if it's not used anymore
    if (InitLocalSP != nullptr)
      if (InitLocalSP->getNumUses() == 0)
        eraseFromParent(InitLocalSP);

    return true;
  }

private:
  auto getPointerTo(const model::QualifiedType &T) const {
    return T.getPointerTo(Binary.Architecture());
  }

  template<typename... Types>
  std::pair<CallInst *, CallInst *>
  createCallWithAddressOf(IRBuilder<> &B,
                          model::QualifiedType &AllocatedType,
                          FunctionCallee Callee,
                          Types... Arguments) {

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

    auto *Call = B.CreateCall(Callee, ArgumentsValues);
    auto CallType = Call->getType();

    // Inject a call to AddressOf
    llvm::Constant *ModelTypeString = serializeToLLVMString(AllocatedType, M);
    auto *AddressOfFunctionType = getAddressOfType(PtrSizedInteger, CallType);
    auto *AddressOfFunction = AddressOfPool.get({ PtrSizedInteger, CallType },
                                                AddressOfFunctionType,
                                                "AddressOf");
    auto *AddressofCall = B.CreateCall(AddressOfFunction,
                                       { ModelTypeString, Call });
    return { Call, AddressofCall };
  }

  void upgradeDynamicFunctions() {
    SmallVector<Function *, 8> Functions;
    for (Function &F : FunctionTags::DynamicFunction.functions(&M))
      Functions.push_back(&F);

    // Identify all functions that have stack arguments
    for (Function *OldFunction : Functions) {
      // TODO: this is not very nice
      auto SymbolName = stripPrefix("dynamic_", OldFunction->getName()).str();
      auto &ImportedFunction = Binary.ImportedDynamicFunctions().at(SymbolName);
      model::TypePath Prototype = ImportedFunction.prototype(Binary);
      auto [NewFunction, Layout] = recreateApplyingModelPrototype(OldFunction,
                                                                  Prototype);
    }
  }

  /// Upgrade all the functions to reflect their model prototype
  void upgradeLocalFunctions() {
    SmallVector<Function *, 8> IsolatedFunctions;
    for (Function &F : FunctionTags::StackPointerPromoted.functions(&M))
      IsolatedFunctions.push_back(&F);

    // Identify all functions that have stack arguments
    for (Function *OldFunction : IsolatedFunctions) {
      bool IsDeclaration = OldFunction->isDeclaration();
      MetaAddress Entry = getMetaAddressMetadata(OldFunction,
                                                 "revng.function.entry");
      revng_assert(Entry.isValid());

      const model::Function &ModelFunction = Binary.Functions().at(Entry);

      //
      // Create new FunctionType
      //
      auto Prototype = ModelFunction.prototype(Binary);
      auto [NewFunction, Layout] = recreateApplyingModelPrototype(OldFunction,
                                                                  Prototype);

      // The rest of this loop handles with the body of the function, ignore if
      // just a declaration
      if (IsDeclaration)
        continue;

      //
      // Map llvm::Argument * to model::Register
      //
      std::map<model::Register::Values, llvm::Argument *> ArgumentToRegister;
      auto ArgumentRegisters = Layout.argumentRegisters();
      for (const auto &[Register, OldArgument] :
           zip(ArgumentRegisters, OldFunction->args()))
        ArgumentToRegister[Register] = &OldArgument;

      //
      // Update references to old arguments
      //
      IRBuilder<> Builder(&NewFunction->getEntryBlock());
      setInsertPointToFirstNonAlloca(Builder, *NewFunction);

      // Create StackAccessRedirector, if required
      StackAccessRedirector *Redirector = nullptr;
      auto IsStackArgument = [](const auto &Argument) -> bool {
        return Argument.Stack.has_value();
      };
      if (llvm::any_of(Layout.Arguments, IsStackArgument)) {
        auto It = StackArgumentsRedirectors.emplace(NewFunction, 0).first;
        Redirector = &It->second;
      }

      auto ModelArguments = llvm::make_range(Layout.Arguments.begin(),
                                             Layout.Arguments.end());

      bool ReturnsAggregate = Layout.returnsAggregateType();
      Value *ReturnValuePointer = nullptr;
      Value *ReturnValueReference = nullptr;
      if (ReturnsAggregate) {
        // Identify the SPTAR and make some sanity checks
        auto &ModelArgument = Layout.Arguments[0];

        // Get call to local variable
        auto *LocalVarFunctionType = getLocalVarType(PtrSizedInteger);
        auto *LocalVarFunction = LocalVarPool.get(PtrSizedInteger,
                                                  LocalVarFunctionType,
                                                  "LocalVariable");

        // Allocate variable for return value
        model::QualifiedType Pointee = stripPointer(ModelArgument.Type);
        llvm::Constant *ReferenceString = serializeToLLVMString(Pointee, M);
        ReturnValueReference = Builder.CreateCall(LocalVarFunction,
                                                  { ReferenceString });

        // Take the address
        auto *T = ReturnValueReference->getType();
        auto *AddressOfFunctionType = getAddressOfType(T, T);
        auto *AddressOfFunction = AddressOfPool.get({ T, T },
                                                    AddressOfFunctionType,
                                                    "AddressOf");
        ReturnValuePointer = Builder.CreateCall(AddressOfFunction,
                                                { ReferenceString,
                                                  ReturnValueReference });

        // Handle the argument pointing to the return value
        if (ModelArgument.Stack) {
          revng_assert(ModelArgument.Registers.size() == 0);
          Redirector->recordSpan(*ModelArgument.Stack + CallInstructionPushSize,
                                 ReturnValuePointer);
        } else {
          // It's in a register
          revng_assert(ModelArgument.Registers.size() == 1);
          Argument *OldArgument = nullptr;
          OldArgument = ArgumentToRegister.at(ModelArgument.Registers[0]);
          OldArgument->replaceAllUsesWith(ReturnValuePointer);
        }

        // Exclude this argument from the list to process
        ModelArguments = llvm::drop_begin(ModelArguments);
      }

      // Handle arguments
      for (auto [ModelArgument, NewArgument] :
           zip(ModelArguments, NewFunction->args())) {

        // Extract from the new argument the old arguments
        unsigned OffsetInNewArgument = 0;
        Type *NewArgumentType = NewArgument.getType();
        unsigned NewArgumentSize = NewArgumentType->getIntegerBitWidth() / 8;

        llvm::Value *ToRecordSpan = nullptr;

        using namespace abi::FunctionType::ArgumentKind;
        if (ModelArgument.Kind == Scalar) {
          revng_assert(ModelArgument.Type.isScalar());
          // Handle scalar argument
          for (model::Register::Values Register : ModelArgument.Registers) {
            Argument *OldArgument = ArgumentToRegister.at(Register);
            Type *OldArgumentType = OldArgument->getType();
            auto OldArgumentSize = OldArgumentType->getIntegerBitWidth() / 8;
            revng_assert(model::Register::getSize(Register) == OldArgumentSize);

            // Compute the shift amount
            unsigned ShiftAmount = shiftAmount(OffsetInNewArgument,
                                               NewArgumentSize,
                                               OldArgumentSize);

            // Shift and trunc
            Value *Shifted = &NewArgument;
            if (ShiftAmount != 0)
              Shifted = Builder.CreateLShr(&NewArgument, ShiftAmount);
            Value *Trunced = Builder.CreateZExtOrTrunc(Shifted,
                                                       OldArgumentType);

            // Replace old argument with the extracted valued
            OldArgument->replaceAllUsesWith(Trunced);

            // Consume size
            OffsetInNewArgument += OldArgumentSize;
          }

          if (ModelArgument.Stack)
            ToRecordSpan = &NewArgument;

        } else if (ModelArgument.Kind == ReferenceToAggregate) {

          // Handle non-scalar argument (passed by pointer)
          llvm::Constant
            *ModelTypeString = serializeToLLVMString(ModelArgument.Type, M);
          auto *AddressOfFunctionType = getAddressOfType(PtrSizedInteger,
                                                         NewArgumentType);
          auto *AddressOfFunction = AddressOfPool.get({ PtrSizedInteger,
                                                        NewArgumentType },
                                                      AddressOfFunctionType,
                                                      "AddressOf");
          auto *AddressOfNewArgument = Builder.CreateCall(AddressOfFunction,
                                                          { ModelTypeString,
                                                            &NewArgument });

          for (model::Register::Values Register : ModelArgument.Registers) {
            Argument *OldArgument = ArgumentToRegister.at(Register);
            Type *OldArgumentPtrType = OldArgument->getType()->getPointerTo();

            // Load value
            Value *ArgumentPointer = computeAddress(Builder,
                                                    OldArgumentPtrType,
                                                    AddressOfNewArgument,
                                                    OffsetInNewArgument);
            Value *ArgumentValue = Builder.CreateLoad(OldArgument->getType(),
                                                      ArgumentPointer);

            // Replace
            OldArgument->replaceAllUsesWith(ArgumentValue);

            // Consume size
            OffsetInNewArgument += model::Register::getSize(Register);
          }

          if (ModelArgument.Stack)
            ToRecordSpan = AddressOfNewArgument;
        }

        if (ToRecordSpan)
          Redirector->recordSpan(*ModelArgument.Stack + CallInstructionPushSize,
                                 ToRecordSpan);
      }

      if (ReturnsAggregate) {
        // Replace return instructions with returning ReturnValueReference
        for (BasicBlock &BB : *NewFunction) {
          if (auto *Ret = dyn_cast<ReturnInst>(BB.getTerminator())) {
            ReturnInst::Create(M.getContext(), ReturnValueReference, Ret);
            Ret->eraseFromParent();
          }
        }
      }
    }
  }

  void segregateStackAccesses(FunctionMetadataCache &Cache, Function &F) {
    if (F.isDeclaration())
      return;

    revng_assert(InitLocalSP != nullptr);

    setInsertPointToFirstNonAlloca(SABuilder, F);

    // Get model::Function
    MetaAddress Entry = getMetaAddressMetadata(&F, "revng.function.entry");
    const model::Function &ModelFunction = Binary.Functions().at(Entry);

    revng_log(Log, "Segregating " << ModelFunction.name().str());
    LoggerIndent<> Indent(Log);

    // Lookup the redirector, if any
    auto It = StackArgumentsRedirectors.find(&F);
    StackAccessRedirector *Redirector = nullptr;
    if (It != StackArgumentsRedirectors.end())
      Redirector = &It->second;

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
          //
          // Handle a call to an isolated function
          //
          handleCallSite(Cache, ModelFunction, AnalysisResult, SSACSCall);
        } else if ((isa<LoadInst>(&I) or isa<StoreInst>(&I))
                   and Redirector != nullptr) {
          //
          // Handle memory access, possibly targeting stack arguments
          //
          handleMemoryAccess(*Redirector, &I);
        }
      }
    }

    //
    // Fix stack frame
    //
    adjustStackFrame(ModelFunction, F);
  }

  void pushALAP() {
    // Push ALAP all stack arguments allocations
    Function *LastFunction = nullptr;
    DominatorTree DT;
    for (Instruction *I : ToPushALAP) {
      if (not I->getNumUses())
        continue;
      Function *F = I->getParent()->getParent();
      if (F != LastFunction) {
        LastFunction = F;
        DT.recalculate(*LastFunction);
      }

      pushInstructionALAP(DT, I);
    }
  }

  void handleCallSite(FunctionMetadataCache &Cache,
                      const model::Function &ModelFunction,
                      MFIResult &AnalysisResult,
                      CallInst *SSACSCall) {
    revng_log(Log, "Handling call site " << getName(SSACSCall));
    LoggerIndent<> Indent(Log);

    //
    // Find call to revng_init_local_sp
    //
    Function *Caller = SSACSCall->getParent()->getParent();
    CallInst *StackPointer = findCallTo(Caller, InitLocalSP);

    // Get stack size at call site
    auto MaybeStackSize = getSignedConstantArg(SSACSCall, 0);

    // Obtain RawFunctionType
    auto *MD = SSACSCall->getMetadata("revng.callerblock.start");
    revng_assert(MD != nullptr);
    auto Prototype = Cache.getCallSitePrototype(Binary,
                                                SSACSCall,
                                                &ModelFunction);
    using namespace abi::FunctionType;
    abi::FunctionType::Layout Layout = Layout::make(*Prototype.get());

    // Find old call instruction
    CallInst *OldCall = findAssociatedCall(SSACSCall);

    if (not OldCall) {
      // We can't find the original call, it might have been DCE'd away
      return;
    }

    IRBuilder<> Builder(OldCall);

    //
    // Map llvm::Argument * to model::Register
    //
    std::map<model::Register::Values, llvm::Value *> ArgumentToRegister;
    auto ArgumentRegisters = Layout.argumentRegisters();
    for (auto [Register, OldArgument] : zip(ArgumentRegisters, OldCall->args()))
      ArgumentToRegister[Register] = OldArgument.get();

    // Check if it's a direct call
    Function *Callee = OldCall->getCalledFunction();
    bool IsDirect = (Callee != nullptr);

    // Obtain or compute the function type for the call
    FunctionType *CalleeType = nullptr;
    Value *CalledValue = nullptr;
    if (IsDirect) {
      CalledValue = OldToNew.at(Callee);
      CalleeType = OldToNew.at(Callee)->getFunctionType();
    } else {
      Type *ReturnType = OldCall->getType();
      CalleeType = &layoutToLLVMFunctionType(Layout, ReturnType);
      CalledValue = Builder.CreateBitCast(OldCall->getCalledOperand(),
                                          CalleeType->getPointerTo());
    }

    SmallVector<llvm::Value *, 4> Arguments;

    StackAccessRedirector Redirector(-MaybeStackSize.value_or(0)
                                     + CallInstructionPushSize);

    bool ReturnsAggregate = Layout.returnsAggregateType();
    SmallVector<llvm::Type *, 8> LLVMArgumentTypes;
    if (ReturnsAggregate) {
      revng_assert(Layout.Arguments.size() > 0);
      uint64_t SPTARSize = *Layout.Arguments[0].Type.size();
      LLVMArgumentTypes.push_back(Builder.getIntNTy(SPTARSize * 8));
    }
    copy(CalleeType->params(), std::back_inserter(LLVMArgumentTypes));

    bool MessageEmitted = false;
    for (auto [LLVMType, ModelArgument] :
         llvm::zip(LLVMArgumentTypes, Layout.Arguments)) {
      model::QualifiedType ArgumentType = ModelArgument.Type;
      uint64_t NewSize = *ArgumentType.size();

      switch (ModelArgument.Kind) {

      case ArgumentKind::Scalar:
      case ArgumentKind::ShadowPointerToAggregateReturnValue: {
        revng_assert(ArgumentType.isScalar());
        Value *Accumulator = ConstantInt::get(LLVMType, 0);
        unsigned OffsetInNewArgument = 0;
        for (auto &Register : ModelArgument.Registers) {
          Value *OldArgument = ArgumentToRegister.at(Register);
          unsigned OldSize = model::Register::getSize(Register);

          Value *Extended = Builder.CreateZExtOrTrunc(OldArgument, LLVMType);

          unsigned ShiftAmount = shiftAmount(OffsetInNewArgument,
                                             NewSize,
                                             OldSize);
          Value *Shifted = Extended;
          if (ShiftAmount != 0)
            Shifted = Builder.CreateLShr(Extended, ShiftAmount);

          Accumulator = Builder.CreateOr(Accumulator, Shifted);

          // Consume size
          OffsetInNewArgument += OldSize;
        }

        if (ModelArgument.Stack and not MaybeStackSize) {
          if (not MessageEmitted) {
            MessageEmitted = true;
            emitMessage(OldCall,
                        "Ignoring stack arguments for this call site: stack "
                        "size at call site unknown");
          }
        } else if (ModelArgument.Stack) {
          revng_assert(ModelArgument.Stack->Size <= 128 / 8);
          unsigned OldSize = ModelArgument.Stack->Size;
          Type *LoadTy = Builder.getIntNTy(OldSize * 8);
          Type *LoadPointerTy = LoadTy->getPointerTo();
          revng_assert(StackPointer != nullptr);

          revng_assert(MaybeStackSize);
          auto ArgumentStackOffset = (-*MaybeStackSize + CallInstructionPushSize
                                      + ModelArgument.Stack->Offset);

          // Compute load address
          Constant *Offset = ConstantInt::get(StackPointer->getType(),
                                              ArgumentStackOffset);
          Value *Address = Builder.CreateAdd(StackPointer, Offset);

          // Load value
          Value *Pointer = Builder.CreateIntToPtr(Address, LoadPointerTy);
          Value *Loaded = Builder.CreateLoad(LoadTy, Pointer);

          // Extend, shift and or in Accumulator
          // Note: here we might truncate too, since certain architectures
          //       report a stack span of 8 bytes but the associated type is
          //       actually 32 bits
          Value *Extended = Builder.CreateZExtOrTrunc(Loaded, LLVMType);

          unsigned ShiftAmount = shiftAmount(OffsetInNewArgument,
                                             NewSize,
                                             OldSize);
          Value *Shifted = Extended;
          if (ShiftAmount != 0)
            Builder.CreateShl(Extended, ShiftAmount);

          Accumulator = Builder.CreateOr(Accumulator, Shifted);
        }

        Arguments.push_back(Accumulator);
      } break;

      case ArgumentKind::ReferenceToAggregate: {
        // Allocate memory for stack arguments
        llvm::Constant *ArgumentType = serializeToLLVMString(ModelArgument.Type,
                                                             M);
        auto [StackArgsCall,
              AddrOfCall] = createCallWithAddressOf(SABuilder,
                                                    ModelArgument.Type,
                                                    CallStackArgumentsAllocator,
                                                    ArgumentType,
                                                    NewSize);

        StackArgsCall->setMetadata("revng.callerblock.start", MD);

        // Record for pushing ALAP. AddrOfCall should be pushed ALAP first to
        // leave slack to StackArgsCall
        ToPushALAP.push_back(AddrOfCall);
        ToPushALAP.push_back(StackArgsCall);

        unsigned OffsetInNewArgument = 0;
        for (auto &Register : ModelArgument.Registers) {
          Value *OldArgument = ArgumentToRegister.at(Register);
          unsigned OldSize = model::Register::getSize(Register);

          Constant *Offset = ConstantInt::get(AddrOfCall->getType(),
                                              OffsetInNewArgument);
          Value *Address = Builder.CreateAdd(AddrOfCall, Offset);

          // Store value
          Type *StorePointerTy = OldArgument->getType()->getPointerTo();
          Value *Pointer = Builder.CreateIntToPtr(Address, StorePointerTy);
          Builder.CreateStore(OldArgument, Pointer);

          // Consume size
          OffsetInNewArgument += OldSize;
        }

        if (ModelArgument.Stack)
          Redirector.recordSpan(*ModelArgument.Stack, AddrOfCall);

        Arguments.push_back(StackArgsCall);
      } break;

      default:
        revng_abort();
      }
    }

    if (Log.isEnabled()) {
      Log << "Redirector data:\n";
      LoggerIndent<> X(Log);
      Redirector.dump(Log);
      Log << DoLog;
    }

    revng_assert(Redirector.verify());

    // Handle SPTAR by dropping the actual argument and saving it for later
    Value *ReturnValuePointer = nullptr;
    if (ReturnsAggregate) {
      revng_assert(Arguments.size() > 0);
      ReturnValuePointer = Arguments[0];
      Arguments.erase(Arguments.begin());
    }

    // Actually create the new call and replace the old one
    auto *NewCall = Builder.CreateCall(CalleeType, CalledValue, Arguments);

    if (ReturnsAggregate) {
      // Perform a couple of safety checks
      revng_assert(Layout.Arguments.size() > 0);
      auto &Argument = Layout.Arguments[0];
      using namespace abi::FunctionType::ArgumentKind;
      revng_assert(Argument.Kind == ShadowPointerToAggregateReturnValue);

      // Obtain the SPTAR value
      revng_assert(ReturnValuePointer != nullptr);

      // Extract return type by stripping the pointer qualifier from SPTAR
      model::QualifiedType ReturnType = stripPointer(Argument.Type);

      // Make reference out of ReturnValuePointer
      Type *T = ReturnValuePointer->getType();
      Function *GetModelGEPFunction = getModelGEP(M, T, T);
      auto *BaseTypeConstantStrPtr = serializeToLLVMString(ReturnType, M);
      Value *ReturnValueReference = Builder.CreateCall(GetModelGEPFunction,
                                                       { BaseTypeConstantStrPtr,
                                                         ReturnValuePointer });

      auto *ReturnValueType = ReturnValueReference->getType();
      auto *AssignFnType = getAssignFunctionType(NewCall->getType(),
                                                 ReturnValueType);
      Function *AssignFunction = AssignPool.get(NewCall->getType(),
                                                AssignFnType,
                                                "Assign");
      Builder.CreateCall(AssignFunction, { NewCall, ReturnValueReference });
    }

    NewCall->copyMetadata(*OldCall);
    OldCall->replaceAllUsesWith(NewCall);
    eraseFromParent(OldCall);
    revng_assert(CalleeType->getPointerTo() == CalledValue->getType());

    if (not MaybeStackSize)
      return;

    int64_t StackSizeAtCallSite = *MaybeStackSize;

    // Identify all the StoredBytes targeting this call sites' stack
    // arguments
    struct StoreInfo {
      unsigned Count = 0;
      int64_t Offset = 0;
    };
    std::map<StoreInst *, StoreInfo> Stores;
    BasicBlock *BB = SSACSCall->getParent();
    const std::set<StoredByte> &BlockFinalResult = AnalysisResult.at(BB)
                                                     .OutValue;
    for (const StoredByte &Byte : BlockFinalResult) {
      StoreInfo &Info = Stores[Byte.Store];
      Info.Count += 1;
      Info.Offset = Byte.StackOffset - Byte.StoreOffset;
    }

    // Process MarkedStores
    for (const auto &[Store, Info] : Stores) {
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

      if (Size != Info.Count) {
        revng_log(Log,
                  "Warning: " << getName(Store) << " has size " << Size
                              << " but only " << Info.Count << " bytes target "
                              << getName(SSACSCall)
                              << " stack arguments. Ignoring.");
        continue;
      }

      // OK, this call site owns this store entirely

      // Check if we're writing to the return address
      int64_t NegativePushSize = -CallInstructionPushSize;
      bool TargetsReturnAddress = (StackArgumentsOffset == NegativePushSize
                                   and Size == CallInstructionPushSize);

      if (TargetsReturnAddress) {
        // This store targets the saved return address slot, drop it
        revng_log(Log,
                  "This store is saving the return address: we'll drop it");
        ToPurge.insert(Store);
      } else if (auto NewBase = Redirector.computeNewBase(Info.Offset, Size)) {
        // This ends up in a stack argument
        replace(Store, NewBase->second, NewBase->first);
      }
    }
  }

  void
  handleMemoryAccess(const StackAccessRedirector &Redirector, Instruction *I) {
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
    auto NewBase = Redirector.computeNewBase(StackOffset, AccessSize);
    if (NewBase)
      replace(I, NewBase->second, NewBase->first);
  }

  void adjustStackFrame(const model::Function &ModelFunction, Function &F) {
    //
    // Find call to revng_init_local_sp
    //
    CallInst *Call = findCallTo(&F, InitLocalSP);
    if (Call == nullptr or not ModelFunction.StackFrameType().isValid())
      return;

    //
    // Get stack frame size
    //
    std::optional<uint64_t> MaybeStackFrameSize;
    if (const model::Type *T = ModelFunction.StackFrameType().get())
      MaybeStackFrameSize = T->size(VH);

    uint64_t StackFrameSize = MaybeStackFrameSize.value_or(0);

    //
    // Create call and rebase SP0, if StackFrameSize is not zero
    //
    if (StackFrameSize != 0) {
      IRBuilder<> Builder(Call);
      model::QualifiedType StackFrameType(ModelFunction.StackFrameType(), {});
      auto [_, StackFrameCall] = createCallWithAddressOf(Builder,
                                                         StackFrameType,
                                                         StackFrameAllocator,
                                                         StackFrameSize);
      auto *SP0 = Builder.CreateAdd(StackFrameCall,
                                    getSPConstant(StackFrameSize));
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
    return ConstantInt::get(StackPointerType, Value);
  }

  Value *computeAddress(IRBuilder<> &B,
                        Type *PointerType,
                        Value *Base,
                        int64_t Offset) const {
    auto *NewOffset = ConstantInt::get(Base->getType(), Offset);
    return B.CreateIntToPtr(B.CreateAdd(Base, NewOffset), PointerType);
  }

  void replace(Instruction *I, Value *Base, int64_t Offset) {
    ToPurge.insert(I);

    IRBuilder<> B(I);
    auto *NewAddress = computeAddress(B,
                                      getPointer(I)->getType(),
                                      Base,
                                      Offset);

    Instruction *NewInstruction = nullptr;
    if (auto *Store = dyn_cast<StoreInst>(I)) {
      NewInstruction = B.CreateStore(Store->getValueOperand(), NewAddress);
    } else if (auto *Load = dyn_cast<LoadInst>(I)) {
      NewInstruction = B.CreateLoad(I->getType(), NewAddress);
    }

    I->replaceAllUsesWith(NewInstruction);
    NewInstruction->copyMetadata(*I);
  }

private:
  std::pair<llvm::Function *, abi::FunctionType::Layout>
  recreateApplyingModelPrototype(Function *OldFunction,
                                 const model::TypePath &Prototype) {
    auto Layout = abi::FunctionType::Layout::make(Prototype);

    Type *ReturnType = nullptr;

    bool ReturnsAggregate = Layout.returnsAggregateType();
    if (ReturnsAggregate) {
      // Ensure the return type is correct
      auto ReturnValuesCount = Layout.returnValueRegisterCount();
      if (ReturnValuesCount == 0) {
        revng_assert(OldFunction->getReturnType()->isVoidTy());
      } else if (ReturnValuesCount == 1) {
        revng_assert(OldFunction->getReturnType() == StackPointerType);
      } else {
        revng_abort("Unexpected number of return values");
      }

      ReturnType = StackPointerType;
    } else {
      ReturnType = OldFunction->getReturnType();
    }

    FunctionType &NewType = layoutToLLVMFunctionType(Layout, ReturnType);

    //
    // Steal the body
    //
    Function &NewFunction = moveToNewFunctionType(*OldFunction, NewType);

    // Record the old-to-new mapping
    OldToNew[OldFunction] = &NewFunction;

    // Drop all tags so we don't go over this again
    OldFunction->clearMetadata();

    return { &NewFunction, Layout };
  }

  llvm::FunctionType &
  layoutToLLVMFunctionType(const abi::FunctionType::Layout &Layout,
                           Type *ReturnType) const {
    using namespace abi::FunctionType;
    SmallVector<Type *> FunctionArguments;
    for (const Layout::Argument &Argument : Layout.Arguments) {
      model::QualifiedType ArgumentType = Argument.Type;
      using namespace abi::FunctionType::ArgumentKind;

      switch (Argument.Kind) {
      case ShadowPointerToAggregateReturnValue:
        continue;
        break;
      case ReferenceToAggregate:
        ArgumentType = getPointerTo(ArgumentType);
        break;
      case Scalar:
        // Do nothing
        break;
      default:
        revng_abort();
      }

      auto *LLVMType = getLLVMTypeForScalar(M.getContext(), ArgumentType);
      FunctionArguments.push_back(LLVMType);
    }

    return *FunctionType::get(ReturnType, FunctionArguments, false);
  }

  unsigned
  shiftAmount(unsigned Offset, unsigned NewSize, unsigned OldSize) const {
    if (NewSize >= OldSize)
      return 0;
    if (model::Architecture::isLittleEndian(Binary.Architecture())) {
      return Offset * 8;
    } else {
      return (NewSize - Offset - OldSize) * 8;
    }
  }

  /// \}
};

bool SegregateStackAccessesPass::runOnModule(Module &M) {
  // Get model::Binary
  auto &ModelWrapper = getAnalysis<LoadModelWrapperPass>().get();
  const model::Binary &Binary = *ModelWrapper.getReadOnlyModel();

  // Get the stack pointer type
  auto &GCBI = getAnalysis<GeneratedCodeBasicInfoWrapperPass>().getGCBI();

  SegregateStackAccesses SSA(getAnalysis<FunctionMetadataCachePass>().get(),
                             Binary,
                             M,
                             GCBI.spReg());
  return SSA.run();
}

void SegregateStackAccessesPass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesCFG();
  AU.addRequired<LoadModelWrapperPass>();
  AU.addRequired<GeneratedCodeBasicInfoWrapperPass>();
  AU.addRequired<FunctionMetadataCachePass>();
}

char SegregateStackAccessesPass::ID = 0;

static constexpr const char *Flag = "segregate-stack-accesses";

using Reg = RegisterPass<SegregateStackAccessesPass>;
static Reg R(Flag, "Segregate Stack Accesses Pass");

struct SegregateStackAccessesPipe {
  static constexpr auto Name = Flag;

  std::vector<pipeline::ContractGroup> getContract() const {
    using namespace pipeline;
    using namespace revng::kinds;
    return { ContractGroup::transformOnlyArgument(StackPointerPromoted,
                                                  StackAccessesSegregated,
                                                  InputPreservation::Erase) };
  }

  void registerPasses(legacy::PassManager &Manager) {
    Manager.add(new SegregateStackAccessesPass());
  }
};

static pipeline::RegisterLLVMPass<SegregateStackAccessesPipe> Y;
