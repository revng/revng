//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <optional>
#include <set>

#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Verifier.h"

#include "revng/ABI/FunctionType/Layout.h"
#include "revng/ABI/ModelHelpers.h"
#include "revng/BasicAnalyses/GeneratedCodeBasicInfo.h"
#include "revng/LocalVariables/LocalVariableBuilder.h"
#include "revng/MFP/MFP.h"
#include "revng/MFP/SetLattices.h"
#include "revng/Model/IRHelpers.h"
#include "revng/Model/LoadModelPass.h"
#include "revng/Model/NameBuilder.h"
#include "revng/Model/VerifyHelper.h"
#include "revng/Pipeline/RegisterLLVMPass.h"
#include "revng/Pipes/FunctionPass.h"
#include "revng/Pipes/Kinds.h"
#include "revng/PromoteStackPointer/InstrumentStackAccessesPass.h"
#include "revng/Support/FunctionTags.h"
#include "revng/Support/Generator.h"
#include "revng/Support/IRHelpers.h"
#include "revng/Support/OverflowSafeInt.h"

#include "Helpers.h"

using namespace llvm;

static Logger<> Log("segregate-stack-accesses");

static Value *createAdd(IRBuilder<> &B, Value *V, uint64_t Addend) {
  return B.CreateAdd(V, ConstantInt::get(V->getType(), Addend));
}

static StringRef stripPrefix(StringRef Prefix, StringRef String) {
  revng_assert(String.startswith(Prefix));
  return String.substr(Prefix.size());
}

static unsigned getCallPushSize(const model::Binary &Binary) {
  return model::Architecture::getCallPushSize(Binary.Architecture());
}

static auto snapshot(auto &&Range) {
  SmallVector<std::decay_t<decltype(*Range.begin())>, 16> Result;
  llvm::copy(Range, std::back_inserter(Result));
  return Result;
}

static unsigned getBitOffsetAt(StructType *Struct, unsigned TargetFieldIndex) {
  unsigned Result = 0;
  for (unsigned FieldIndex = 0; FieldIndex < TargetFieldIndex; ++FieldIndex) {
    Result += Struct->getTypeAtIndex(FieldIndex)->getIntegerBitWidth();
  }
  return Result;
}

static CallInst *findCallTo(Function *F, Function *ToSearch) {
  CallInst *Call = nullptr;
  for (BasicBlock &BB : *F)
    for (Instruction &I : BB)
      if ((Call = getCallTo(&I, ToSearch)))
        return Call;
  return nullptr;
}

static std::optional<int64_t> getStackOffset(Instruction *I) {
  auto *Pointer = getPointer(I);

  auto *PointerInstruction = dyn_cast<Instruction>(skipCasts(Pointer));
  if (PointerInstruction == nullptr)
    return {};

  if (auto *Call = dyn_cast<CallInst>(PointerInstruction)) {
    if (auto *Callee = getCallee(Call)) {
      if (FunctionTags::StackOffsetMarker.isTagOf(Callee)) {
        // Check if this is a stack access, i.e., targets an exact range
        unsigned AccessSize = getMemoryAccessSize(I);
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
    revng_assert(BaseAddress->getType()->isIntegerTy());
    auto Offset = BaseOffset + Span.Offset;
    revng_assert(!Map.contains(Offset));
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
      for (auto &&[Current, Next] : llvm::zip(FirstToSemiLast, SecondToLast)) {
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
    for (auto &&[K, V] : Map) {
      Stream << K << ": [" << V.first << ", " << getName(V.second) << "]\n";
    }
  }

  void dump() const debug_function { dump(dbg); }
};

using Lattice = std::set<StoredByte>;

struct SegregateStackAccessesMFI : public SetUnionLattice<Lattice> {
  using Label = llvm::BasicBlock *;
  using GraphType = llvm::Function *;

  static LatticeElement applyTransferFunction(llvm::BasicBlock *BB,
                                              const LatticeElement &Value) {
    using namespace llvm;
    revng_log(Log, "Analyzing block " << getName(BB));
    LoggerIndent<> Indent(Log);

    LatticeElement StackBytes = Value;

    for (Instruction &I : *BB) {
      if (isCallToIsolatedFunction(&I)) {
        StackBytes.clear();
        continue;
      }

      // If it's not a load/store, pointer is nullptr
      if (not isa<LoadInst>(&I) and not isa<StoreInst>(&I))
        continue;

      revng_log(Log, "Analyzing instruction " << getName(&I));
      LoggerIndent<> Indent(Log);

      // Get stack offset, if available
      auto MaybeStartStackOffset = getStackOffset(&I);
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

static CallInst *
getAsModelGEP(IRBuilder<> &B, Value *Pointer, const model::Type &ModelType) {
  Module &M = *B.GetInsertBlock()->getModule();
  llvm::Type *T = Pointer->getType();
  Function *ModelGEPFunction = getModelGEP(M, T, T);
  auto *TypeString = toLLVMString(ModelType, M);
  auto *Int64Type = IntegerType::getInt64Ty(M.getContext());
  auto *Zero = ConstantInt::get(Int64Type, 0);
  return B.CreateCall(ModelGEPFunction, { TypeString, Pointer, Zero });
}

using GCBIWP = GeneratedCodeBasicInfoWrapperPass;

template<bool IsLegacy>
using LVB = LocalVariableBuilder<IsLegacy>;

/// This pass has two modes of operation:
/// - when LegacyLocalVariables is true it uses old FunctionTags and dedicated
/// functions to represent local variables, and accesses to them;
/// - when LegacyLocalVariables is false it represents local variables as
///   regular LLVM allocas, while accesses are modeled as regular load/store
///   instructions
///
/// TODO: At some point the legacy mode will be discontinued and we can remove
/// the template parameter.
template<bool LegacyLocalVariables>
class SegregateStackAccesses : public pipeline::FunctionPassImpl {
private:
  using MFIResult = std::map<BasicBlock *,
                             MFP::MFPResult<std::set<StoredByte>>>;

private:
  const model::Binary &Binary;
  Module &M;
  Function *SSACS = nullptr;
  Function *InitLocalSP = nullptr;
  std::set<Instruction *> ToPurge;
  model::VerifyHelper VH;
  const size_t CallInstructionPushSize = 0;
  std::map<Function *, Function *> OldToNew;
  std::set<Function *> FunctionsWithStackArguments;
  std::map<Function *, StackAccessRedirector> StackArgumentsRedirectors;
  std::vector<Instruction *> ToPushALAP;

  llvm::Type *TargetPointerSizedInteger = nullptr;
  llvm::Type *OpaquePointerType = nullptr;
  OpaqueFunctionsPool<FunctionTags::TypePair> AddressOfPool;
  LocalVariableBuilder<LegacyLocalVariables> VariableBuilder;

public:
  SegregateStackAccesses(llvm::ModulePass &Pass,
                         const model::Binary &Binary,
                         llvm::Module &M)
    requires(not LegacyLocalVariables)
    :
    pipeline::FunctionPassImpl(Pass),
    Binary(Binary),
    M(M),
    SSACS(M.getFunction("stack_size_at_call_site")),
    InitLocalSP(M.getFunction("init_local_sp")),
    CallInstructionPushSize(getCallPushSize(Binary)),
    TargetPointerSizedInteger(getPointerSizedInteger(M.getContext(), Binary)),
    OpaquePointerType(PointerType::get(M.getContext(), 0)),
    AddressOfPool(FunctionTags::AddressOf.getPool(M)),
    VariableBuilder(LVB</* IsLegacy */ false>::make(Binary, M)) {

    revng_assert(SSACS != nullptr);

    // After segregate, we should not introduce new calls to
    // `init_local_sp`: enable to DCE it away
    InitLocalSP->setOnlyReadsMemory();
  }

  SegregateStackAccesses(llvm::ModulePass &Pass,
                         const model::Binary &Binary,
                         llvm::Module &M)
    requires LegacyLocalVariables
    :
    pipeline::FunctionPassImpl(Pass),
    Binary(Binary),
    M(M),
    SSACS(M.getFunction("stack_size_at_call_site")),
    InitLocalSP(M.getFunction("_init_local_sp")),
    CallInstructionPushSize(getCallPushSize(Binary)),
    TargetPointerSizedInteger(getPointerSizedInteger(M.getContext(), Binary)),
    OpaquePointerType(PointerType::get(M.getContext(), 0)),
    AddressOfPool(FunctionTags::AddressOf.getPool(M)),
    VariableBuilder(LVB<true>::makeLegacyStackBuilder(Binary,
                                                      M,
                                                      AddressOfPool)) {

    revng_assert(SSACS != nullptr);

    // After segregate, we should not introduce new calls to
    // `_init_local_sp`: enable to DCE it away
    InitLocalSP->setOnlyReadsMemory();
  }

public:
  static void getAnalysisUsage(llvm::AnalysisUsage &AU);

public:
  bool prologue() final {
    upgradeDynamicFunctions();
    return true;
  }

  bool runOnFunction(const model::Function &ModelFunction,
                     llvm::Function &Function) final {

    llvm::Function &NewFunction = upgradeLocalFunction(&Function);
    segregateStackAccesses(NewFunction);

    return true;
  }

  bool epilogue() final {
    pushALAP();

    // Purge stores that have been used at least once
    for (Instruction *I : ToPurge)
      eraseFromParent(I);

    // Erase original functions
    for (auto &&[OldFunction, NewFunction] : OldToNew)
      eraseFromParent(OldFunction);

    return true;
  }

private:
  Value *pointer(IRBuilder<> &B, Value *V) const {
    return B.CreateIntToPtr(V, OpaquePointerType);
  }

  CallInst *createAddressOf(IRBuilder<> &B,
                            Value *V,
                            const model::UpcastableType &AllocatedType) {
    revng_assert(LegacyLocalVariables);

    auto *ArgType = V->getType();
    // Inject a call to AddressOf
    Constant *ModelTypeString = toLLVMString(AllocatedType, M);
    auto *AddressOfFunctionType = getAddressOfType(TargetPointerSizedInteger,
                                                   ArgType);
    auto *AddressOfFunction = AddressOfPool.get({ TargetPointerSizedInteger,
                                                  ArgType },
                                                AddressOfFunctionType,
                                                "AddressOf");
    return B.CreateCall(AddressOfFunction, { ModelTypeString, V });
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
      auto &ProtoT = *Binary.prototypeOrDefault(ImportedFunction.prototype());
      recreateApplyingModelPrototype(OldFunction, ProtoT);
    }
  }

  std::pair<llvm::Function *, abi::FunctionType::Layout>
  getOrCreateNewLocalFunction(Function *OldFunction) {
    MetaAddress Entry = getMetaAddressMetadata(OldFunction,
                                               "revng.function.entry");
    revng_assert(Entry.isValid());

    const model::Function &ModelFunction = Binary.Functions().at(Entry);

    // Create new FunctionType
    auto &Prototype = *Binary.prototypeOrDefault(ModelFunction.prototype());

    return recreateApplyingModelPrototype(OldFunction, Prototype);
  }

  llvm::Function *getOrCreateNewFunction(Function *OldFunction) {
    MetaAddress Entry = getMetaAddressMetadata(OldFunction,
                                               "revng.function.entry");

    if (Entry.isValid()) {
      return getOrCreateNewLocalFunction(OldFunction).first;
    } else {
      revng_assert(FunctionTags::DynamicFunction.isTagOf(OldFunction));
      return OldToNew.at(OldFunction);
    }
  }

  /// Upgrade function to reflect their model prototype
  Function &upgradeLocalFunction(Function *OldFunction) {
    using namespace abi::FunctionType;

    auto &&[NewFunction, Layout] = getOrCreateNewLocalFunction(OldFunction);

    // Let the new function steal the body from the old function
    moveBlocksInto(*OldFunction, *NewFunction);

    FunctionTags::StackAccessesSegregated.addTo(NewFunction);

    Type *NewReturnType = NewFunction->getReturnType();

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

    // Perform sanity checks on the return value and extract the type of the
    // result variable, if we're returning through a variable
    auto ReturnMethod = Layout.returnMethod();
    switch (ReturnMethod) {
    case ReturnMethod::Void:
      revng_assert(NewReturnType->isVoidTy());
      break;

    case ReturnMethod::ModelAggregate:
      if constexpr (LegacyLocalVariables) {
        // Nothing to check here.
      } else {
        if (Layout.hasSPTAR()) {
          // Nothing to check here
        } else {
          revng_assert(NewReturnType->isArrayTy());
          auto *ArrayTy = cast<llvm::ArrayType>(NewReturnType);
          auto *ElemTy = ArrayTy->getElementType();
          revng_assert(cast<llvm::IntegerType>(ElemTy)->getBitWidth() == 8);
          unsigned NumElems = ArrayTy->getNumElements();
          size_t ModelAggregateSize = *Layout.returnValueAggregateType().size();
          revng_assert(ModelAggregateSize == NumElems);
        }
      }
      break;

    case ReturnMethod::RegisterSet:
      // Assert each return instruction is using a StructInitializer
      for (BasicBlock &BB : *NewFunction) {
        if (auto *Ret = dyn_cast<ReturnInst>(BB.getTerminator())) {
          auto *Call = cast<CallInst>(Ret->getReturnValue());
          auto *Callee = getCalledFunction(Call);
          revng_assert(Call != nullptr);
          revng_assert(FunctionTags::StructInitializer.isTagOf(Callee));
        }
      }
      break;
    case ReturnMethod::Scalar:
      break;
    }

    Value *ReturnValueAllocation = nullptr;
    Value *ReturnValueIntAddress = nullptr;
    if (ReturnMethod == ReturnMethod::ModelAggregate) {
      const model::Type &A = Layout.returnValueAggregateType();
      VariableBuilder.setTargetFunction(NewFunction);
      tie(ReturnValueAllocation,
          ReturnValueIntAddress) = VariableBuilder
                                     .createLocalVariableAndTakeIntAddress(A);
      revng_assert(ReturnValueAllocation);
      revng_assert(ReturnValueIntAddress);

      if (Layout.hasSPTAR()) {
        // Identify the SPTAR
        auto &ModelArgument = Layout.Arguments[0];
        // Handle the argument pointing to the return value
        if (ModelArgument.Stack) {
          revng_assert(ModelArgument.Registers.size() == 0);
          Redirector->recordSpan(*ModelArgument.Stack + CallInstructionPushSize,
                                 ReturnValueIntAddress);
        } else {
          // It's in a register
          revng_assert(ModelArgument.Registers.size() == 1);
          Argument *OldArgument = nullptr;
          OldArgument = ArgumentToRegister.at(ModelArgument.Registers[0]);
          OldArgument->replaceAllUsesWith(ReturnValueIntAddress);
        }

        // Exclude the SPTAR from the list to process
        ModelArguments = llvm::drop_begin(ModelArguments);
      }
    }

    IRBuilder<> B(&NewFunction->getEntryBlock());
    setInsertPointToFirstNonAlloca(B, *NewFunction);

    // Handle arguments
    for (auto &&[ModelArgument, NewArgument] :
         zip(ModelArguments, NewFunction->args())) {

      // Extract from the new argument the old arguments
      unsigned OffsetInNewArgument = 0;
      Type *NewArgumentType = NewArgument.getType();
      unsigned NewArgumentSize = NewArgumentType->getIntegerBitWidth() / 8;

      llvm::Value *ToRecordSpan = nullptr;
      bool UsesStack = ModelArgument.Stack.has_value();

      using namespace abi::FunctionType::ArgumentKind;
      if (ModelArgument.Kind == PointerToCopy) {
        auto Architecture = Binary.Architecture();
        auto PointerSize = model::Architecture::getPointerSize(Architecture);
        revng_assert(ModelArgument.Type->size() > PointerSize);
        Value *AddressOfNewArgument = &NewArgument;
        if constexpr (LegacyLocalVariables)
          AddressOfNewArgument = createAddressOf(B,
                                                 &NewArgument,
                                                 ModelArgument.Type);

        if (UsesStack) {
          // When loading from this stack slot, return the address of the
          // address of the new argument
          revng_assert(ModelArgument.Registers.size() == 0);
          ToRecordSpan = AddressOfNewArgument;
        } else {
          // Replace the old argument with an address of the new argument
          revng_assert(ModelArgument.Registers.size() == 1);
          auto Register = ModelArgument.Registers[0];
          Argument *OldArgument = ArgumentToRegister.at(Register);
          OldArgument->replaceAllUsesWith(AddressOfNewArgument);
        }

      } else if (ModelArgument.Kind == Scalar) {
        revng_assert(ModelArgument.Type->isScalar());
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
            Shifted = B.CreateLShr(&NewArgument, ShiftAmount);
          Value *Trunced = B.CreateZExtOrTrunc(Shifted, OldArgumentType);

          // Replace old argument with the extracted valued
          OldArgument->replaceAllUsesWith(Trunced);

          // Consume size
          OffsetInNewArgument += OldArgumentSize;
        }

        if (ModelArgument.Stack) {
          Type *ArgumentType = NewArgument.getType();
          auto Pair = VariableBuilder.createAllocaWithPtrToInt(NewFunction,
                                                               ArgumentType);
          auto [Alloca, PtrToInt] = Pair;
          B.CreateStore(&NewArgument, Alloca);
          ToRecordSpan = PtrToInt;
        }

      } else if (ModelArgument.Kind == ReferenceToAggregate) {

        // Handle non-scalar argument (passed by pointer)
        Value *AddressOfNewArgument = &NewArgument;
        if constexpr (LegacyLocalVariables)
          AddressOfNewArgument = createAddressOf(B,
                                                 &NewArgument,
                                                 ModelArgument.Type);

        for (model::Register::Values Register : ModelArgument.Registers) {
          Argument *OldArgument = ArgumentToRegister.at(Register);

          // Load value
          Value *ArgumentPointer = computeAddress(B,
                                                  AddressOfNewArgument,
                                                  OffsetInNewArgument);
          Value *ArgumentValue = B.CreateLoad(OldArgument->getType(),
                                              ArgumentPointer);

          // Replace
          OldArgument->replaceAllUsesWith(ArgumentValue);

          // Consume size
          OffsetInNewArgument += model::Register::getSize(Register);
        }

        if (ModelArgument.Stack)
          ToRecordSpan = AddressOfNewArgument;
      }

      if (ToRecordSpan) {
        Redirector->recordSpan(*ModelArgument.Stack + CallInstructionPushSize,
                               ToRecordSpan);
      }
    }

    SmallVector<ReturnInst *, 4> Returns;
    for (BasicBlock &BB : *NewFunction)
      if (auto *Ret = dyn_cast<ReturnInst>(BB.getTerminator()))
        Returns.push_back(Ret);

    for (BasicBlock &BB : *NewFunction)
      revng_assert(BB.getTerminator() != nullptr);

    // Handle return values
    switch (ReturnMethod) {
    case ReturnMethod::ModelAggregate: {
      // Replace return instructions with returning a copy of the local variable
      // representing the return value
      for (ReturnInst *Ret : Returns) {
        B.SetInsertPoint(Ret);

        if (not Layout.hasSPTAR()) {
          // We have an aggregate returned through registers, fill in the
          // struct using stores
          revng_assert(Layout.returnValueRegisterCount() > 0);

          // Collect returned values
          SmallVector<llvm::Value *, 4> ReturnValues;
          Value *RetValue = Ret->getReturnValue();
          if (Layout.returnValueRegisterCount() == 1) {
            ReturnValues.push_back(RetValue);
          } else {
            auto *Call = cast<CallInst>(Ret->getReturnValue());
            auto *Callee = getCalledFunction(Call);
            revng_assert(Call != nullptr);
            revng_assert(FunctionTags::StructInitializer.isTagOf(Callee));
            llvm::copy(Call->args(), std::back_inserter(ReturnValues));
          }

          // Populate the local variable we're returning with the returned
          // values
          uint64_t Offset = 0;
          for (Value *ReturnValue : ReturnValues) {
            Value *Pointer = createAdd(B, ReturnValueIntAddress, Offset);
            B.CreateStore(ReturnValue, pointer(B, Pointer));
            Offset += ReturnValue->getType()->getIntegerBitWidth() / 8;
          }
        }

        // Return the pointer to the result variable
        revng_assert(ReturnValueAllocation);
        revng_assert(ReturnValueIntAddress);
        Value *ToReturn = nullptr;
        if constexpr (LegacyLocalVariables) {
          ToReturn = ReturnValueAllocation;
        } else {
          if (Layout.hasSPTAR()) {
            ToReturn = ReturnValueIntAddress;
          } else {
            ToReturn = B.CreateLoad(NewReturnType, ReturnValueAllocation);
          }
        }
        B.CreateRet(ToReturn);
        Ret->eraseFromParent();
      }
    } break;

    case ReturnMethod::Scalar: {
      Type *OldReturnType = OldFunction->getReturnType();

      if (OldReturnType != NewReturnType) {
        if (OldReturnType->isIntegerTy() and NewReturnType->isIntegerTy()) {
          // Handle return values smaller than the original function
          for (ReturnInst *Ret : Returns) {
            B.SetInsertPoint(Ret);
            B.CreateRet(B.CreateTrunc(Ret->getReturnValue(), NewReturnType));
            Ret->eraseFromParent();
          }
        } else if (OldReturnType->isStructTy()
                   and NewReturnType->isIntegerTy()) {
          // Turn struct_initializer into a an integer
          for (ReturnInst *Ret : Returns) {
            auto *Call = cast<CallInst>(Ret->getReturnValue());
            auto *Callee = getCalledFunction(Call);
            revng_assert(Call != nullptr);
            revng_assert(FunctionTags::StructInitializer.isTagOf(Callee));

            B.SetInsertPoint(Ret);
            Value *Accumulator = ConstantInt::get(NewReturnType, 0);
            uint64_t ShiftAmount = 0;
            for (Value *Argument : Call->args()) {
              auto *Extended = B.CreateZExtOrTrunc(Argument, NewReturnType);
              Accumulator = B.CreateOr(Accumulator,
                                       B.CreateShl(Extended, ShiftAmount));
              ShiftAmount += Argument->getType()->getIntegerBitWidth();
            }
            B.CreateRet(Accumulator);
            Ret->eraseFromParent();
            Call->eraseFromParent();
          }
        }
      }
    } break;

    case ReturnMethod::Void:
    case ReturnMethod::RegisterSet:
      // Nothing to do here
      break;

    default:
      revng_abort();
    }

    for (BasicBlock &BB : *NewFunction)
      revng_assert(BB.getTerminator() != nullptr);

    return *NewFunction;
  }

  void segregateStackAccesses(Function &F) {
    revng_assert(InitLocalSP != nullptr);

    // Get model::Function
    MetaAddress Entry = getMetaAddressMetadata(&F, "revng.function.entry");
    const model::Function &ModelFunction = Binary.Functions().at(Entry);

    revng_log(Log,
              "Segregating "
                << model::CNameBuilder(Binary).name(ModelFunction));

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

    //
    // Handle a call to an isolated function
    //
    for (BasicBlock &BB : F)
      for (Instruction &I : BB)
        if (CallInst *SSACSCall = getCallTo(&I, SSACS))
          handleCallSite(AnalysisResult, SSACSCall);

    //
    // Handle memory access, possibly targeting stack arguments
    //
    if (Redirector != nullptr)
      for (BasicBlock &BB : F)
        for (Instruction &I : BB)
          if (isa<LoadInst>(&I) or isa<StoreInst>(&I))
            handleMemoryAccess(*Redirector, &I);

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

  void handleCallSite(MFIResult &AnalysisResult, CallInst *SSACSCall) {
    LoggerIndent<> Indent(Log);

    //
    // Find call to init_local_sp
    //
    Function *Caller = SSACSCall->getParent()->getParent();

    // Get stack size at call site
    auto MaybeStackSize = getSignedConstantArg(SSACSCall, 0);

    // Obtain the prototype
    const auto &Prototype = *getCallSitePrototype(Binary, SSACSCall);
    using namespace abi::FunctionType;
    abi::FunctionType::Layout Layout = Layout::make(Prototype);

    // Find old call instruction
    CallInst *OldCall = findAssociatedCall(SSACSCall);
    revng_assert(OldCall != nullptr);

    IRBuilder<> B(OldCall);

    //
    // Map llvm::Argument * to model::Register
    //
    std::map<model::Register::Values, llvm::Value *> ArgumentToRegister;
    auto ArgumentRegisters = Layout.argumentRegisters();
    for (auto &&[Register, OldArg] : zip(ArgumentRegisters, OldCall->args()))
      ArgumentToRegister[Register] = OldArg.get();

    // Check if it's a direct call
    auto *Callee = dyn_cast<Function>(OldCall->getCalledOperand());
    bool IsDirect = (Callee != nullptr);

    // Obtain or compute the function type for the call
    FunctionType *CalleeType = nullptr;
    Value *CalledValue = nullptr;
    if (IsDirect) {
      Function *NewCallee = getOrCreateNewFunction(Callee);
      CalledValue = NewCallee;
      CalleeType = NewCallee->getFunctionType();
    } else {
      CalleeType = &layoutToLLVMFunctionType(Layout, OldCall->getType());
      CalledValue = B.CreateBitCast(OldCall->getCalledOperand(),
                                    CalleeType->getPointerTo());
    }

    SmallVector<llvm::Value *, 4> Arguments;

    StackAccessRedirector Redirector(-MaybeStackSize.value_or(0)
                                     + CallInstructionPushSize);

    SmallVector<llvm::Type *, 8> LLVMArgumentTypes;
    bool HasSPTAR = Layout.hasSPTAR();

    auto ReturnMethod = Layout.returnMethod();
    if (ReturnMethod == ReturnMethod::ModelAggregate and HasSPTAR) {
      // Inject the SPTAR in LLVMArgumentTypes
      revng_assert(Layout.Arguments.size() > 0);
      uint64_t SPTARSize = *Layout.Arguments[0].Type->size();
      LLVMArgumentTypes.push_back(B.getIntNTy(SPTARSize * 8));
    }

    copy(CalleeType->params(), std::back_inserter(LLVMArgumentTypes));

    bool MessageEmitted = false;
    for (auto &&[LLVMType, ModelArgument] :
         llvm::zip(LLVMArgumentTypes, Layout.Arguments)) {
      uint64_t NewSize = *ModelArgument.Type->size();

      switch (ModelArgument.Kind) {

      case ArgumentKind::PointerToCopy: {
        Value *Pointer = nullptr;

        if (ModelArgument.Stack) {
          model::Architecture::Values Architecture = Binary.Architecture();
          auto PointerSize = model::Architecture::getPointerSize(Architecture);
          revng_assert(ModelArgument.Type->size() > PointerSize);
          revng_assert(ModelArgument.Registers.size() == 0);
          revng_assert(ModelArgument.Stack->Size == PointerSize);
          revng_assert(MaybeStackSize);

          // Create an alloca
          auto *StackSpanType = B.getIntNTy(ModelArgument.Stack->Size * 8);
          auto Pair = VariableBuilder.createAllocaWithPtrToInt(Caller,
                                                               StackSpanType);
          auto [Alloca, PtrToInt] = Pair;

          // Record its portion of the stack for redirection
          Redirector.recordSpan(*ModelArgument.Stack, PtrToInt);

          // Load the alloca and record it as a pointer
          Pointer = B.CreateLoad(Alloca->getAllocatedType(), Alloca);
        } else {
          revng_assert(ModelArgument.Registers.size() == 1);
          auto Register = ModelArgument.Registers[0];
          Pointer = ArgumentToRegister.at(Register);
        }

        // Pass as argument the pointer compute above.
        // In legacy mode, wrap it into a ModelGEP at offset 0.
        if constexpr (LegacyLocalVariables) {
          Pointer = getAsModelGEP(B, Pointer, *ModelArgument.Type);
        }
        Arguments.push_back(Pointer);
      } break;

      case ArgumentKind::Scalar:
      case ArgumentKind::ShadowPointerToAggregateReturnValue: {
        revng_assert(ModelArgument.Type->isScalar());
        Value *Accumulator = ConstantInt::get(LLVMType, 0);
        unsigned OffsetInNewArgument = 0;
        for (auto &Register : ModelArgument.Registers) {
          Value *OldArgument = ArgumentToRegister.at(Register);
          unsigned OldSize = model::Register::getSize(Register);

          Value *Extended = B.CreateZExtOrTrunc(OldArgument, LLVMType);

          unsigned ShiftAmount = shiftAmount(OffsetInNewArgument,
                                             NewSize,
                                             OldSize);
          Value *Shifted = Extended;
          if (ShiftAmount != 0)
            Shifted = B.CreateLShr(Extended, ShiftAmount);

          Accumulator = B.CreateOr(Accumulator, Shifted);

          // Consume size
          OffsetInNewArgument += OldSize;
        }

        if (ModelArgument.Stack and not MaybeStackSize) {
          if (not MessageEmitted) {
            MessageEmitted = true;
            emitMessage(OldCall,
                        "Ignoring stack arguments for this call site: "
                        "stack size at call site unknown",
                        OldCall->getDebugLoc());
          }
        } else if (ModelArgument.Stack) {
          unsigned OldSize = ModelArgument.Stack->Size;
          revng_assert(OldSize <= 128 / 8);
          revng_assert(MaybeStackSize);

          // Create an alloca
          IntegerType *StackSpanType = B.getIntNTy(OldSize * 8);
          auto Pair = VariableBuilder.createAllocaWithPtrToInt(Caller,
                                                               StackSpanType);
          auto [Alloca, PtrToInt] = Pair;

          // Record its portion of the stack for redirection
          Redirector.recordSpan(*ModelArgument.Stack, PtrToInt);

          Value *Loaded = B.CreateLoad(Alloca->getAllocatedType(), Alloca);

          // Extend, shift and or in Accumulator
          // Note: here we might truncate too, since certain architectures
          //       report a stack span of 8 bytes but the associated type is
          //       actually 32 bits
          Value *Extended = B.CreateZExtOrTrunc(Loaded, LLVMType);

          unsigned ShiftAmount = shiftAmount(OffsetInNewArgument,
                                             NewSize,
                                             OldSize);
          Value *Shifted = Extended;
          if (ShiftAmount != 0)
            Shifted = B.CreateShl(Extended, ShiftAmount);

          Accumulator = B.CreateOr(Accumulator, Shifted);
        }

        Arguments.push_back(Accumulator);
      } break;

      case ArgumentKind::ReferenceToAggregate: {

        VariableBuilder.setTargetFunction(SSACSCall->getFunction());
        Instruction
          *StackArgsAddress = VariableBuilder
                                .createCallStackArgumentVariable(*ModelArgument
                                                                    .Type);
        revng_assert(StackArgsAddress);
        Instruction *StackArgsAllocation = nullptr;
        if constexpr (LegacyLocalVariables) {
          // When in legacy mode, the actual instruction performing the stack
          // allocation is the first operand of StackArgsAddress, which is
          // guaranteed to be a call to AddressOf.
          auto *CallToAddressOf = getCallToTagged(StackArgsAddress,
                                                  FunctionTags::AddressOf);
          auto *AllocationInst = CallToAddressOf->getArgOperand(1);
          StackArgsAllocation = cast<Instruction>(AllocationInst);
          // Then we have to push the address computation and the allocation
          // ALAP. The address should be pushed ALAP first to leave slack for
          // the allocation instruction to also be pushed ALAP afterwards.
          ToPushALAP.push_back(StackArgsAddress);
          ToPushALAP.push_back(StackArgsAllocation);
        } else {
          // When not in legacy mode, the instruction returning the address of
          // the stack arguments is also the instruction performing the actual
          // allocation, so we can just say they're equal.
          //
          // Also, there's no need to push it ALAP, since it's an alloca.
          // We do have to push ALAP its cast to an integer though.
          revng_assert(isa<PtrToIntInst>(StackArgsAddress));
          revng_assert(isa<AllocaInst>(StackArgsAddress->getOperand(0)));
          StackArgsAllocation = StackArgsAddress;
          ToPushALAP.push_back(StackArgsAddress);
        }

        // We also have to copy over metadata, from the annotation about the
        // size of the stack arguments.
        StackArgsAllocation->copyMetadata(*SSACSCall);

        unsigned OffsetInNewArgument = 0;
        for (auto &Register : ModelArgument.Registers) {
          Value *OldArgument = ArgumentToRegister.at(Register);
          unsigned OldSize = model::Register::getSize(Register);

          Value *Address = createAdd(B, StackArgsAddress, OffsetInNewArgument);

          // Store value
          Value *Pointer = pointer(B, Address);
          B.CreateStore(OldArgument, Pointer);

          // Consume size
          OffsetInNewArgument += OldSize;
        }

        if (ModelArgument.Stack)
          Redirector.recordSpan(*ModelArgument.Stack, StackArgsAddress);

        Arguments.push_back(StackArgsAllocation);
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

    Value *ReturnValuePointer = nullptr;
    // Handle SPTAR by dropping the actual argument and saving it for later
    if (HasSPTAR) {
      revng_assert(Arguments.size() > 0);

      // The return value is pointed by the SPTAR
      ReturnValuePointer = Arguments[0];

      Arguments.erase(Arguments.begin());
    }

    // If the old return type and the new one are identical, switch to the old
    // one in the new call
    auto *OldCallType = OldCall->getFunctionType();
    auto *OldReturnType = OldCallType->getReturnType();
    auto *NewReturnType = CalleeType->getReturnType();
    if (auto *OldStructType = dyn_cast<StructType>(OldReturnType)) {
      if (auto *NewStructType = dyn_cast<StructType>(NewReturnType)) {
        if (NewStructType->isLayoutIdentical(OldStructType)) {
          CalleeType = FunctionType::get(OldReturnType,
                                         CalleeType->params(),
                                         CalleeType->isVarArg());
        }
      }
    }

    // Actually create the new call and replace the old one
    CallInst *NewCall = B.CreateCall(CalleeType, CalledValue, Arguments);
    NewCall->copyMetadata(*OldCall);
    NewCall->setAttributes(OldCall->getAttributes());

    switch (Layout.returnMethod()) {
    case ReturnMethod::ModelAggregate: {
      if (HasSPTAR) {
        // In legacy mode, make a reference out of ReturnValuePointer, using a
        // ModelGEP at offset 0.
        if constexpr (LegacyLocalVariables) {
          getAsModelGEP(B,
                        ReturnValuePointer,
                        Layout.returnValueAggregateType());
        }
      } else {
        revng_assert(not ReturnValuePointer);
        const auto &ReturnType = Layout.returnValueAggregateType();
        if constexpr (LegacyLocalVariables) {
          ReturnValuePointer = createAddressOf(B, NewCall, ReturnType);
        } else {
          auto &VB = VariableBuilder;
          VB.setTargetFunction(Caller);
          Value *Allocation = nullptr;
          Value *IntAddress = nullptr;
          tie(Allocation,
              IntAddress) = VB.createLocalVariableAndTakeIntAddress(ReturnType);
          B.CreateStore(NewCall, Allocation);
          ReturnValuePointer = IntAddress;
        }
      }

      if (HasSPTAR) {
        revng_assert(not OldReturnType->isStructTy());
        OldCall->replaceAllUsesWith(ReturnValuePointer);
      } else {
        // We're returning an aggregate, but not via SPTAR, we're using one or
        // more registers
        if (OldReturnType->isStructTy()) {
          SmallVector<SmallPtrSet<CallInst *, 2>, 2>
            ExtractedValues = getExtractedValuesFromInstruction(OldCall);
          for (auto &Group : llvm::enumerate(ExtractedValues)) {

            unsigned FieldIndex = Group.index();
            SmallPtrSet<CallInst *, 2> &ExtractedAtIndex = Group.value();
            if (ExtractedAtIndex.empty())
              continue;

            unsigned BitOffset = getBitOffsetAt(cast<StructType>(OldReturnType),
                                                FieldIndex);
            revng_assert(0 == (BitOffset % 8));
            unsigned ByteOffset = BitOffset / 8;

            Value *Pointer = createAdd(B, ReturnValuePointer, ByteOffset);
            Type *ExtractedType = (*ExtractedAtIndex.begin())->getType();
            auto *Load = B.CreateLoad(ExtractedType, pointer(B, Pointer));

            for (CallInst *Extractor : Group.value()) {
              Extractor->replaceAllUsesWith(Load);
              eraseFromParent(Extractor);
            }
          }
        } else {
          auto *Load = B.CreateLoad(ReturnValuePointer->getType(),
                                    pointer(B, ReturnValuePointer));
          OldCall->replaceAllUsesWith(Load);
        }
      }

    } break;

    case ReturnMethod::Scalar:

      if (OldReturnType != NewReturnType and OldReturnType->isIntegerTy()
          and NewReturnType->isIntegerTy()) {
        // We're using a large register to return a smaller integer value (e.g.,
        // returning a 32-bit integer through rax, which is 64-bit)
        auto OldSize = OldReturnType->getIntegerBitWidth();
        auto NewSize = NewReturnType->getIntegerBitWidth();
        revng_assert(NewSize <= OldSize);
        auto *Extended = cast<Instruction>(B.CreateZExt(NewCall,
                                                        OldReturnType));
        OldCall->replaceAllUsesWith(Extended);
      } else if (OldReturnType->isStructTy() and NewReturnType->isIntegerTy()) {
        // We're returning a large integer value through multiple values (e.g.,
        // returning a 64-bit integer through two registers in i386)
        SmallVector<SmallPtrSet<CallInst *, 2>, 2>
          ExtractedValues = getExtractedValuesFromInstruction(OldCall);
        for (auto &Group : llvm::enumerate(ExtractedValues)) {

          unsigned FieldIndex = Group.index();
          SmallPtrSet<CallInst *, 2> &ExtractedAtIndex = Group.value();
          if (ExtractedAtIndex.empty())
            continue;

          unsigned ShiftAmount = getBitOffsetAt(cast<StructType>(OldReturnType),
                                                FieldIndex);
          Type *TruncatedType = (*ExtractedAtIndex.begin())->getType();
          Value *Replacement = B.CreateTrunc(B.CreateLShr(NewCall, ShiftAmount),
                                             TruncatedType);
          for (CallInst *Extractor : Group.value()) {
            revng_assert(TruncatedType == Extractor->getType());
            Extractor->replaceAllUsesWith(Replacement);
            eraseFromParent(Extractor);
          }
        }
      } else {
        revng_assert(not OldReturnType->isStructTy());
        OldCall->replaceAllUsesWith(NewCall);
      }
      break;

    case ReturnMethod::Void:
      // Nothing to do here
      break;
    case ReturnMethod::RegisterSet:
      OldCall->replaceAllUsesWith(NewCall);
      break;

    default:
      revng_abort();
    }

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

  void handleMemoryAccess(const StackAccessRedirector &Redirector,
                          Instruction *I) {
    revng_log(Log, "Handling memory access " << getName(I));
    LoggerIndent<> Indent(Log);

    auto MaybeStackOffset = getStackOffset(I);
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
    // Find call to init_local_sp
    //
    CallInst *InitLocalSPCall = findCallTo(&F, InitLocalSP);
    if (InitLocalSPCall == nullptr or ModelFunction.StackFrameType().isEmpty())
      return;

    //
    // Get stack frame size
    //
    uint64_t StackFrameSize = 0;
    if (const model::TypeDefinition *T = ModelFunction.stackFrameType())
      StackFrameSize = *T->size(VH);

    //
    // Create call and rebase SP0, if StackFrameSize is not zero
    //
    if (StackFrameSize != 0) {
      VariableBuilder.setTargetFunction(&F);
      Instruction *StackFrameAddress = VariableBuilder
                                         .createStackFrameVariable();

      IRBuilder<> Builder(InitLocalSPCall);
      auto *SP0 = Builder.CreateAdd(StackFrameAddress,
                                    getSPConstant(StackFrameSize));
      InitLocalSPCall->replaceAllUsesWith(SP0);

      // Cleanup init_local_sp
      eraseFromParent(InitLocalSPCall);
    }
  }

private:
  /// \name Support functions
  /// \{

  Constant *getSPConstant(uint64_t Value) const {
    return ConstantInt::get(TargetPointerSizedInteger, Value);
  }

  Value *computeAddress(IRBuilder<> &B, Value *Base, int64_t Offset) const {
    return pointer(B, createAdd(B, Base, Offset));
  }

  void replace(Instruction *I, Value *Base, int64_t Offset) {
    ToPurge.insert(I);

    IRBuilder<> B(I);
    auto *NewAddress = computeAddress(B, Base, Offset);

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
                                 const model::TypeDefinition &Prototype) {
    using namespace abi::FunctionType;
    auto Layout = Layout::make(Prototype);

    Type *OldReturnType = OldFunction->getReturnType();
    FunctionType &NewType = layoutToLLVMFunctionType(Layout, OldReturnType);

    // NOTE: all the model *must* be read above this line!
    //       If we don't do this, we will break invalidation tracking
    //       information.
    auto It = OldToNew.find(OldFunction);
    if (It != OldToNew.end())
      return { It->second, Layout };

    // Create the new function, stealing the name
    Function &NewFunction = recreateWithoutBody(*OldFunction, NewType);

    // Record the old-to-new mapping
    OldToNew[OldFunction] = &NewFunction;

    return { &NewFunction, Layout };
  }

  llvm::FunctionType &
  layoutToLLVMFunctionType(const abi::FunctionType::Layout &Layout,
                           Type *OldReturnType) const {
    // Process arguments
    using namespace abi::FunctionType;
    SmallVector<Type *> FunctionArguments;
    for (const Layout::Argument &Argument : Layout.Arguments) {
      model::UpcastableType ArgumentType = Argument.Type;

      switch (Argument.Kind) {
      case abi::FunctionType::ArgumentKind::ShadowPointerToAggregateReturnValue:
        // Skip SPTAR
        continue;
      case abi::FunctionType::ArgumentKind::PointerToCopy:
      case abi::FunctionType::ArgumentKind::ReferenceToAggregate:
        ArgumentType = model::PointerType::make(std::move(ArgumentType),
                                                Binary.Architecture());
        break;
      case abi::FunctionType::ArgumentKind::Scalar:
        // Do nothing
        break;
      default:
        revng_abort();
      }

      auto *LLVMType = getLLVMTypeForScalar(M.getContext(), *ArgumentType);
      FunctionArguments.push_back(LLVMType);
    }

    // Process return type
    Type *ReturnType = nullptr;
    switch (Layout.returnMethod()) {
    case ReturnMethod::Void:
      // No return values, forward returning void
      revng_assert(OldReturnType->isVoidTy());
      ReturnType = OldReturnType;
      break;

    case ReturnMethod::ModelAggregate:
      if constexpr (LegacyLocalVariables) {
        ReturnType = TargetPointerSizedInteger;
      } else {
        if (Layout.hasSPTAR()) {
          ReturnType = TargetPointerSizedInteger;
        } else {
          const model::Type &ReturnAggregate = Layout
                                                 .returnValueAggregateType();
          size_t ReturnSize = *ReturnAggregate.size();
          auto &Context = OldReturnType->getContext();
          auto *Int8 = llvm::IntegerType::getInt8Ty(Context);
          ReturnType = llvm::ArrayType::get(Int8, ReturnSize);
        }
      }
      break;

    case ReturnMethod::Scalar: {
      // We either have a return value that fits in a single register, or it's
      // CABIFunctionDefinition returning stuff through registers
      unsigned Bits = 0;
      for (const Layout::ReturnValue &ReturnValue : Layout.ReturnValues)
        Bits += ReturnValue.Type->size().value() * 8;
      ReturnType = IntegerType::getIntNTy(OldReturnType->getContext(), Bits);
    } break;

    case ReturnMethod::RegisterSet:
      // We have a RawFunctionDefinition returning things over multiple
      // registers
      revng_assert(Layout.returnValueRegisterCount() > 1);
      ReturnType = OldReturnType;
      break;

    default:
      revng_abort();
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

static void getAnalysisUsage(llvm::AnalysisUsage &AU) {
  AU.setPreservesCFG();
  AU.addRequired<LoadModelWrapperPass>();
  AU.addRequired<GeneratedCodeBasicInfoWrapperPass>();
}

template<>
void SegregateStackAccesses<true>::getAnalysisUsage(AnalysisUsage &AU) {
  return ::getAnalysisUsage(AU);
}

template<>
void SegregateStackAccesses<false>::getAnalysisUsage(AnalysisUsage &AU) {
  return ::getAnalysisUsage(AU);
}

template<>
char pipeline::FunctionPass<SegregateStackAccesses<true>>::ID = 0;

template<>
char pipeline::FunctionPass<SegregateStackAccesses<false>>::ID = 0;

static constexpr const char *LegacyFlag = "legacy-segregate-stack-accesses";

struct LegacySegregateStackAccessesPipe {
  static constexpr auto Name = LegacyFlag;

  std::vector<pipeline::ContractGroup> getContract() const {
    using namespace pipeline;
    using namespace revng::kinds;
    return { ContractGroup::transformOnlyArgument(StackPointerPromoted,
                                                  StackAccessesSegregated,
                                                  InputPreservation::Erase) };
  }

  void registerPasses(legacy::PassManager &Manager) {
    using Pass = SegregateStackAccesses</* LegacyLocalVariables = */ true>;
    Manager.add(new pipeline::FunctionPass<Pass>);
  }
};

static pipeline::RegisterLLVMPass<LegacySegregateStackAccessesPipe> X;

static constexpr const char *Flag = "segregate-stack-accesses";

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
    using Pass = SegregateStackAccesses</* LegacyLocalVariables = */ false>;
    Manager.add(new pipeline::FunctionPass<Pass>);
  }
};

static pipeline::RegisterLLVMPass<SegregateStackAccessesPipe> Y;
