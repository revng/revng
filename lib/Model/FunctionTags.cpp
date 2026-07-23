//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/FunctionTags.h"
#include "revng/Model/IRHelpers.h"
#include "revng/Model/NamedTypedRegister.h"
#include "revng/Model/ProgramCounterHandler.h"
#include "revng/Support/IRBuilder.h"
#include "revng/Support/IRHelpers.h"
#include "revng/Support/Tag.h"

namespace FunctionTags {

Tag QEMU("qemu");
Tag Helper("helper");

Tag ABIEnforced("abi-enforced", Isolated);
Tag CSVsPromoted("csvs-promoted", ABIEnforced);

Tag Exceptional("exceptional");
Tag StructInitializer("struct-initializer");
Tag OpaqueCSVValue("opaque-csv-value");
Tag FunctionDispatcher("function-dispatcher");
Tag Root("root");
Tag IsolatedRoot("isolated-root");
Tag CSVsAsArgumentsWrapper("csvs-as-arguments-wrapper");
Tag Marker("marker");
Tag DynamicFunction("dynamic-function");
Tag ClobbererFunction("clobberer-function");
Tag WriterFunction("writer-function");
Tag ReaderFunction("reader-function");
Tag OpaqueReturnAddressFunction("opaque-return-address");

Tag CSV("csv");

Tag AllocatesLocalVariable("allocates-local-variable");
Tag ReturnsPolymorphic("returns-polymorphic");
Tag IsRef("is-ref");

Tag ScopeCloserMarker("scope-closer");
Tag GotoBlockMarker("goto-block");

FunctionPoolTag<TypePair>
  AddressOf("address-of",
            { llvm::Attribute::NoUnwind,
              llvm::Attribute::WillReturn,
              llvm::Attribute::NoMerge },
            llvm::MemoryEffects::none(),
            { &FunctionTags::UniquedByPrototype },
            [](OpaqueFunctionsPool<TypePair> &Pool,
               llvm::Module &M,
               const FunctionPoolTag<TypePair> &Tag) {
              for (llvm::Function &F : Tag.functions(&M)) {
                revng_assert(AddressOf.isTagOf(&F));
                revng_assert(Tag.isTagOf(&F));
                auto *ArgType = F.getFunctionType()->getParamType(1);
                auto *RetType = F.getFunctionType()->getReturnType();
                Pool.record({ RetType, ArgType }, &F);
              }
            });

FunctionPoolTag<TypePair>
  OpaqueExtractValue("opaque-extract-value",
                     { llvm::Attribute::NoInline,
                       llvm::Attribute::NoMerge,
                       llvm::Attribute::NoUnwind,
                       llvm::Attribute::WillReturn },
                     llvm::MemoryEffects::none(),
                     { &FunctionTags::UniquedByPrototype },
                     [](OpaqueFunctionsPool<TypePair> &Pool,
                        llvm::Module &M,
                        const FunctionPoolTag<TypePair> &Tag) {
                       for (llvm::Function &F : Tag.functions(&M)) {
                         auto Struct = F.getFunctionType()->getParamType(0);
                         auto RetType = F.getFunctionType()->getReturnType();
                         Pool.record({ RetType, Struct }, &F);
                       }
                     });

FunctionPoolTag<llvm::Type *>
  Parentheses("parentheses",
              { llvm::Attribute::NoUnwind, llvm::Attribute::WillReturn },
              llvm::MemoryEffects::none(),
              { &FunctionTags::UniquedByPrototype },
              InitializationMode::InitializeFromReturnType);

Tag LiteralPrintDecorator("literal-print-decorator");

FunctionPoolTag<llvm::Type *>
  HexInteger("hex-integer",
             { llvm::Attribute::NoUnwind, llvm::Attribute::WillReturn },
             llvm::MemoryEffects::none(),
             { &FunctionTags::LiteralPrintDecorator,
               &FunctionTags::UniquedByPrototype },
             InitializationMode::InitializeFromReturnType);

FunctionPoolTag<llvm::Type *>
  CharInteger("char-integer",
              { llvm::Attribute::NoUnwind, llvm::Attribute::WillReturn },
              llvm::MemoryEffects::none(),
              { &FunctionTags::LiteralPrintDecorator,
                &FunctionTags::UniquedByPrototype },
              InitializationMode::InitializeFromReturnType);

FunctionPoolTag<llvm::Type *>
  BoolInteger("bool-integer",
              { llvm::Attribute::NoUnwind, llvm::Attribute::WillReturn },
              llvm::MemoryEffects::none(),
              { &FunctionTags::LiteralPrintDecorator,
                &FunctionTags::UniquedByPrototype },
              InitializationMode::InitializeFromReturnType);

FunctionPoolTag<llvm::Type *>
  NullPtr("nullptr",
          { llvm::Attribute::NoUnwind, llvm::Attribute::WillReturn },
          llvm::MemoryEffects::none(),
          { &FunctionTags::LiteralPrintDecorator,
            &FunctionTags::UniquedByPrototype },
          InitializationMode::InitializeFromReturnType);

FunctionPoolTag<llvm::Type *>
  LocalVariable("local-variable",
                { llvm::Attribute::NoUnwind,
                  llvm::Attribute::WillReturn,
                  llvm::Attribute::NoMerge },
                llvm::MemoryEffects::none(),
                { &FunctionTags::IsRef,
                  &FunctionTags::AllocatesLocalVariable,
                  &FunctionTags::ReturnsPolymorphic,
                  &FunctionTags::UniquedByPrototype },
                InitializationMode::InitializeFromReturnType);

FunctionPoolTag<llvm::Type *>
  Assign("assign",
         { llvm::Attribute::NoUnwind, llvm::Attribute::WillReturn },
         llvm::MemoryEffects::writeOnly(),
         { &FunctionTags::UniquedByPrototype },
         InitializationMode::InitializeFromArgument0);

FunctionPoolTag<llvm::Type *>
  Copy("copy",
       { llvm::Attribute::NoUnwind, llvm::Attribute::WillReturn },
       llvm::MemoryEffects::readOnly(),
       { &FunctionTags::UniquedByPrototype },
       InitializationMode::InitializeFromReturnType);

/// Tag for global variables representing segments
Tag SegmentGlobal("segment-global");

/// Tag for functions that must survive function isolation.
Tag KeepPostIsolation("keep-post-isolation");

inline void
segmentGlobalGetterInitializer(OpaqueFunctionsPool<SegmentRefPoolKey> &Pool,
                               llvm::Module &M,
                               const FunctionPoolTag<SegmentRefPoolKey> &Tag) {
  for (llvm::Function &F : Tag.functions(&M)) {
    MetaAddress StartAddress = extractSegmentKeyFromMetadata(F);
    // The virtual size is part of the pool key but not of the segment metadata:
    // recover it from the size of the segment's global variable.
    auto *Global = M.getGlobalVariable(SegmentGlobal::getNameFor(StartAddress),
                                       /* AllowInternal */ true);
    revng_assert(Global != nullptr);
    auto *GlobalTy = llvm::cast<llvm::ArrayType>(Global->getValueType());
    uint64_t VirtualSize = GlobalTy->getNumElements();
    Pool.record({ StartAddress, VirtualSize }, &F);
  }
}

inline llvm::Function &segmentGlobalGetterFactory(llvm::Module &M,
                                                  SegmentRefPoolKey Key) {
  using namespace llvm;
  auto [StartAddress, VirtualSize] = Key;
  auto *ReturnType = M.getDataLayout().getIntPtrType(M.getContext());
  auto *FT = FunctionType::get(ReturnType, {}, false);
  std::string Name = "get_" + SegmentGlobal::getNameFor(StartAddress);
  Function &Result = *Function::Create(FT,
                                       GlobalValue::ExternalLinkage,
                                       Name,
                                       M);
  setSegmentKeyMetadata(Result, StartAddress);

  // Fill in body
  auto *Entry = llvm::BasicBlock::Create(Result.getContext(), "", &Result);
  revng::IRBuilder B(Entry);
  auto &Global = SegmentGlobal::get(M, StartAddress, VirtualSize);
  B.CreateRet(B.CreatePtrToInt(&Global, Result.getReturnType()));

  return Result;
}

/// Tag for functions returning an intptr_t of a specific segment.
///
/// This is important since LLVM does not optimize arithmetic done with
/// ConstantExpr. Once optimizations are done these can go away using
/// inline-segment-global-getter.
FunctionPoolTag<SegmentRefPoolKey>
  SegmentGlobalGetter("segment-global-getter",
                      { llvm::Attribute::NoUnwind,
                        llvm::Attribute::WillReturn,
                        llvm::Attribute::NoInline },
                      llvm::MemoryEffects::none(),
                      { &FunctionTags::IsRef,
                        &FunctionTags::UniquedByMetadata,
                        &FunctionTags::KeepPostIsolation },
                      segmentGlobalGetterInitializer,
                      segmentGlobalGetterFactory);

Tag LiftingArtifactsRemoved("lifting-artifacts-removed", CSVsPromoted);

Tag StackPointerPromoted("stack-pointer-promoted", LiftingArtifactsRemoved);

Tag StackAccessesSegregated("stack-accesses-segregated", StackPointerPromoted);

Tag Decompiled("decompiled", StackPointerPromoted);

Tag StackOffsetMarker("stack-offset-marker");

Tag BinaryOperationOverflows("binary-operation-overflow");

Tag Comment("comment");

} // namespace FunctionTags

template<typename T>
concept DerivedValue = std::is_base_of_v<llvm::Value, T>;

using std::conditional_t;

template<DerivedValue ConstnessT, DerivedValue ResultT>
using PossiblyConstValueT = conditional_t<std::is_const_v<ConstnessT>,
                                          std::add_const_t<ResultT>,
                                          std::remove_const_t<ResultT>>;

template<DerivedValue T>
using ValueT = PossiblyConstValueT<T, llvm::Value>;

template<DerivedValue T>
using CallT = PossiblyConstValueT<T, llvm::CallInst>;

template<DerivedValue T>
using CallPtrSet = llvm::SmallPtrSet<CallT<T> *, 2>;

template<DerivedValue T>
llvm::SmallVector<CallPtrSet<T>, 2>
getConstQualifiedExtractedValuesFromInstruction(T *I) {

  llvm::SmallVector<CallPtrSet<T>, 2> Results;

  auto *StructTy = llvm::cast<llvm::StructType>(I->getType());
  unsigned NumFields = StructTy->getNumElements();
  Results.resize(NumFields, {});

  // Find extract value uses transitively, traversing PHIs and markers
  CallPtrSet<T> Calls;
  for (auto *TheUser : I->users()) {
    if (auto *ExtractV = getCallToTagged(TheUser,
                                         FunctionTags::OpaqueExtractValue)) {
      Calls.insert(ExtractV);
    } else {
      if (auto *Call = dyn_cast<llvm::CallInst>(TheUser)) {
        if (not isCallToTagged(Call, FunctionTags::Parentheses))
          continue;
      }

      // traverse PHIS and markers until we find extractvalues
      llvm::SmallPtrSet<ValueT<T> *, 8> Visited = {};
      llvm::SmallPtrSet<ValueT<T> *, 8> ToVisit = { TheUser };
      while (not ToVisit.empty()) {

        llvm::SmallPtrSet<ValueT<T> *, 8> NextToVisit = {};

        for (ValueT<T> *Ident : ToVisit) {
          Visited.insert(Ident);
          NextToVisit.erase(Ident);

          for (auto *User : Ident->users()) {
            using FunctionTags::OpaqueExtractValue;
            if (auto *EV = getCallToTagged(User, OpaqueExtractValue)) {
              Calls.insert(EV);
            } else if (auto *IdentUser = llvm::dyn_cast<llvm::CallInst>(User)) {
              if (isCallToTagged(IdentUser, FunctionTags::Parentheses))
                NextToVisit.insert(IdentUser);
            } else if (auto *PHIUser = llvm::dyn_cast<llvm::PHINode>(User)) {
              if (not Visited.contains(PHIUser))
                NextToVisit.insert(PHIUser);
            }
          }
        }

        ToVisit = NextToVisit;
      }
    }
  }

  for (auto *E : Calls) {
    revng_assert(isa<llvm::IntegerType>(E->getType())
                 or isa<llvm::PointerType>(E->getType()));
    auto FieldId = cast<llvm::ConstantInt>(E->getArgOperand(1))->getZExtValue();
    Results[FieldId].insert(E);
  }

  return Results;
};

llvm::SmallVector<llvm::SmallPtrSet<llvm::CallInst *, 2>, 2>
getExtractedValuesFromInstruction(llvm::Instruction *I) {
  return getConstQualifiedExtractedValuesFromInstruction(I);
}

llvm::SmallVector<llvm::SmallPtrSet<const llvm::CallInst *, 2>, 2>
getExtractedValuesFromInstruction(const llvm::Instruction *I) {
  return getConstQualifiedExtractedValuesFromInstruction(I);
}

void setSegmentKeyMetadata(llvm::Function &SegmentRefFunction,
                           MetaAddress StartAddress) {
  using namespace llvm;

  auto &Context = SegmentRefFunction.getContext();

  QuickMetadata QMD(Context);

  auto *SAMD = QMD.get(StartAddress.toString());
  revng_assert(SAMD != nullptr);
  SegmentRefFunction.setMetadata(FunctionTags::UniqueIDMDName,
                                 QMD.tuple({ SAMD }));
}

bool hasSegmentKeyMetadata(const llvm::Function &F) {
  auto &Context = F.getContext();
  auto SegmentRefMDKind = Context.getMDKindID(FunctionTags::UniqueIDMDName);
  return nullptr != F.getMetadata(SegmentRefMDKind);
}

MetaAddress extractSegmentKeyFromMetadata(const llvm::Function &F) {
  using namespace llvm;
  revng_assert(hasSegmentKeyMetadata(F));

  auto &Context = F.getContext();

  auto SegmentRefMDKind = Context.getMDKindID(FunctionTags::UniqueIDMDName);
  auto *Node = F.getMetadata(SegmentRefMDKind);

  auto *SAMD = cast<MDString>(Node->getOperand(0));
  MetaAddress StartAddress = MetaAddress::fromString(SAMD->getString());
  revng_assert(StartAddress.isValid());
  return StartAddress;
}

// This name corresponds to a function in `early-linked`.
RegisterIRHelper AbortHelper(AbortFunctionName.str());

template<bool ShouldTerminateTheBlock>
llvm::CallInst &emitMessageImpl(revng::IRBuilder &Builder,
                                const llvm::Twine &Message,
                                const llvm::DebugLoc &DbgLocation,
                                const ProgramCounterHandler *PCH) {
  using namespace llvm;

  // Create the function if there's not already one.
  Module *M = getModule(Builder.GetInsertBlock());
  auto *FT = createFunctionType<void, const uint8_t *>(M->getContext());
  auto Callee = getOrInsertIRHelper(AbortFunctionName, *M, FT);

  // Ensure it's marked as a helper.
  Function *F = cast<Function>(Callee.getCallee());
  if (not FunctionTags::Helper.isTagOf(F))
    FunctionTags::Helper.addTo(F);

  // Optionally update the program counter.
  if (PCH != nullptr) {
    MetaAddress SourcePC = MetaAddress::invalid();

    if (Instruction *T = Builder.GetInsertBlock()->getTerminator())
      SourcePC = getPC(T).first;

    PCH->setLastPCPlainMetaAddress(Builder, SourcePC);
    PCH->setCurrentPCPlainMetaAddress(Builder);
  }

  llvm::DebugLoc DebugLocation = DbgLocation ?
                                   DbgLocation :
                                   Builder.getCurrentDebugLocation();

  // Create the call.
  auto *NewCall = Builder.CreateCall(Callee, getUniqueString(M, Message.str()));
  NewCall->setDebugLoc(DebugLocation);

  if constexpr (ShouldTerminateTheBlock) {
    // Add an unreachable mark after this call.
    Instruction *T = Builder.CreateUnreachable();
    T->setDebugLoc(DebugLocation);

    // Assert there's one and only one terminator
    auto *BB = Builder.GetInsertBlock();
    unsigned Terminators = 0;
    for (Instruction &I : *BB)
      if (I.isTerminator())
        ++Terminators;
    revng_assert(Terminators == 1,
                 "There's already a terminator in this basic block. "
                 "Did you mean to use `emitMessage` instead?");
  }

  return *NewCall;
}

llvm::CallInst &emitAbort(revng::IRBuilder &Builder,
                          const llvm::Twine &Message,
                          const llvm::DebugLoc &DbgLocation,
                          const ProgramCounterHandler *PCH) {
  return emitMessageImpl<true>(Builder, Message, DbgLocation, PCH);
}

llvm::CallInst &emitMessage(revng::IRBuilder &Builder,
                            const llvm::Twine &Message,
                            const llvm::DebugLoc &DbgLocation,
                            const ProgramCounterHandler *PCH) {
  return emitMessageImpl<false>(Builder, Message, DbgLocation, PCH);
}

llvm::FunctionType *getAddressOfType(llvm::Type *RetType,
                                     llvm::Type *BaseType) {
  // There are 2 fixed arguments:
  // - the first is a pointer to a constant string that contains a serialization
  //   of the key of the base type;
  // - the second is BaseType, i.e. the type of the base pointer.
  auto &C = RetType->getContext();
  llvm::SmallVector<llvm::Type *, 2> FixedArgs = { getStringPtrType(C),
                                                   BaseType };
  return llvm::FunctionType::get(RetType, FixedArgs, false /* IsVarArg */);
}

llvm::FunctionType *getLocalVarType(llvm::Type *ReturnedType) {
  using namespace llvm;

  // There only argument is a pointer to a constant string that contains a
  // serialization of the allocated variable's type
  auto &C = ReturnedType->getContext();
  SmallVector<llvm::Type *, 1> FixedArgs = { getStringPtrType(C) };
  return FunctionType::get(ReturnedType, FixedArgs, false /* IsVarArg */);
}

llvm::FunctionType *getOpaqueEVFunctionType(llvm::ExtractValueInst *Extract) {
  using namespace llvm;

  revng_assert(Extract->getNumIndices() == 1);

  // The first argument is the struct we are extracting from, the second is the
  // index, with i64 type.
  std::vector<llvm::Type *> ArgTypes = {
    Extract->getAggregateOperand()->getType(),
    IntegerType::getInt64Ty(Extract->getContext())
  };

  // The return type is the type of the extracted field
  Type *ReturnType = Extract->getType();

  return FunctionType::get(ReturnType, ArgTypes, false);
}

llvm::FunctionType *getAssignFunctionType(llvm::Type *ValueType,
                                          llvm::Type *PtrType) {
  llvm::SmallVector<llvm::Type *, 2> FixedArgs = { ValueType, PtrType };
  auto &C = ValueType->getContext();
  return llvm::FunctionType::get(llvm::Type::getVoidTy(C),
                                 FixedArgs,
                                 false /* IsVarArg */);
}

llvm::FunctionType *getCopyType(llvm::Type *ReturnedType,
                                llvm::Type *VariableReferenceType) {
  using namespace llvm;
  // The argument is an llvm::Value representing a reference
  // It's not part of the key in the Copy pool, because all references should
  // have the same underlying LLVM type, which is a pointer-sized integer.
  // This is a hack, but Copy will go away in the clift-base decompilation
  // pipeline, so it's temporary.
  SmallVector<llvm::Type *, 1> FixedArgs = { VariableReferenceType };
  return FunctionType::get(ReturnedType, FixedArgs, false /* IsVarArg */);
}

static std::vector<llvm::GlobalVariable *> extractCSVs(llvm::Function *F,
                                                       unsigned MDKindID) {
  using namespace llvm;

  std::vector<GlobalVariable *> Result;
  auto *Tuple = cast_or_null<MDTuple>(F->getMetadata(MDKindID));
  if (Tuple == nullptr)
    return Result;

  llvm::Module *M = F->getParent();
  QuickMetadata QMD(M->getContext());

  auto OperandsRange = QMD.extract<MDTuple *>(Tuple, 1)->operands();
  for (const MDOperand &Operand : OperandsRange) {
    if (Metadata *MD = Operand.get()) {
      auto CSVName = QMD.extract<StringRef>(MD);

      // Note: here we record the *names* of CSVs as opposed to a
      // ConstantAsMetadata pointing to the GlobalVariable because otherwise,
      // during linking, these get null-ified.
      if (auto *CSV = M->getGlobalVariable(CSVName, true))
        Result.push_back(CSV);
    }
  }

  return Result;
}

std::optional<CSVsUsage>
tryGetCSVUsedByHelperCall(const llvm::Instruction *Call) {
  revng_assert(isCallToHelper(Call));

  auto *Callee = getCalledFunction(cast<llvm::CallBase>(Call));

  const llvm::Module *M = getModule(Call);
  const auto LoadMDKind = M->getMDKindID("revng.csvaccess.offsets.load");
  const auto StoreMDKind = M->getMDKindID("revng.csvaccess.offsets.store");

  if (Callee->getMetadata(LoadMDKind) == nullptr
      and Callee->getMetadata(StoreMDKind) == nullptr) {
    return {};
  }

  CSVsUsage Result;
  Result.Read = extractCSVs(Callee, LoadMDKind);
  Result.Written = extractCSVs(Callee, StoreMDKind);
  return Result;
}

const llvm::CallInst *getCallToIsolatedFunction(const llvm::Value *V) {
  if (const llvm::CallInst *Call = getCallToTagged(V, FunctionTags::Isolated)) {
    // The callee is an isolated function
    return Call;
  } else if (const llvm::CallInst
               *Call = getCallToTagged(V, FunctionTags::DynamicFunction)) {
    // The callee is a dynamic function
    return Call;
  } else if (auto *Call = dyn_cast<llvm::CallInst>(V)) {
    // It's a call to an isolated function if it's indirect
    return getCalledFunction(Call) == nullptr ? Call : nullptr;
  } else {
    return nullptr;
  }
}

llvm::CallInst *getCallToIsolatedFunction(llvm::Value *V) {
  if (llvm::CallInst *Call = getCallToTagged(V, FunctionTags::Isolated)) {
    // The callee is an isolated function
    return Call;
  } else if (llvm::CallInst
               *Call = getCallToTagged(V, FunctionTags::DynamicFunction)) {
    // The callee is a dynamic function
    return Call;
  } else if (auto *Call = dyn_cast<llvm::CallInst>(V)) {
    // It's a call to an isolated function if it's indirect
    return getCalledFunction(Call) == nullptr ? Call : nullptr;
  } else {
    return nullptr;
  }
}
