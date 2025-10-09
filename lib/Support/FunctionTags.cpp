//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Support/FunctionTags.h"
#include "revng/Support/ProgramCounterHandler.h"

namespace FunctionTags {

Tag QEMU("qemu");
Tag Helper("helper");

Tag Isolated("isolated");
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

Tag UniquedByPrototype("uniqued-by-prototype");

Tag UniquedByMetadata("uniqued-by-metadata");

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

FunctionPoolTag<StringLiteralPoolKey>
  StringLiteral("string-literal",
                { llvm::Attribute::NoUnwind, llvm::Attribute::WillReturn },
                llvm::MemoryEffects::none(),
                { &FunctionTags::UniquedByMetadata },
                [](OpaqueFunctionsPool<StringLiteralPoolKey> &Pool,
                   llvm::Module &M,
                   const FunctionPoolTag<StringLiteralPoolKey> &Tag) {
                  for (llvm::Function &F : Tag.functions(&M)) {
                    const auto &[StartAddress,
                                 VirtualSize,
                                 Offset,
                                 StrLen,
                                 Type] = extractStringLiteralFromMetadata(F);
                    StringLiteralPoolKey Key = {
                      StartAddress, VirtualSize, Offset, Type
                    };
                    Pool.record(Key, &F);
                  }
                });

FunctionPoolTag<TypePair>
  ModelCast("model-cast",
            { llvm::Attribute::NoUnwind, llvm::Attribute::WillReturn },
            llvm::MemoryEffects::none(),
            { &FunctionTags::UniquedByPrototype },
            [](OpaqueFunctionsPool<TypePair> &Pool,
               llvm::Module &M,
               const FunctionPoolTag<TypePair> &Tag) {
              for (llvm::Function &F : Tag.functions(&M)) {
                auto *FunctionType = F.getFunctionType();
                revng_assert(FunctionType->getNumParams() == 3);
                revng_assert(not FunctionType->isVarArg());
                auto *ReturnType = F.getFunctionType()->getReturnType();
                auto *OperandToCastType = F.getFunctionType()->getParamType(1);
                Pool.record({ ReturnType, OperandToCastType }, &F);
              }
            });

Tag ModelGEP("model-gep");
Tag ModelGEPRef("model-gep-ref");

FunctionPoolTag<TypePair>
  OpaqueExtractValue("opaque-extract-value",
                     { llvm::Attribute::OptimizeNone,
                       llvm::Attribute::NoInline,
                       llvm::Attribute::NoMerge,
                       llvm::Attribute::NoUnwind,
                       llvm::Attribute::WillReturn },
                     llvm::MemoryEffects::inaccessibleMemOnly()
                       | llvm::MemoryEffects::readOnly(),
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

using SegmentRefPoolKey = std::tuple<MetaAddress, uint64_t, llvm::Type *>;
FunctionPoolTag<SegmentRefPoolKey>
  SegmentRef("segment-ref",
             { llvm::Attribute::NoUnwind, llvm::Attribute::WillReturn },
             llvm::MemoryEffects::none(),
             { &FunctionTags::IsRef, &FunctionTags::UniquedByMetadata },
             [](OpaqueFunctionsPool<SegmentRefPoolKey> &Pool,
                llvm::Module &M,
                const FunctionPoolTag<SegmentRefPoolKey> &Tag) {
               for (llvm::Function &F : Tag.functions(&M)) {
                 const auto &[StartAddress,
                              VirtualSize] = extractSegmentKeyFromMetadata(F);
                 auto *RetType = F.getFunctionType()->getReturnType();

                 SegmentRefPoolKey Key = { StartAddress, VirtualSize, RetType };
                 Pool.record(Key, &F);
               }
             });

FunctionPoolTag<llvm::Type *>
  UnaryMinus("unary-minus",
             { llvm::Attribute::NoUnwind, llvm::Attribute::WillReturn },
             llvm::MemoryEffects::none(),
             { &FunctionTags::UniquedByPrototype },
             InitializationMode::InitializeFromReturnType);

FunctionPoolTag<llvm::Type *>
  BinaryNot("binary-not",
            { llvm::Attribute::NoUnwind, llvm::Attribute::WillReturn },
            llvm::MemoryEffects::none(),
            { &FunctionTags::UniquedByPrototype },
            InitializationMode::InitializeFromReturnType);

FunctionPoolTag<llvm::Type *>
  BooleanNot("boolean-not",
             { llvm::Attribute::NoUnwind, llvm::Attribute::WillReturn },
             llvm::MemoryEffects::none(),
             { &FunctionTags::UniquedByPrototype },
             InitializationMode::InitializeFromArgument0);

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
                           MetaAddress StartAddress,
                           uint64_t VirtualSize) {
  using namespace llvm;

  auto &Context = SegmentRefFunction.getContext();

  QuickMetadata QMD(Context);

  auto *SAMD = QMD.get(StartAddress.toString());
  revng_assert(SAMD != nullptr);

  auto *VSConstant = ConstantInt::get(Type::getInt64Ty(Context), VirtualSize);
  auto *VSMD = ConstantAsMetadata::get(VSConstant);

  SegmentRefFunction.setMetadata(FunctionTags::UniqueIDMDName,
                                 QMD.tuple({ SAMD, VSMD }));
}

bool hasSegmentKeyMetadata(const llvm::Function &F) {
  auto &Context = F.getContext();
  auto SegmentRefMDKind = Context.getMDKindID(FunctionTags::UniqueIDMDName);
  return nullptr != F.getMetadata(SegmentRefMDKind);
}

std::pair<MetaAddress, uint64_t>
extractSegmentKeyFromMetadata(const llvm::Function &F) {
  using namespace llvm;
  revng_assert(hasSegmentKeyMetadata(F));

  auto &Context = F.getContext();

  auto SegmentRefMDKind = Context.getMDKindID(FunctionTags::UniqueIDMDName);
  auto *Node = F.getMetadata(SegmentRefMDKind);

  auto *SAMD = cast<MDString>(Node->getOperand(0));
  MetaAddress StartAddress = MetaAddress::fromString(SAMD->getString());
  revng_assert(StartAddress.isValid());
  auto *VSMD = cast<ConstantAsMetadata>(Node->getOperand(1))->getValue();
  uint64_t VirtualSize = cast<ConstantInt>(VSMD)->getZExtValue();

  return { StartAddress, VirtualSize };
}

void setStringLiteralMetadata(llvm::Function &StringLiteralFunction,
                              MetaAddress StartAddress,
                              uint64_t VirtualSize,
                              uint64_t Offset,
                              uint64_t StringLength,
                              llvm::Type *ReturnType) {
  using namespace llvm;

  auto *M = StringLiteralFunction.getParent();
  auto &Context = StringLiteralFunction.getContext();

  QuickMetadata QMD(M->getContext());
  auto StringLiteralMDKind = Context.getMDKindID(FunctionTags::UniqueIDMDName);

  auto *SAMD = QMD.get(StartAddress.toString());

  auto *VSConstant = ConstantInt::get(Type::getInt64Ty(Context), VirtualSize);
  auto *VSMD = ConstantAsMetadata::get(VSConstant);

  auto *OffsetConstant = ConstantInt::get(Type::getInt64Ty(Context), Offset);
  auto *OffsetMD = ConstantAsMetadata::get(OffsetConstant);

  auto *StrLenConstant = ConstantInt::get(Type::getInt64Ty(Context),
                                          StringLength);
  auto *StrLenMD = ConstantAsMetadata::get(StrLenConstant);

  unsigned Value = ReturnType->isPointerTy() ? 0 :
                                               ReturnType->getIntegerBitWidth();
  auto *ReturnTypeConstant = ConstantInt::get(Type::getInt64Ty(Context), Value);
  auto *ReturnTypeMD = ConstantAsMetadata::get(ReturnTypeConstant);

  auto QMDTuple = QMD.tuple({ SAMD, VSMD, OffsetMD, StrLenMD, ReturnTypeMD });
  StringLiteralFunction.setMetadata(StringLiteralMDKind, QMDTuple);
}

bool hasStringLiteralMetadata(const llvm::Function &F) {
  auto &Context = F.getContext();
  auto StringLiteralMDKind = Context.getMDKindID(FunctionTags::UniqueIDMDName);
  return nullptr != F.getMetadata(StringLiteralMDKind);
}

std::tuple<MetaAddress, uint64_t, uint64_t, uint64_t, llvm::Type *>
extractStringLiteralFromMetadata(const llvm::Function &F) {
  using namespace llvm;
  revng_assert(hasStringLiteralMetadata(F));

  auto &Context = F.getContext();

  auto StringLiteralMDKind = Context.getMDKindID(FunctionTags::UniqueIDMDName);
  auto *Node = F.getMetadata(StringLiteralMDKind);

  StringRef SAMD = cast<MDString>(Node->getOperand(0))->getString();
  MetaAddress StartAddress = MetaAddress::fromString(SAMD);
  revng_assert(StartAddress.isValid());

  auto ExtractInteger = [](const MDOperand &Operand) {
    auto *MD = cast<ConstantAsMetadata>(Operand)->getValue();
    return cast<ConstantInt>(MD)->getZExtValue();
  };

  uint64_t VirtualSize = ExtractInteger(Node->getOperand(1));
  uint64_t Offset = ExtractInteger(Node->getOperand(2));
  uint64_t StrLen = ExtractInteger(Node->getOperand(3));
  uint64_t ReturnTypeLength = ExtractInteger(Node->getOperand(4));
  llvm::Type *PointerType = llvm::PointerType::get(Context, 0);
  llvm::Type *ReturnType = ReturnTypeLength == 0 ?
                             PointerType :
                             llvm::IntegerType::get(Context, ReturnTypeLength);

  return { StartAddress, VirtualSize, Offset, StrLen, ReturnType };
}

// This name corresponds to a function in `early-linked`.
RegisterIRHelper RevngAbortHelper(AbortFunctionName.str());

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

static constexpr const char *const ModelGEPName = "ModelGEP";
static constexpr const char *const ModelGEPRefName = "ModelGEPRef";

// This is very simple for now.
// In the future we might consider making it more robust using something like
// Punycode https://tools.ietf.org/html/rfc3492 , which also has the nice
// property of being deterministically reversible.
static std::string makeCIdentifier(std::string S) {
  llvm::for_each(S, [](char &C) {
    if (not std::isalnum(C))
      C = '_';
  });
  return S;
}

static std::string makeTypeName(const llvm::Type *Ty) {
  std::string Name;
  if (auto *PtrTy = llvm::dyn_cast<llvm::PointerType>(Ty)) {
    Name = "ptr";
  } else if (auto *IntTy = llvm::dyn_cast<llvm::IntegerType>(Ty)) {
    Name = "i" + std::to_string(IntTy->getBitWidth());
  } else if (auto *StrucTy = llvm::dyn_cast<llvm::StructType>(Ty)) {
    Name = "struct_";
    Name += std::to_string(reinterpret_cast<uint64_t>(Ty));
    if (StrucTy->isLiteral() or not StrucTy->hasName()) {
      for (const auto *FieldTy : StrucTy->elements())
        Name += "_" + makeTypeName(FieldTy);
    } else {
      Name += "_" + makeCIdentifier(StrucTy->getStructName().str());
    }
  } else if (auto *FunTy = llvm::dyn_cast<llvm::FunctionType>(Ty)) {
    Name = "func_" + makeTypeName(FunTy->getReturnType());
    if (not FunTy->params().empty()) {
      Name += "_args";
      for (const auto &ArgT : FunTy->params())
        Name += "_" + makeTypeName(ArgT);
    }
  } else if (Ty->isVoidTy()) {
    Name += "void";
  } else {
    revng_unreachable("cannot build Type name");
  }
  return Name;
}

static std::string makeTypeBasedSuffix(const llvm::Type *RetTy,
                                       const llvm::Type *BaseAddressTy,
                                       llvm::StringRef Prefix) {
  using llvm::Twine;
  return (Prefix + Twine("_ret_") + Twine(makeTypeName(RetTy))
          + Twine("_baseptr_") + Twine(makeTypeName(BaseAddressTy)))
    .str();
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

llvm::Function *
getModelGEP(llvm::Module &M, llvm::Type *RetType, llvm::Type *BaseType) {

  using namespace llvm;

  // There are 3 fixed arguments:
  // - the first is a pointer to a constant string that contains a serialization
  //   of the key of the base type;
  // - the second is the type of the base pointer.
  // - the third argument represents the member of the array access based on the
  //   second. if it's 0 it's a regular pointer access, otherwise an array
  //   access.
  auto *Int64Type = llvm::IntegerType::getIntNTy(M.getContext(), 64);
  SmallVector<llvm::Type *, 3> FixedArgs = { getStringPtrType(M.getContext()),
                                             BaseType,
                                             Int64Type };
  // The function is vararg, because we might need to access a number of fields
  // that is variable.
  FunctionType *ModelGEPType = FunctionType::get(RetType,
                                                 FixedArgs,

                                                 true /* IsVarArg */);

  FunctionCallee
    MGEPCallee = M.getOrInsertFunction(makeTypeBasedSuffix(RetType,
                                                           BaseType,
                                                           ModelGEPName),
                                       ModelGEPType);

  auto *ModelGEPFunction = cast<Function>(MGEPCallee.getCallee());
  ModelGEPFunction->addFnAttr(llvm::Attribute::NoUnwind);
  ModelGEPFunction->addFnAttr(llvm::Attribute::WillReturn);
  ModelGEPFunction->setMemoryEffects(llvm::MemoryEffects::none());
  FunctionTags::ModelGEP.addTo(ModelGEPFunction);
  FunctionTags::IsRef.addTo(ModelGEPFunction);

  return ModelGEPFunction;
}

llvm::Function *
getModelGEPRef(llvm::Module &M, llvm::Type *ReturnType, llvm::Type *BaseType) {

  using namespace llvm;
  // There are 2 fixed arguments:
  // - the first is a pointer to a constant string that contains a serialization
  //   of the key of the base type;
  // - the second is the type of the base pointer.
  //
  // Notice that, unlike ModelGEP, ModelGEPRef doesn't have a mandatory third
  // argument to represent the array access, because in case of reference
  // there's no way to do an array-like access
  SmallVector<llvm::Type *, 2> FixedArgs = { getStringPtrType(M.getContext()),
                                             BaseType };
  // The function is vararg, because we might need to access a number of fields
  // that is variable.
  FunctionType *ModelGEPType = FunctionType::get(ReturnType,
                                                 FixedArgs,
                                                 true /* IsVarArg */);

  FunctionCallee
    MGEPCallee = M.getOrInsertFunction(makeTypeBasedSuffix(ReturnType,
                                                           BaseType,
                                                           ModelGEPRefName),
                                       ModelGEPType);

  auto *ModelGEPFunction = cast<Function>(MGEPCallee.getCallee());
  ModelGEPFunction->addFnAttr(llvm::Attribute::NoUnwind);
  ModelGEPFunction->addFnAttr(llvm::Attribute::WillReturn);

  // This is NoMerge, because merging two of them would cause a PHINode among
  // IsRef opcodes.
  ModelGEPFunction->addFnAttr(llvm::Attribute::NoMerge);

  ModelGEPFunction->setMemoryEffects(llvm::MemoryEffects::none());
  FunctionTags::ModelGEPRef.addTo(ModelGEPFunction);
  FunctionTags::IsRef.addTo(ModelGEPFunction);

  return ModelGEPFunction;
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
