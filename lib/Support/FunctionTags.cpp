//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"

#include "revng/Support/Assert.h"
#include "revng/Support/FunctionTags.h"
#include "revng/Support/IRHelpers.h"
#include "revng/Support/MetaAddress.h"
#include "revng/Support/OpaqueFunctionsPool.h"

#include "revng-c/Support/FunctionTags.h"
#include "revng-c/Support/IRHelpers.h"
#include "revng-c/Support/Mangling.h"

static constexpr const char *const ModelGEPName = "ModelGEP";
static constexpr const char *const ModelGEPRefName = "ModelGEPRef";

namespace FunctionTags {
Tag AllocatesLocalVariable("allocates-local-variable");
Tag ReturnsPolymorphic("returns-polymorphic");
Tag IsRef("is-ref");
Tag AddressOf("address-of");
Tag StringLiteral("string-literal");
Tag ModelCast("model-cast");
Tag ModelGEP("model-gep");
Tag ModelGEPRef("model-gep-ref");
Tag OpaqueExtractValue("opaque-extract-value");
Tag Parentheses("parentheses");
Tag LiteralPrintDecorator("literal-print-decorator");
Tag HexInteger("hex-integer");
Tag CharInteger("char-integer");
Tag BoolInteger("bool-integer");
Tag NullPtr("nullptr");
Tag LocalVariable("local-variable");
Tag Assign("assign");
Tag Copy("copy");
Tag SegmentRef("segment-ref");
Tag UnaryMinus("unary-minus");
Tag BinaryNot("binary-not");
Tag BooleanNot("boolean-not");
} // namespace FunctionTags

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

void initAddressOfPool(OpaqueFunctionsPool<TypePair> &Pool, llvm::Module *M) {
  // Set attributes
  Pool.addFnAttribute(llvm::Attribute::NoUnwind);
  Pool.addFnAttribute(llvm::Attribute::WillReturn);
  Pool.setMemoryEffects(llvm::MemoryEffects::none());

  // This is NoMerge, because merging two of them would cause a PHINode among
  // IsRef opcodes.
  Pool.addFnAttribute(llvm::Attribute::NoMerge);

  // Set revng tags
  Pool.setTags({ &FunctionTags::AddressOf, &FunctionTags::UniquedByPrototype });

  // Initialize the pool from its internal llvm::Module if possible.
  auto &AddressOfTag = FunctionTags::AddressOf;
  for (llvm::Function &F : AddressOfTag.functions(M)) {
    auto *ArgType = F.getFunctionType()->getParamType(1);
    auto *RetType = F.getFunctionType()->getReturnType();
    Pool.record({ RetType, ArgType }, &F);
  }
}

void initStringLiteralPool(OpaqueFunctionsPool<StringLiteralPoolKey> &Pool,
                           llvm::Module *M) {
  // Set attributes
  Pool.addFnAttribute(llvm::Attribute::NoUnwind);
  Pool.addFnAttribute(llvm::Attribute::WillReturn);
  Pool.setMemoryEffects(llvm::MemoryEffects::none());

  // Set revng tags
  Pool.setTags({ &FunctionTags::StringLiteral,
                 &FunctionTags::UniquedByMetadata });

  // Initialize the pool
  for (llvm::Function &F : FunctionTags::StringLiteral.functions(M)) {
    const auto &[StartAddress,
                 VirtualSize,
                 Offset,
                 StrLen,
                 Type] = extractStringLiteralFromMetadata(F);
    StringLiteralPoolKey Key = { StartAddress, VirtualSize, Offset, Type };
    Pool.record(Key, &F);
  }
}

void initModelCastPool(OpaqueFunctionsPool<TypePair> &Pool, llvm::Module *M) {
  // Set attributes
  Pool.addFnAttribute(llvm::Attribute::NoUnwind);
  Pool.addFnAttribute(llvm::Attribute::WillReturn);
  Pool.setMemoryEffects(llvm::MemoryEffects::none());

  // Set revng tags
  Pool.setTags({ &FunctionTags::ModelCast, &FunctionTags::UniquedByPrototype });

  // Initialize the pool from its internal llvm::Module if possible.
  for (llvm::Function &F : FunctionTags::ModelCast.functions(M)) {
    auto *FunctionType = F.getFunctionType();
    revng_assert(FunctionType->getNumParams() == 3);
    revng_assert(not FunctionType->isVarArg());
    auto *ReturnType = F.getFunctionType()->getReturnType();
    auto *OperandToCastType = F.getFunctionType()->getParamType(1);
    Pool.record({ ReturnType, OperandToCastType }, &F);
  }
}

void initParenthesesPool(OpaqueFunctionsPool<llvm::Type *> &Pool) {
  // Set attributes
  Pool.addFnAttribute(llvm::Attribute::NoUnwind);
  Pool.addFnAttribute(llvm::Attribute::WillReturn);
  Pool.setMemoryEffects(llvm::MemoryEffects::none());

  // Set revng tags
  Pool.setTags({ &FunctionTags::Parentheses,
                 &FunctionTags::UniquedByPrototype });

  // Initialize the pool from its internal llvm::Module if possible.
  Pool.initializeFromReturnType(FunctionTags::Parentheses);
}

void initHexPrintPool(OpaqueFunctionsPool<llvm::Type *> &Pool) {
  // Set attributes
  Pool.addFnAttribute(llvm::Attribute::NoUnwind);
  Pool.addFnAttribute(llvm::Attribute::WillReturn);
  Pool.setMemoryEffects(llvm::MemoryEffects::none());

  // Set revng tags
  Pool.setTags({ &FunctionTags::HexInteger,
                 &FunctionTags::LiteralPrintDecorator,
                 &FunctionTags::UniquedByPrototype });

  // Initialize the pool from its internal llvm::Module if possible.
  Pool.initializeFromReturnType(FunctionTags::HexInteger);
}

void initCharPrintPool(OpaqueFunctionsPool<llvm::Type *> &Pool) {
  // Set attributes
  Pool.addFnAttribute(llvm::Attribute::NoUnwind);
  Pool.addFnAttribute(llvm::Attribute::WillReturn);
  Pool.setMemoryEffects(llvm::MemoryEffects::none());

  // Set revng tags
  Pool.setTags({ &FunctionTags::CharInteger,
                 &FunctionTags::LiteralPrintDecorator,
                 &FunctionTags::UniquedByPrototype });

  // Initialize the pool from its internal llvm::Module if possible.
  Pool.initializeFromReturnType(FunctionTags::CharInteger);
}

void initBoolPrintPool(OpaqueFunctionsPool<llvm::Type *> &Pool) {
  // Set attributes
  Pool.addFnAttribute(llvm::Attribute::NoUnwind);
  Pool.addFnAttribute(llvm::Attribute::WillReturn);
  Pool.setMemoryEffects(llvm::MemoryEffects::none());

  // Set revng tags
  Pool.setTags({ &FunctionTags::BoolInteger,
                 &FunctionTags::LiteralPrintDecorator,
                 &FunctionTags::UniquedByPrototype });

  // Initialize the pool from its internal llvm::Module if possible.
  Pool.initializeFromReturnType(FunctionTags::BoolInteger);
}

void initNullPtrPrintPool(OpaqueFunctionsPool<llvm::Type *> &Pool) {
  // Set attributes
  Pool.addFnAttribute(llvm::Attribute::NoUnwind);
  Pool.addFnAttribute(llvm::Attribute::WillReturn);
  Pool.setMemoryEffects(llvm::MemoryEffects::none());

  // Set revng tags
  Pool.setTags({ &FunctionTags::NullPtr,
                 &FunctionTags::LiteralPrintDecorator,
                 &FunctionTags::UniquedByPrototype });

  // Initialize the pool from its internal llvm::Module if possible.
  Pool.initializeFromReturnType(FunctionTags::NullPtr);
}

void initUnaryMinusPool(OpaqueFunctionsPool<llvm::Type *> &Pool) {
  // Set attributes
  Pool.addFnAttribute(llvm::Attribute::NoUnwind);
  Pool.addFnAttribute(llvm::Attribute::WillReturn);
  Pool.setMemoryEffects(llvm::MemoryEffects::none());

  // Set revng tags
  Pool.setTags({ &FunctionTags::UnaryMinus,
                 &FunctionTags::UniquedByPrototype });

  // Initialize the pool from its internal llvm::Module if possible.
  Pool.initializeFromReturnType(FunctionTags::UnaryMinus);
}

void initBinaryNotPool(OpaqueFunctionsPool<llvm::Type *> &Pool) {
  // Set attributes
  Pool.addFnAttribute(llvm::Attribute::NoUnwind);
  Pool.addFnAttribute(llvm::Attribute::WillReturn);
  Pool.setMemoryEffects(llvm::MemoryEffects::none());

  // Set revng tags
  Pool.setTags({ &FunctionTags::BinaryNot, &FunctionTags::UniquedByPrototype });

  // Initialize the pool from its internal llvm::Module if possible.
  Pool.initializeFromReturnType(FunctionTags::BinaryNot);
}

void initBooleanNotPool(OpaqueFunctionsPool<llvm::Type *> &Pool) {
  // Set attributes
  Pool.addFnAttribute(llvm::Attribute::NoUnwind);
  Pool.addFnAttribute(llvm::Attribute::WillReturn);
  Pool.setMemoryEffects(llvm::MemoryEffects::none());

  // Set revng tags
  Pool.setTags({ &FunctionTags::BooleanNot,
                 &FunctionTags::UniquedByPrototype });

  // Initialize the pool from its internal llvm::Module if possible.
  Pool.initializeFromNthArgType(FunctionTags::BooleanNot, 0);
}

void initSegmentRefPool(OpaqueFunctionsPool<SegmentRefPoolKey> &Pool,
                        llvm::Module *M) {
  // Set attributes
  Pool.addFnAttribute(llvm::Attribute::NoUnwind);
  Pool.addFnAttribute(llvm::Attribute::WillReturn);
  Pool.setMemoryEffects(llvm::MemoryEffects::none());

  // Set revng tags
  Pool.setTags({ &FunctionTags::SegmentRef,
                 &FunctionTags::IsRef,
                 &FunctionTags::UniquedByMetadata });

  // Initialize the pool
  for (llvm::Function &F : FunctionTags::SegmentRef.functions(M)) {
    const auto &[StartAddress, VirtualSize] = extractSegmentKeyFromMetadata(F);
    auto *RetType = F.getFunctionType()->getReturnType();

    SegmentRefPoolKey Key = { StartAddress, VirtualSize, RetType };
    Pool.record(Key, &F);
  }
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

void initLocalVarPool(OpaqueFunctionsPool<llvm::Type *> &Pool) {
  // Set attributes
  Pool.addFnAttribute(llvm::Attribute::NoUnwind);
  Pool.addFnAttribute(llvm::Attribute::WillReturn);
  Pool.setMemoryEffects(llvm::MemoryEffects::none());

  // NoMerge because merging two of them would merge to local variables
  Pool.addFnAttribute(llvm::Attribute::NoMerge);

  // Set revng tags
  Pool.setTags({ &FunctionTags::LocalVariable,
                 &FunctionTags::IsRef,
                 &FunctionTags::AllocatesLocalVariable,
                 &FunctionTags::ReturnsPolymorphic,
                 &FunctionTags::UniquedByPrototype });

  // Initialize the pool from its internal llvm::Module if possible.
  // Use the stored type as a key.
  Pool.initializeFromReturnType(FunctionTags::LocalVariable);
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

void initOpaqueEVPool(OpaqueFunctionsPool<TypePair> &Pool, llvm::Module *M) {
  // Don't optimize these calls
  Pool.addFnAttribute(llvm::Attribute::OptimizeNone);
  Pool.addFnAttribute(llvm::Attribute::NoInline);
  // This is NoMerge because we make strong assumptions about how
  // OpaqueExtractValues are placed in the CFG in relationship with the CallInst
  // they extract values from. Without NoMerge, those assumptions would fail.
  Pool.addFnAttribute(llvm::Attribute::NoMerge);
  Pool.addFnAttribute(llvm::Attribute::NoUnwind);
  Pool.addFnAttribute(llvm::Attribute::WillReturn);
  Pool.setMemoryEffects(llvm::MemoryEffects::inaccessibleMemOnly()
                        | llvm::MemoryEffects::readOnly());

  const auto &EVTag = FunctionTags::OpaqueExtractValue;
  Pool.setTags({ &EVTag, &FunctionTags::UniquedByPrototype });

  // Initialize the pool from its internal llvm::Module if possible.
  for (llvm::Function &F : EVTag.functions(M)) {
    auto StructDefinition = F.getFunctionType()->getParamType(0);
    auto RetType = F.getFunctionType()->getReturnType();
    Pool.record({ RetType, StructDefinition }, &F);
  }
}

llvm::FunctionType *getAssignFunctionType(llvm::Type *ValueType,
                                          llvm::Type *PtrType) {
  llvm::SmallVector<llvm::Type *, 2> FixedArgs = { ValueType, PtrType };
  auto &C = ValueType->getContext();
  return llvm::FunctionType::get(llvm::Type::getVoidTy(C),
                                 FixedArgs,
                                 false /* IsVarArg */);
}

void initAssignPool(OpaqueFunctionsPool<llvm::Type *> &Pool) {
  // Set attributes
  Pool.addFnAttribute(llvm::Attribute::NoUnwind);
  Pool.addFnAttribute(llvm::Attribute::WillReturn);
  Pool.setMemoryEffects(llvm::MemoryEffects::writeOnly());
  Pool.setTags({ &FunctionTags::Assign, &FunctionTags::UniquedByPrototype });

  // Initialize the pool from its internal llvm::Module if possible.
  // Use the stored type as a key.
  Pool.initializeFromNthArgType(FunctionTags::Assign, 0);
}

llvm::FunctionType *getCopyType(llvm::Type *ReturnedType) {
  using namespace llvm;

  // The argument is an llvm::Value representing a reference
  SmallVector<llvm::Type *, 1> FixedArgs = { ReturnedType };
  return FunctionType::get(ReturnedType, FixedArgs, false /* IsVarArg */);
}

void initCopyPool(OpaqueFunctionsPool<llvm::Type *> &Pool) {
  // Set attributes
  Pool.addFnAttribute(llvm::Attribute::NoUnwind);
  Pool.addFnAttribute(llvm::Attribute::WillReturn);
  Pool.setMemoryEffects(llvm::MemoryEffects::readOnly());
  Pool.setTags({ &FunctionTags::Copy, &FunctionTags::UniquedByPrototype });

  // Initialize the pool from its internal llvm::Module if possible.
  // Use the stored type as a key.
  Pool.initializeFromReturnType(FunctionTags::Copy);
}
