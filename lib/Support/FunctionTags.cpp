//
// Copyright (c) rev.ng Labs Srl. See LICENSE.md for details.
//

#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"

#include "revng/Support/Assert.h"
#include "revng/Support/IRHelpers.h"
#include "revng/Support/MetaAddress.h"
#include "revng/Support/OpaqueFunctionsPool.h"

#include "revng-c/Support/FunctionTags.h"
#include "revng-c/Support/IRHelpers.h"
#include "revng-c/Support/Mangling.h"

static constexpr const char *const ModelGEPName = "ModelGEP";

namespace FunctionTags {
Tag AllocatesLocalVariable("AllocatesLocalVariable");
Tag MallocLike("MallocLike");
Tag IsRef("IsRef");
Tag AddressOf("AddressOf");
Tag StringLiteral("StringLiteral");
Tag ModelCast("ModelCast");
Tag ModelGEP(ModelGEPName);
Tag ModelGEPRef("ModelGEPRef");
Tag OpaqueExtractValue("OpaqueExtractvalue");
Tag Parentheses("Parentheses");
Tag HexInteger("HexInteger");
Tag CharInteger("CharInteger");
Tag BoolInteger("BoolInteger");
Tag LocalVariable("LocalVariable");
Tag Assign("Assign");
Tag Copy("Copy");
Tag WritesMemory("WritesMemory");
Tag ReadsMemory("ReadsMemory");
Tag SegmentRef("SegmentRef");
Tag UnaryMinus("UnaryMinus");
Tag BinaryNot("BinaryNot");
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

static std::string makeModelGEPName(const llvm::Type *RetTy,
                                    const llvm::Type *BaseAddressTy,
                                    llvm::StringRef Prefix) {
  using llvm::Twine;
  return (Prefix + Twine("_ret_") + Twine(makeTypeName(RetTy))
          + Twine("_baseptr_") + Twine(makeTypeName(BaseAddressTy)))
    .str();
}

llvm::FunctionType *
getAddressOfType(llvm::Type *RetType, llvm::Type *BaseType) {
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

  // Set revng tags
  Pool.setTags({ &FunctionTags::AddressOf });

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
  Pool.setTags({ &FunctionTags::StringLiteral });

  // Initialize the pool
  for (llvm::Function &F : FunctionTags::StringLiteral.functions(M)) {
    const auto &[StartAddress,
                 VirtualSize,
                 Offset,
                 StrLen] = extractStringLiteralFromMetadata(F);
    auto Type = F.getFunctionType()->getReturnType();
    StringLiteralPoolKey Key = { StartAddress, VirtualSize, Offset, Type };
    Pool.record(Key, &F);
  }
}

void initModelCastPool(OpaqueFunctionsPool<llvm::Type *> &Pool) {
  // Set attributes
  Pool.addFnAttribute(llvm::Attribute::NoUnwind);
  Pool.addFnAttribute(llvm::Attribute::WillReturn);
  Pool.setMemoryEffects(llvm::MemoryEffects::none());

  // Set revng tags
  Pool.setTags({ &FunctionTags::ModelCast });

  // Initialize the pool from its internal llvm::Module if possible.
  Pool.initializeFromReturnType(FunctionTags::ModelCast);
}

void initParenthesesPool(OpaqueFunctionsPool<llvm::Type *> &Pool) {
  // Set attributes
  Pool.addFnAttribute(llvm::Attribute::NoUnwind);
  Pool.addFnAttribute(llvm::Attribute::WillReturn);
  Pool.setMemoryEffects(llvm::MemoryEffects::none());

  // Set revng tags
  Pool.setTags({ &FunctionTags::Parentheses, &FunctionTags::Marker });

  // Initialize the pool from its internal llvm::Module if possible.
  Pool.initializeFromReturnType(FunctionTags::Parentheses);
}

void initHexPrintPool(OpaqueFunctionsPool<llvm::Type *> &Pool) {
  // Set attributes
  Pool.addFnAttribute(llvm::Attribute::NoUnwind);
  Pool.addFnAttribute(llvm::Attribute::WillReturn);
  Pool.setMemoryEffects(llvm::MemoryEffects::none());

  // Set revng tags
  Pool.setTags({ &FunctionTags::HexInteger });

  // Initialize the pool from its internal llvm::Module if possible.
  Pool.initializeFromReturnType(FunctionTags::HexInteger);
}

void initCharPrintPool(OpaqueFunctionsPool<llvm::Type *> &Pool) {
  // Set attributes
  Pool.addFnAttribute(llvm::Attribute::NoUnwind);
  Pool.addFnAttribute(llvm::Attribute::WillReturn);
  Pool.setMemoryEffects(llvm::MemoryEffects::none());

  // Set revng tags
  Pool.setTags({ &FunctionTags::CharInteger });

  // Initialize the pool from its internal llvm::Module if possible.
  Pool.initializeFromReturnType(FunctionTags::CharInteger);
}

void initBoolPrintPool(OpaqueFunctionsPool<llvm::Type *> &Pool) {
  // Set attributes
  Pool.addFnAttribute(llvm::Attribute::NoUnwind);
  Pool.addFnAttribute(llvm::Attribute::WillReturn);
  Pool.setMemoryEffects(llvm::MemoryEffects::none());

  // Set revng tags
  Pool.setTags({ &FunctionTags::BoolInteger });

  // Initialize the pool from its internal llvm::Module if possible.
  Pool.initializeFromReturnType(FunctionTags::BoolInteger);
}

void initUnaryMinusPool(OpaqueFunctionsPool<llvm::Type *> &Pool) {
  // Set attributes
  Pool.addFnAttribute(llvm::Attribute::NoUnwind);
  Pool.addFnAttribute(llvm::Attribute::WillReturn);
  Pool.setMemoryEffects(llvm::MemoryEffects::none());

  // Set revng tags
  Pool.setTags({ &FunctionTags::UnaryMinus });

  // Initialize the pool from its internal llvm::Module if possible.
  Pool.initializeFromReturnType(FunctionTags::UnaryMinus);
}

void initBinaryNotPool(OpaqueFunctionsPool<llvm::Type *> &Pool) {
  // Set attributes
  Pool.addFnAttribute(llvm::Attribute::NoUnwind);
  Pool.addFnAttribute(llvm::Attribute::WillReturn);
  Pool.setMemoryEffects(llvm::MemoryEffects::none());

  // Set revng tags
  Pool.setTags({ &FunctionTags::BinaryNot });

  // Initialize the pool from its internal llvm::Module if possible.
  Pool.initializeFromReturnType(FunctionTags::BinaryNot);
}

void initSegmentRefPool(OpaqueFunctionsPool<SegmentRefPoolKey> &Pool,
                        llvm::Module *M) {
  // Set attributes
  Pool.addFnAttribute(llvm::Attribute::NoUnwind);
  Pool.addFnAttribute(llvm::Attribute::WillReturn);
  Pool.setMemoryEffects(llvm::MemoryEffects::none());

  // Set revng tags
  Pool.setTags({ &FunctionTags::SegmentRef, &FunctionTags::IsRef });

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

  // There are 2 fixed arguments:
  // - the first is a pointer to a constant string that contains a serialization
  //   of the key of the base type;
  // - the second is the type of the base pointer.
  SmallVector<llvm::Type *, 2> FixedArgs = { getStringPtrType(M.getContext()),
                                             BaseType };
  // The function is vararg, because we might need to access a number of fields
  // that is variable.
  FunctionType *ModelGEPType = FunctionType::get(RetType,
                                                 FixedArgs,
                                                 true /* IsVarArg */);

  FunctionCallee
    MGEPCallee = M.getOrInsertFunction(makeModelGEPName(RetType,
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
  SmallVector<llvm::Type *, 2> FixedArgs = { getStringPtrType(M.getContext()),
                                             BaseType };
  // The function is vararg, because we might need to access a number of fields
  // that is variable.
  FunctionType *ModelGEPType = FunctionType::get(ReturnType,
                                                 FixedArgs,
                                                 true /* IsVarArg */);

  FunctionCallee MGEPCallee = M.getOrInsertFunction(makeModelGEPName(ReturnType,
                                                                     BaseType,
                                                                     "ModelGEPR"
                                                                     "ef"),
                                                    ModelGEPType);

  auto *ModelGEPFunction = cast<Function>(MGEPCallee.getCallee());
  ModelGEPFunction->addFnAttr(llvm::Attribute::NoUnwind);
  ModelGEPFunction->addFnAttr(llvm::Attribute::WillReturn);
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

  // Set revng tags
  Pool.setTags({ &FunctionTags::LocalVariable,
                 &FunctionTags::IsRef,
                 &FunctionTags::AllocatesLocalVariable });

  // Initialize the pool from its internal llvm::Module if possible.
  // Use the stored type as a key.
  Pool.initializeFromReturnType(FunctionTags::LocalVariable);
}

llvm::FunctionType *getOpaqueEVFunctionType(llvm::ExtractValueInst *Extract) {
  using namespace llvm;
  // First argument is the struct we are extracting from
  std::vector ArgTypes = { Extract->getAggregateOperand()->getType() };

  // All other arguments are indices, which we decided to be of type i64
  auto &C = Extract->getContext();
  Type *I64Type = IntegerType::getInt64Ty(C);
  ArgTypes.insert(ArgTypes.end(), Extract->getNumIndices(), I64Type);

  // The return type is the type of the extracted field
  Type *ReturnType = Extract->getType();

  return FunctionType::get(ReturnType, ArgTypes, false);
}

void initOpaqueEVPool(OpaqueFunctionsPool<TypePair> &Pool, llvm::Module *M) {
  // Don't optimize these calls
  Pool.addFnAttribute(llvm::Attribute::OptimizeNone);
  Pool.addFnAttribute(llvm::Attribute::NoInline);

  const auto &EVTag = FunctionTags::OpaqueExtractValue;
  Pool.setTags({ &EVTag });

  // Initialize the pool from its internal llvm::Module if possible.
  for (llvm::Function &F : EVTag.functions(M)) {
    auto StructType = F.getFunctionType()->getParamType(0);
    auto RetType = F.getFunctionType()->getReturnType();
    Pool.record({ RetType, StructType }, &F);
  }
}

llvm::FunctionType *
getAssignFunctionType(llvm::Type *ValueType, llvm::Type *PtrType) {
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

  // Set revng tags
  Pool.setTags({ &FunctionTags::Assign, &FunctionTags::WritesMemory });

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

  // Set revng tags
  Pool.setTags({ &FunctionTags::Copy, &FunctionTags::ReadsMemory });

  // Initialize the pool from its internal llvm::Module if possible.
  // Use the stored type as a key.
  Pool.initializeFromReturnType(FunctionTags::Copy);
}
