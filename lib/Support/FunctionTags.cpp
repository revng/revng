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
#include "revng-c/Support/Mangling.h"

static constexpr const char *const ModelGEPName = "ModelGEP";
static constexpr const char *const ModelGEPRefName = "ModelGEPRef";

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

llvm::FunctionType *getCopyType(llvm::Type *ReturnedType) {
  using namespace llvm;

  // The argument is an llvm::Value representing a reference
  SmallVector<llvm::Type *, 1> FixedArgs = { ReturnedType };
  return FunctionType::get(ReturnedType, FixedArgs, false /* IsVarArg */);
}
