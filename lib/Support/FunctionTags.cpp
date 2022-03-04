//
// Copyright (c) rev.ng Srls. See LICENSE.md for details.
//

#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"

#include "revng/Support/Assert.h"
#include "revng/Support/IRHelpers.h"

#include "revng-c/Support/FunctionTags.h"
#include "revng-c/Support/Mangling.h"

static constexpr const char *const ModelGEPName = "ModelGEP";
static constexpr const char *const MarkerName = "SerializationMarker";
static constexpr const char *const AddressOfName = "AddressOf";

namespace FunctionTags {
Tag AllocatesLocalVariable("AllocatesLocalVariable");
Tag MallocLike("MallocLike");
Tag IsRef("IsRef");
Tag AddressOf(AddressOfName);
Tag ModelGEP(ModelGEPName);
Tag SerializationMarker(MarkerName);
} // namespace FunctionTags

static std::string makeTypeName(const llvm::Type *Ty) {
  std::string Name;
  if (auto *PtrTy = llvm::dyn_cast<llvm::PointerType>(Ty)) {
    Name = "ptr_to_" + makeTypeName(PtrTy->getElementType());
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

static std::string makeAddressOfName(const llvm::Type *Ty) {
  return AddressOfName + makeTypeName(Ty);
}

static std::string
makeModelGEPName(const llvm::Type *RetTy, const llvm::Type *BaseAddressTy) {
  using llvm::Twine;
  return (Twine(ModelGEPName) + Twine("_ret_") + Twine(makeTypeName(RetTy))
          + Twine("_baseptr_") + Twine(makeTypeName(BaseAddressTy)))
    .str();
}

static std::string makeMarkerName(const llvm::Type *Ty) {
  return MarkerName + makeTypeName(Ty);
}

llvm::Function *getAddressOf(llvm::Module &M, llvm::Type *T) {

  using namespace llvm;
  // There are 2 fixed arguments:
  // - the first is a pointer to a constant string that contains a serialization
  //   of the key of the base type;
  // - the second is T, i.e. the type of the base pointer.
  SmallVector<llvm::Type *, 2> FixedArgs = { getStringPtrType(M.getContext()),
                                             T };
  // The function is vararg, because we might need to access a number of fields
  // that is variable.
  FunctionType *AddressOfType = FunctionType::get(T,
                                                  FixedArgs,
                                                  false /* IsVarArg */);
  FunctionCallee AddressOfCallee = M.getOrInsertFunction(makeAddressOfName(T),
                                                         AddressOfType);

  auto *AddressOfFunction = cast<Function>(AddressOfCallee.getCallee());
  AddressOfFunction->addFnAttr(llvm::Attribute::NoUnwind);
  AddressOfFunction->addFnAttr(llvm::Attribute::WillReturn);
  AddressOfFunction->addFnAttr(llvm::Attribute::ReadNone);
  FunctionTags::AddressOf.addTo(AddressOfFunction);

  return AddressOfFunction;
}

llvm::Function *
getModelGEP(llvm::Module &M, llvm::Type *RetTy, llvm::Type *BaseAddressTy) {

  using namespace llvm;
  // There are 2 fixed arguments:
  // - the first is a pointer to a constant string that contains a serialization
  //   of the key of the base type;
  // - the second is BaseAddressTy, i.e. the type of the base pointer.
  SmallVector<llvm::Type *, 2> FixedArgs = { getStringPtrType(M.getContext()),
                                             BaseAddressTy };
  // The function is vararg, because we might need to access a number of fields
  // that is variable.
  FunctionType *ModelGEPType = FunctionType::get(RetTy,
                                                 FixedArgs,
                                                 true /* IsVarArg */);

  FunctionCallee
    MGEPCallee = M.getOrInsertFunction(makeModelGEPName(RetTy, BaseAddressTy),
                                       ModelGEPType);

  auto *ModelGEPFunction = cast<Function>(MGEPCallee.getCallee());
  ModelGEPFunction->addFnAttr(llvm::Attribute::NoUnwind);
  ModelGEPFunction->addFnAttr(llvm::Attribute::WillReturn);
  ModelGEPFunction->addFnAttr(llvm::Attribute::ReadNone);
  FunctionTags::ModelGEP.addTo(ModelGEPFunction);
  FunctionTags::IsRef.addTo(ModelGEPFunction);

  return ModelGEPFunction;
}

llvm::Function *getSerializationMarker(llvm::Module &M, llvm::Type *T) {

  using namespace llvm;
  // Create a function, with T as return type, and 2 arguments.
  // The first argument has type T, the second argument is a boolean.
  // If the second argument is 'true', it means the marked instructions has
  // side effects that need to be taken in consideration for serialization.
  auto MarkerCallee = M.getOrInsertFunction(makeMarkerName(T),
                                            T,
                                            T,
                                            IntegerType::get(M.getContext(),
                                                             1));

  auto *MarkerF = cast<Function>(MarkerCallee.getCallee());
  MarkerF->addFnAttr(llvm::Attribute::NoUnwind);
  MarkerF->addFnAttr(llvm::Attribute::WillReturn);
  MarkerF->addFnAttr(llvm::Attribute::ReadNone);
  FunctionTags::SerializationMarker.addTo(MarkerF);
  FunctionTags::Marker.addTo(MarkerF);

  return MarkerF;
}
