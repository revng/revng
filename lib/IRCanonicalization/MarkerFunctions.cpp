//
// Copyright rev.ng Srls. See LICENSE.md for details.
//

#include <string>

#include "llvm/IR/Function.h"
#include "llvm/IR/Type.h"

#include "revng/Support/Assert.h"

#include "revng-c/Support/Mangling.h"

static constexpr const char *const MarkerPrefix = "revng_serialization_marker_";
static constexpr const char *const ModelGEPPrefix = "revng_model_gep_";

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

std::string makeMarkerName(const llvm::Type *Ty) {
  return MarkerPrefix + makeTypeName(Ty);
}

std::string makeModelGEPName(const llvm::Type *Ty) {
  return ModelGEPPrefix + makeTypeName(Ty);
}

bool isMarker(const llvm::Function &F) {
  return F.hasName() and F.getName().startswith(MarkerPrefix);
}

bool isModelGEP(const llvm::Function &F) {
  return F.hasName() and F.getName().startswith(ModelGEPPrefix);
}
