//
// Copyright (c) rev.ng Labs Srl. See LICENSE.md for details.
//

#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"

#include "revng/Support/Assert.h"
#include "revng/Support/IRHelpers.h"
#include "revng/Support/OpaqueFunctionsPool.h"

#include "revng-c/Support/FunctionTags.h"
#include "revng-c/Support/Mangling.h"

static constexpr const char *const ModelGEPName = "ModelGEP";
static constexpr const char *const MarkerName = "AssignmentMarker";

namespace FunctionTags {
Tag AllocatesLocalVariable("AllocatesLocalVariable");
Tag MallocLike("MallocLike");
Tag IsRef("IsRef");
Tag AddressOf("AddressOf");
Tag ModelCast("ModelCast");
Tag ModelGEP(ModelGEPName);
Tag AssignmentMarker(MarkerName);
Tag OpaqueExtractValue("OpaqueExtractvalue");
Tag Parentheses("Parentheses");
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

llvm::FunctionType *
getAddressOfFunctionType(llvm::LLVMContext &C, llvm::Type *T) {
  // There are 2 fixed arguments:
  // - the first is a pointer to a constant string that contains a serialization
  //   of the key of the base type;
  // - the second is T, i.e. the type of the base pointer.
  llvm::SmallVector<llvm::Type *, 2> FixedArgs = { getStringPtrType(C), T };
  return llvm::FunctionType::get(T, FixedArgs, false /* IsVarArg */);
}

void initAddressOfPool(OpaqueFunctionsPool<llvm::Type *> &Pool) {
  // Set attributes
  Pool.addFnAttribute(llvm::Attribute::NoUnwind);
  Pool.addFnAttribute(llvm::Attribute::WillReturn);
  Pool.addFnAttribute(llvm::Attribute::ReadOnly);
  Pool.addFnAttribute(llvm::Attribute::InaccessibleMemOnly);
  // Set revng tags
  Pool.setTags({ &FunctionTags::AddressOf });
  // Initialize the pool from its internal llvm::Module if possible.
  Pool.initializeFromReturnType(FunctionTags::AddressOf);
}

void initModelCastPool(OpaqueFunctionsPool<llvm::Type *> &Pool) {
  // Set attributes
  Pool.addFnAttribute(llvm::Attribute::NoUnwind);
  Pool.addFnAttribute(llvm::Attribute::WillReturn);
  Pool.addFnAttribute(llvm::Attribute::ReadNone);
  // Set revng tags
  Pool.setTags({ &FunctionTags::ModelCast });
  // Initialize the pool from its internal llvm::Module if possible.
  Pool.initializeFromReturnType(FunctionTags::ModelCast);
}

void initParenthesesPool(OpaqueFunctionsPool<llvm::Type *> &Pool) {
  // Set attributes
  Pool.addFnAttribute(llvm::Attribute::NoUnwind);
  Pool.addFnAttribute(llvm::Attribute::WillReturn);
  Pool.addFnAttribute(llvm::Attribute::ReadNone);
  // Set revng tags
  Pool.setTags({ &FunctionTags::Parentheses });
  // Initialize the pool from its internal llvm::Module if possible.
  Pool.initializeFromReturnType(FunctionTags::Parentheses);
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
  ModelGEPFunction->addFnAttr(llvm::Attribute::ReadOnly);
  ModelGEPFunction->addFnAttr(llvm::Attribute::InaccessibleMemOnly);
  FunctionTags::ModelGEP.addTo(ModelGEPFunction);
  FunctionTags::IsRef.addTo(ModelGEPFunction);

  return ModelGEPFunction;
}

llvm::Function *getAssignmentMarker(llvm::Module &M, llvm::Type *T) {

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
  FunctionTags::AssignmentMarker.addTo(MarkerF);
  FunctionTags::Marker.addTo(MarkerF);

  return MarkerF;
}

llvm::FunctionType *
getOpaqueEVFunctionType(llvm::LLVMContext &C, llvm::ExtractValueInst *Extract) {
  using namespace llvm;
  // First argument is the struct we are extracting from
  std::vector ArgTypes = { Extract->getAggregateOperand()->getType() };
  // All other arguments are indices, which we decided to be of type i64
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
