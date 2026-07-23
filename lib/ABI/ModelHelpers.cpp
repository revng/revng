//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Casting.h"

#include "revng/ABI/ModelHelpers.h"
#include "revng/Model/Architecture.h"
#include "revng/Model/Binary.h"
#include "revng/Model/FunctionTags.h"
#include "revng/Model/IRHelpers.h"
#include "revng/Support/Assert.h"
#include "revng/Support/IRHelpers.h"

using llvm::dyn_cast;

model::UpcastableType modelType(const llvm::Value *V,
                                const model::Binary &Model) {
  model::UpcastableType Result;

  using namespace llvm;

  llvm::Type *T = V->getType();

  // Handle pointers
  bool AddPointer = false;
  if (isa<llvm::PointerType>(T)) {
    revng_assert(isa<llvm::AllocaInst>(V) or isa<llvm::GlobalVariable>(V));
    AddPointer = true;
    T = getVariableType(V);
    revng_assert(isa<llvm::IntegerType>(T) or isa<llvm::ArrayType>(T));
  } else {
    revng_assert(isa<llvm::IntegerType>(T));
  }

  // Actually build the core type
  if (isa<llvm::IntegerType>(T)) {
    Result = llvmIntToModelType(T, Model);
  } else if (auto *Array = llvm::dyn_cast<llvm::ArrayType>(T)) {
    revng_check(AddPointer);
    Result = llvmIntToModelType(Array->getElementType(), Model);
  }

  revng_assert(Result->verify());

  // If it is a pointer, make sure to mark is as such
  if (AddPointer)
    return model::PointerType::make(std::move(Result), Model.Architecture());
  else
    return Result;
}

model::UpcastableType llvmIntToModelType(const llvm::Type *TypeToConvert,
                                         const model::Binary &Model) {
  model::UpcastableType Result = model::UpcastableType::empty();
  if (isa<llvm::PointerType>(TypeToConvert)) {
    // If it's a pointer, return intptr_t for the current architecture
    //
    // Note: this is suboptimal, in order to avoid this, please use modelType
    // passing the Value instead of invoking llvmIntToModelType passing in just
    // the type

    auto PtrSize = model::Architecture::getPointerSize(Model.Architecture());
    Result = model::PrimitiveType::makeGeneric(PtrSize);
  }

  if (auto *Int = dyn_cast<llvm::IntegerType>(TypeToConvert)) {
    // Convert the integer type
    if (Int->getIntegerBitWidth() == 1) {
      Result = model::PrimitiveType::makeGeneric(1);
    } else {
      revng_assert(Int->getIntegerBitWidth() % 8 == 0);
      Result = model::PrimitiveType::makeGeneric(Int->getIntegerBitWidth() / 8);
    }
  }

  if (Result.isEmpty()) {
    revng_abort("Only integer and pointer types can be directly converted from "
                "LLVM types to C types.");
  }

  revng_assert(Result->verify(true),
               ("Unsupported llvm type: " + toString(Result)).c_str());
  return Result;
}

model::UpcastableType fromLLVMString(llvm::Value *V,
                                     const model::Binary &Model) {
  // Try to get a string out of the llvm::Value
  llvm::StringRef BaseTypeString = extractFromConstantStringPtr(V);
  auto ParsedType = fromString<model::UpcastableType>(BaseTypeString);
  if (not ParsedType) {
    std::string Error = "Could not deserialize the model type from LLVM "
                        "constant string \""
                        + BaseTypeString.str()
                        + "\": " + consumeToString(ParsedType) + ".";
    revng_abort(Error.c_str());
  }

  revng_assert(!ParsedType->isEmpty(),
               "Type in a LLVM constant string was set to "
               "`model::UpcastableType::empty()`. How did it slip through?");

  if (model::DefinedType *Defined = (*ParsedType)->skipToDefinedType()) {
    model::TypeDefinitionReference &Reference = Defined->Definition();

    revng_assert(Reference.isValid() == false);
    Reference.setRoot(&Model);
    revng_assert(Reference.isValid() == true);
    revng_assert(Reference.getConst() != nullptr);
  } else {
    // Primitives have no references, so no need to do anything special.
  }
  revng_assert((*ParsedType)->verify(true));

  return *ParsedType;
}

llvm::Constant *toLLVMString(const model::UpcastableType &Type,
                             llvm::Module &M) {
  return getUniqueString(&M, toString(Type));
}
