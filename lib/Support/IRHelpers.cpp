/// \file IRHelpers.cpp
/// \brief Implementation of IR helper functions

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <fstream>

// LLVM includes
#include "llvm/Support/raw_os_ostream.h"

// Local libraries includes
#include "revng/Support/IRHelpers.h"

using namespace llvm;

void dumpModule(const Module *M, const char *Path) {
  std::ofstream FileStream(Path);
  raw_os_ostream Stream(FileStream);
  M->print(Stream, nullptr, true);
}

GlobalVariable *buildString(Module *M, StringRef String, const Twine &Name) {
  LLVMContext &C = M->getContext();
  auto *Initializer = ConstantDataArray::getString(C, String, true);
  return new GlobalVariable(*M,
                            Initializer->getType(),
                            true,
                            GlobalVariable::InternalLinkage,
                            Initializer,
                            Name);
}

Constant *buildStringPtr(Module *M, StringRef String, const Twine &Name) {
  LLVMContext &C = M->getContext();
  Type *Int8PtrTy = Type::getInt8Ty(C)->getPointerTo();
  GlobalVariable *NewVariable = buildString(M, String, Name);
  return ConstantExpr::getBitCast(NewVariable, Int8PtrTy);
}

Constant *getUniqueString(Module *M,
                          StringRef Namespace,
                          StringRef String,
                          const Twine &Name) {
  LLVMContext &C = M->getContext();
  Type *Int8PtrTy = Type::getInt8Ty(C)->getPointerTo();
  NamedMDNode *StringsList = M->getOrInsertNamedMetadata(Namespace);

  for (MDNode *Operand : StringsList->operands()) {
    auto *T = cast<MDTuple>(Operand);
    revng_assert(T->getNumOperands() == 1);
    auto *CAM = cast<ConstantAsMetadata>(T->getOperand(0).get());
    auto *GV = cast<GlobalVariable>(CAM->getValue());
    revng_assert(GV->isConstant() and GV->hasInitializer());

    const Constant *Initializer = GV->getInitializer();
    StringRef Content = cast<ConstantDataArray>(Initializer)->getAsString();

    // Ignore the terminator
    if (Content.drop_back() == String)
      return ConstantExpr::getBitCast(GV, Int8PtrTy);
  }

  GlobalVariable *NewVariable = buildString(M, String, Name);
  auto *CAM = ConstantAsMetadata::get(NewVariable);
  StringsList->addOperand(MDTuple::get(C, { CAM }));
  return ConstantExpr::getBitCast(NewVariable, Int8PtrTy);
}
