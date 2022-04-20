//
// Copyright (c) rev.ng Labs Srl. See LICENSE.md for details.
//

#include "revng/Model/LoadModelPass.h"
#include "revng/Model/Type.h"

#include "revng-c/Decompiler/AddPrimitiveTypesPass.h"

char AddPrimitiveTypesPass::ID = 0;

using Register = llvm::RegisterPass<AddPrimitiveTypesPass>;
static ::Register
  X("add-primitives", "Pass to add primitive types to Model", false, false);

void AddPrimitiveTypesPass::getAnalysisUsage(llvm::AnalysisUsage &AU) const {
  AU.addRequired<LoadModelWrapperPass>();
  AU.setPreservesAll();
}

using namespace model::PrimitiveTypeKind;

bool AddPrimitiveTypesPass::runOnModule(llvm::Module &M) {
  auto &ModelWrapper = getAnalysis<LoadModelWrapperPass>().get();
  auto &WritableModel = ModelWrapper.getWriteableModel();

  // For each of these types, we want to have in the model a corresponding
  // PrimitiveType for each possible dimension
  static constexpr const Values PrimitiveTypes[] = {
    Generic, PointerOrNumber, Number, Unsigned, Signed
  };
  static constexpr const uint8_t Sizes[] = { 1, 2, 4, 8, 16 };

  // getPrimitiveType() creates the type if it does not exist
  for (auto &Type : PrimitiveTypes)
    for (auto &Size : Sizes)
      WritableModel->getPrimitiveType(Type, Size);

  return true;
}
