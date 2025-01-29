//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/Instruction.h"
#include "llvm/IR/Metadata.h"

#include "revng/ABI/ModelHelpers.h"
#include "revng/LocalVariables/LocalVariableHelpers.h"
#include "revng/Model/Type.h"
#include "revng/Support/IRHelpers.h"

using namespace llvm;

static void
setTypeMetadata(AllocaInst *A, const model::Type &T, const char *Name) {
  QuickMetadata QMD(A->getContext());
  A->setMetadata(Name, QMD.tuple(QMD.get(toLLVMString(T, *A->getModule()))));
}

static model::UpcastableType getTypeFromMetadata(AllocaInst *A,
                                                 const model::Binary &Model,
                                                 const char *Name) {
  revng_assert(hasNamedMetadata(A, Name));

  LLVMContext &C = A->getContext();
  MDNode *Node = A->getMetadata(C.getMDKindID(Name));
  using CAM = llvm::ConstantAsMetadata;
  Value *TypeString = cast<CAM>(Node->getOperand(0))->getValue();

  model::UpcastableType Type = fromLLVMString(TypeString, Model);
  revng_assert(not Type.isEmpty());
  return Type;
}

void setStackTypeMetadata(AllocaInst *A, const model::Type &StackType) {
  return setTypeMetadata(A, StackType, StackTypeMDName);
}

model::UpcastableType getStackTypeFromMetadata(AllocaInst *A,
                                               const model::Binary &Model) {
  return getTypeFromMetadata(A, Model, StackTypeMDName);
}

void setVariableTypeMetadata(AllocaInst *A, const model::Type &VariableType) {
  return setTypeMetadata(A, VariableType, VariableTypeMDName);
}

model::UpcastableType getVariableTypeFromMetadata(AllocaInst *A,
                                                  const model::Binary &Model) {
  return getTypeFromMetadata(A, Model, VariableTypeMDName);
}
