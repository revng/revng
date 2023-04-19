//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

#include <cstddef>
#include <variant>

#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/Casting.h"

#include "revng/Model/Binary.h"
#include "revng/Model/PrimitiveTypeKind.h"
#include "revng/Model/TypeKind.h"
#include "revng/Model/TypedefType.h"
#include "revng/Support/Assert.h"
#include "revng/Support/IRHelpers.h"

#include "revng-c/ValueManipulationAnalysis/TypeColors.h"

#include "TypeFlowNode.h"

using namespace vma;
using namespace llvm;

RecursiveCoroutine<ColorSet> vma::QTToColor(const model::QualifiedType &QT) {

  if (QT.isPointer())
    rc_return POINTERNESS;

  if (QT.is(model::TypeKind::TypedefType)) {

    if (not QT.isScalar())
      rc_return NO_COLOR;

    auto *UnqualT = QT.UnqualifiedType().getConst();
    auto *TypedefT = llvm::cast<model::TypedefType>(UnqualT);

    rc_return rc_recur QTToColor(TypedefT->UnderlyingType());
  }

  if (QT.is(model::TypeKind::PrimitiveType)) {
    auto *UnqualT = QT.UnqualifiedType().getConst();
    auto *PrimitiveT = llvm::cast<model::PrimitiveType>(UnqualT);

    switch (PrimitiveT->PrimitiveKind()) {

    case model::PrimitiveTypeKind::Void:
      rc_return NO_COLOR;
      break;

    case model::PrimitiveTypeKind::Unsigned:
      rc_return UNSIGNEDNESS;
      break;

    case model::PrimitiveTypeKind::Signed:
      rc_return SIGNEDNESS;
      break;

    case model::PrimitiveTypeKind::Float:
      rc_return FLOATNESS;
      break;

    case model::PrimitiveTypeKind::Number:
      rc_return SIGNEDNESS | UNSIGNEDNESS;
      break;

    case model::PrimitiveTypeKind::PointerOrNumber:
      rc_return POINTERNESS | SIGNEDNESS | UNSIGNEDNESS;
      break;

    case model::PrimitiveTypeKind::Generic:
      rc_return ~NUMBERNESS;
      break;

    default:
      revng_abort("Unknown Primitive Type");
    }
  }

  rc_return NO_COLOR;
}

void TypeFlowNodeData::print(llvm::raw_ostream &Out) const {
  if (this->isValue()) {
    Out << "value ";
    this->getValue()->print(Out);
  } else {
    Value *Val = this->getUse()->get();
    Value *User = this->getUse()->getUser();

    Out << "use of ";
    Val->printAsOperand(Out);
    Out << " in ";
    User->print(Out);
  }
}

std::string TypeFlowNodeData::toString() const {
  return dumpToString(this);
}
