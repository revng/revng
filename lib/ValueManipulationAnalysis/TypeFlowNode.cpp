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

#include "revng-c/Support/ModelHelpers.h"
#include "revng-c/ValueManipulationAnalysis/TypeColors.h"

#include "TypeFlowNode.h"

using namespace vma;
using namespace llvm;

RecursiveCoroutine<ColorSet> vma::QTToColor(const model::QualifiedType &QT) {

  if (QT.isPointer())
    rc_return POINTERNESS;

  if (QT.is(model::TypeKind::TypedefType)) {
    auto *UnqualT = QT.UnqualifiedType.getConst();
    auto *TypedefT = llvm::cast<model::TypedefType>(UnqualT);

    rc_return rc_recur QTToColor(TypedefT->UnderlyingType);
  }

  if (QT.is(model::TypeKind::PrimitiveType)) {
    auto *UnqualT = QT.UnqualifiedType.getConst();
    auto *PrimitiveT = llvm::cast<model::PrimitiveType>(UnqualT);

    switch (PrimitiveT->PrimitiveKind) {

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

ColorSet
vma::getAcceptedColors(const UseOrValue &Content, const model::Binary *Model) {
  // Arguments, constants, globals etc.
  if (isValue(Content)) {
    const Value *V = getValue(Content);

    if (isa<Argument>(V))
      return ALL_COLORS;

    // Constants and globals should not be infected, since they don't belong to
    // a single function
    if (isa<Constant>(V) or isa<GlobalValue>(V))
      return NO_COLOR;

    if (not isa<Instruction>(V))
      return NO_COLOR;
  }

  // Instructions and operand uses should be the only thing remaining
  bool IsContentInst = isInst(Content);
  revng_assert(IsContentInst or isUse(Content));

  // If the content of the node is an Instruction's Value, assign colors
  // based on the instruction's opcode. Otherwise, if we are creating a node for
  // one of the operands, find which the user of the operand and check its
  // opcode.
  const Instruction *I = IsContentInst ?
                           cast<Instruction>(getValue(Content)) :
                           cast<Instruction>(getUse(Content)->getUser());

  // Do we have strong model information about this node? If so, use that
  if (Model) {
    // Deduce type for the use or for the value, depending on which type of node
    // we are looking at
    auto DeducedTypes = IsContentInst ?
                          getStrongModelInfo(I, *Model) :
                          getExpectedModelType(getUse(Content), *Model);

    if (DeducedTypes.size() == 1)
      return QTToColor(DeducedTypes.back());

    if (DeducedTypes.size() > 1) {
      // There are cases in which we can associate to an LLVM value (typically
      // an aggregate) more than one model type, e.g. for values returned by
      // RawFunctionTypes or for calls to StructInitializer.
      // In these cases, the aggregate itself has no color. Not that the value
      // extracted from that will instead have a color, which is inferred by
      // `getStrongModelInfo()`.
    }
  }

  // Fallback to manually matching LLVM instructions that provides us with rich
  // type information
  switch (I->getOpcode()) {
  case Instruction::FNeg:
  case Instruction::FAdd:
  case Instruction::FMul:
  case Instruction::FSub:
  case Instruction::FDiv:
  case Instruction::FRem:
  case Instruction::FPExt:
    return FLOATNESS;
    break;

  case Instruction::FCmp:
    if (IsContentInst)
      return BOOLNESS;
    else
      return FLOATNESS;
    break;

  case Instruction::ICmp:
    if (IsContentInst)
      return BOOLNESS;
    if (cast<ICmpInst>(I)->isSigned())
      return SIGNEDNESS;
    if (cast<ICmpInst>(I)->isUnsigned())
      return UNSIGNEDNESS | POINTERNESS;
    break;

  case Instruction::SDiv:
  case Instruction::SRem:
    return SIGNEDNESS | NUMBERNESS;
    break;

  case Instruction::UDiv:
  case Instruction::URem:
    return UNSIGNEDNESS | NUMBERNESS;
    break;

  case Instruction::Alloca:
    if (IsContentInst)
      return POINTERNESS;
    if (getUse(Content)->get() == cast<AllocaInst>(I)->getArraySize())
      return UNSIGNEDNESS;
    break;

  case Instruction::Load:
    if (isUse(Content)
        && getOpNo(Content) == cast<LoadInst>(I)->getPointerOperandIndex())
      return POINTERNESS;
    break;

  case Instruction::Store:
    if (isUse(Content)
        && getOpNo(Content) == cast<StoreInst>(I)->getPointerOperandIndex())
      return POINTERNESS;
    break;

  case Instruction::AShr:
    if (IsContentInst or getOpNo(Content) == 0)
      return SIGNEDNESS;
    if (getOpNo(Content) == 1)
      return ~(FLOATNESS | POINTERNESS);
    break;

  case Instruction::LShr:
    if (IsContentInst or getOpNo(Content) == 0)
      // TODO: rule on first operand too strict?
      return UNSIGNEDNESS;
    if (getOpNo(Content) == 1)
      return ~(FLOATNESS | POINTERNESS);
    break;

  case Instruction::Shl:
    if (IsContentInst)
      return ~(FLOATNESS | POINTERNESS);
    if (getOpNo(Content) == 0)
      // TODO: rule on first operand too strict?
      return (SIGNEDNESS | UNSIGNEDNESS | BOOLNESS);
    if (getOpNo(Content) == 1)
      return ~(FLOATNESS | POINTERNESS);
    break;

  case Instruction::Mul:
    return ~(FLOATNESS | POINTERNESS);
    break;

  case Instruction::Br:
    if (isUse(Content) && cast<BranchInst>(I)->isConditional()
        && getUse(Content)->get() == cast<BranchInst>(I)->getCondition())
      return BOOLNESS;
    break;

  case Instruction::Select:
    if (isUse(Content)
        && getUse(Content)->get() == cast<SelectInst>(I)->getCondition())
      return BOOLNESS;
    break;

  case Instruction::Trunc:
  case Instruction::And:
  case Instruction::Or:
  case Instruction::Xor:
    // TODO: Restrict more what can be accepted by bitwise operations?
    return ~NUMBERNESS;
    break;

  case Instruction::GetElementPtr:
    revng_abort("Didn't expect to find a GEP here");
    break;
  }

  return ~NUMBERNESS;
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
