//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

#include <algorithm>
#include <cstddef>
#include <variant>

#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"

#include "revng/Support/Assert.h"
#include "revng/Support/IRHelpers.h"

#include "revng-c/ValueManipulationAnalysis/TypeColors.h"

#include "TypeFlowNode.h"

using namespace vma;
using namespace llvm;

NodeColorProperty vma::nodeColors(const UseOrValue &Content) {
  // Arguments, constants, globals etc.
  if (isValue(Content)) {
    const Value *V = getValue(Content);

    if (isa<Argument>(V))
      return { /*initial=*/NO_COLOR, /*accepted=*/ALL_COLORS };

    // Constants and globals should not be infected, since they don't belong to
    // a single function
    if (isa<Constant>(V) or isa<GlobalValue>(V))
      return { /*initial=*/NO_COLOR, /*accepted=*/NO_COLOR };

    if (not isa<Instruction>(V))
      return { /*initial=*/NO_COLOR, /*accepted=*/NO_COLOR };
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

  switch (I->getOpcode()) {
  case Instruction::FNeg:
  case Instruction::FAdd:
  case Instruction::FMul:
  case Instruction::FSub:
  case Instruction::FDiv:
  case Instruction::FRem:
  case Instruction::FPExt:
    return { /*initial=*/FLOATNESS, /*accepted=*/FLOATNESS };
    break;

  case Instruction::FCmp:
    if (IsContentInst)
      return { /*initial=*/BOOLNESS, /*accepted=*/BOOLNESS };
    else
      return { /*initial=*/FLOATNESS, /*accepted=*/FLOATNESS };
    break;

  case Instruction::ICmp:
    if (IsContentInst)
      return { /*initial=*/BOOLNESS, /*accepted=*/BOOLNESS };
    if (cast<ICmpInst>(I)->isSigned())
      return { /*initial=*/SIGNEDNESS, /*accepted=*/SIGNEDNESS };
    if (cast<ICmpInst>(I)->isUnsigned())
      return { /*initial=*/UNSIGNEDNESS,
               /*accepted=*/UNSIGNEDNESS | POINTERNESS };
    break;

  case Instruction::SDiv:
  case Instruction::SRem:
    return { /*initial=*/(SIGNEDNESS | NUMBERNESS),
             /*accepted=*/(SIGNEDNESS | NUMBERNESS) };
    break;

  case Instruction::UDiv:
  case Instruction::URem:
    return { /*initial=*/(UNSIGNEDNESS | NUMBERNESS),
             /*accepted=*/(UNSIGNEDNESS | NUMBERNESS) };
    break;

  case Instruction::Alloca:
    if (IsContentInst)
      return { /*initial=*/POINTERNESS,
               /*accepted=*/POINTERNESS };
    if (getUse(Content)->get() == cast<AllocaInst>(I)->getArraySize())
      return { /*initial=*/UNSIGNEDNESS,
               /*accepted=*/UNSIGNEDNESS };
    break;

  case Instruction::Load:
    if (isUse(Content)
        && getOpNo(Content) == cast<LoadInst>(I)->getPointerOperandIndex())
      return { /*initial=*/POINTERNESS,
               /*accepted=*/POINTERNESS };
    break;

  case Instruction::Store:
    if (isUse(Content)
        && getOpNo(Content) == cast<StoreInst>(I)->getPointerOperandIndex())
      return { /*initial=*/POINTERNESS,
               /*accepted=*/POINTERNESS };
    break;

  case Instruction::AShr:
    if (IsContentInst or getOpNo(Content) == 0)
      return { /*initial=*/SIGNEDNESS,
               /*accepted=*/SIGNEDNESS };
    if (getOpNo(Content) == 1)
      return { /*initial=*/UNSIGNEDNESS,
               /*accepted=*/~(FLOATNESS | POINTERNESS) };
    break;

  case Instruction::LShr:
    if (IsContentInst or getOpNo(Content) == 0)
      // TODO: rule on first operand too strict?
      return { /*initial=*/UNSIGNEDNESS,
               /*accepted=*/UNSIGNEDNESS };
    if (getOpNo(Content) == 1)
      return { /*initial=*/UNSIGNEDNESS,
               /*accepted=*/~(FLOATNESS | POINTERNESS) };
    break;

  case Instruction::Shl:
    if (IsContentInst)
      return { /*initial=*/NO_COLOR,
               /*accepted=*/~(FLOATNESS | POINTERNESS) };
    if (getOpNo(Content) == 0)
      // TODO: rule on first operand too strict?
      return { /*initial = */ NO_COLOR,
               /*accepted=*/
               (SIGNEDNESS | UNSIGNEDNESS | BOOLNESS) };
    if (getOpNo(Content) == 1)
      return { /*initial=*/UNSIGNEDNESS,
               /*accepted=*/~(FLOATNESS | POINTERNESS) };
    break;

  case Instruction::Mul:
    return { /*initial=*/NUMBERNESS,
             /*accepted=*/~(FLOATNESS | POINTERNESS) };
    break;

  case Instruction::Br:
    if (isUse(Content) && cast<BranchInst>(I)->isConditional()
        && getUse(Content)->get() == cast<BranchInst>(I)->getCondition())
      return { /*initial=*/BOOLNESS, /*accepted=*/BOOLNESS };
    break;

  case Instruction::Select:
    if (isUse(Content)
        && getUse(Content)->get() == cast<SelectInst>(I)->getCondition())
      return { /*initial=*/BOOLNESS, /*accepted=*/BOOLNESS };
    break;

  case Instruction::Trunc:
  case Instruction::And:
  case Instruction::Or:
  case Instruction::Xor:
    // TODO: Restrict more what can be accepted by bitwise operations?
    return { /*initial=*/NO_COLOR, /*accepted=*/~NUMBERNESS };
    break;

  case Instruction::GetElementPtr:
    revng_abort("Didn't expect to find a GEP here");
    break;
  }

  return { /*initial=*/NO_COLOR, /*accepted=*/~NUMBERNESS };
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
