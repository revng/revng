#include <experimental/coroutine>
#include <limits>

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Casting.h"

#include "revng/Support/Assert.h"

#include "BitLiveness.h"

namespace TypeShrinking {

using BitVector = llvm::BitVector;
using Instruction = llvm::Instruction;

bool hasSideEffect(const Instruction *Ins) {
  if (Ins->mayHaveSideEffects() || Ins->getOpcode() == Instruction::Call
      || Ins->getOpcode() == Instruction::CallBr
      || Ins->getOpcode() == Instruction::Ret
      || Ins->getOpcode() == Instruction::Store
      || Ins->getOpcode() == Instruction::Br
      || Ins->getOpcode() == Instruction::IndirectBr)
    return true;
  return false;
}

unsigned min(const unsigned &a, const unsigned &b) {
  return a < b ? a : b;
}

unsigned max(const unsigned &a, const unsigned &b) {
  return a > b ? a : b;
}

unsigned GetMaxOperandSize(Instruction *Ins) {
  unsigned Max = 0;
  for (unsigned i = 0; i < Ins->getNumOperands(); ++i) {
    auto *Operand = Ins->getOperand(i);
    if (Operand->getType()->isIntegerTy())
      Max = max(Max, Operand->getType()->getIntegerBitWidth());
    else
      return std::numeric_limits<unsigned>::max();
  }
  return Max;
}

/// A specialization of the transfer function for the and instruction
/// In cases where one of the operands is constant
unsigned transferMask(const unsigned &Element, const unsigned &MaskIndex) {
  return min(Element, MaskIndex);
}

unsigned transferAnd(Instruction *Ins, const unsigned &Element) {
  revng_assert(Ins->getOpcode() == Instruction::And);
  unsigned Result = Element;
  for (auto &Operand : Ins->operands()) {
    if (llvm::isa<llvm::ConstantInt>(Operand)) {
      auto OpVal = llvm::cast<llvm::ConstantInt>(Operand)->getUniqueInteger();
      auto MostSignificantBit = OpVal.getBitWidth() - OpVal.countLeadingZeros();
      Result = min(Result, transferMask(Element, MostSignificantBit));
    }
  }
  return Result;
}

unsigned transferShiftLeft(Instruction *Ins, const unsigned &Element) {
  unsigned OperandSize = GetMaxOperandSize(Ins);
  if (auto ConstOp = llvm::dyn_cast<llvm::ConstantInt>(Ins->getOperand(1))) {
    auto OpVal = ConstOp->getZExtValue();
    if (Element < OpVal)
      return 0;
    return Element - OpVal;
  }
  return OperandSize;
}

unsigned transferLogicalShiftRight(Instruction *Ins, const unsigned &Element) {
  unsigned OperandSize = GetMaxOperandSize(Ins);
  if (auto ConstOp = llvm::dyn_cast<llvm::ConstantInt>(Ins->getOperand(1))) {
    auto OpVal = ConstOp->getZExtValue();
    if (std::numeric_limits<unsigned>::max() - OpVal < Element)
      return std::numeric_limits<unsigned>::max();
    return min(OperandSize, Element + OpVal);
  }
  return OperandSize;
}

unsigned
transferArithmeticalShiftRight(Instruction *Ins, const unsigned &Element) {
  unsigned OperandSize = GetMaxOperandSize(Ins);
  if (auto ConstOp = llvm::dyn_cast<llvm::ConstantInt>(Ins->getOperand(1))) {
    auto OpVal = ConstOp->getZExtValue();
    if (std::numeric_limits<unsigned>::max() - OpVal < Element)
      return std::numeric_limits<unsigned>::max();
    return min(OperandSize, Element + OpVal);
  }
  return OperandSize;
}

unsigned transferTrunc(Instruction *Ins, const unsigned &Element) {
  return min(Element, Ins->getType()->getIntegerBitWidth());
}

unsigned transferZExt(Instruction *Ins, const unsigned &Element) {
  return min(Element, GetMaxOperandSize(Ins));
}

unsigned
BitLivenessAnalysis::applyTransferFunction(DataFlowNode *L, const unsigned E) {
  auto *Ins = L->Instruction;
  switch (Ins->getOpcode()) {
  case Instruction::And:
    return transferAnd(Ins, E);
  case Instruction::Xor:
  case Instruction::Or:
  case Instruction::Add:
  case Instruction::Sub:
  case Instruction::Mul:
    return min(E, GetMaxOperandSize(L->Instruction));
  case Instruction::Shl:
    return transferShiftLeft(Ins, E);
  case Instruction::LShr:
    return transferLogicalShiftRight(Ins, E);
  case Instruction::AShr:
    return transferArithmeticalShiftRight(Ins, E);
  case Instruction::Trunc:
    return transferTrunc(Ins, E);
  case Instruction::ZExt:
    return transferZExt(Ins, E);
  default:
    // by default all the bits of the operands can be alive
    return GetMaxOperandSize(L->Instruction);
  }
}

unsigned
BitLivenessAnalysis::combineValues(const unsigned &lh, const unsigned &rh) {
  return max(lh, rh);
}

bool BitLivenessAnalysis::isLessOrEqual(const unsigned &lh,
                                        const unsigned &rh) {
  return lh <= rh;
}

} // namespace TypeShrinking
