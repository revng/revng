#include <experimental/coroutine>
#include <limits>

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Type.h"

#include "revng/Support/Assert.h"

#include "revng-c/TypeShrinking/BitLiveness.h"

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
unsigned min(unsigned a, unsigned b) {
  return a < b ? a : b;
}
unsigned max(unsigned a, unsigned b) {
  return a > b ? a : b;
}
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

unsigned
BitLivenessAnalysis::applyTransferFunction(DataFlowNode *L, const unsigned E) {
  auto *Ins = L->Instruction;
  switch (Ins->getOpcode()) {
  case Instruction::And:
    return transferAnd(Ins, E);
  case Instruction::Add:
  case Instruction::Sub:
  case Instruction::Mul:
    return E;
  default:
    return std::numeric_limits<unsigned>::max();
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
