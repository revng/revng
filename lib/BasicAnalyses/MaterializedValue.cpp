/// \file MaterializedValue.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"

#include "revng/BasicAnalyses/MaterializedValue.h"
#include "revng/Support/IRHelpers.h"

MaterializedValue
MaterializedValue::apply(llvm::Instruction *Operation,
                         llvm::ArrayRef<MaterializedValue> Operands) {
  using namespace llvm;

  if (isa<AllocaInst>(Operation))
    return MaterializedValue::invalid();

  if (isa<IntToPtrInst>(Operation)) {
    revng_assert(Operands.size() == 1);
    return Operands[0];
  }

  MaterializedValue Result;
  Result.IsValid = true;

  //
  // Merge AffectedBy sets
  //
  for (const MaterializedValue &Value : Operands)
    for (const MemoryRange &Range : Value.AffectedBy)
      Result.AffectedBy.push_back(Range);

  //
  // Detect symbol name
  //
  if (Operands.size() > 0 and Operands[0].hasSymbol())
    Result.SymbolName = Operands[0].symbolName();

  // Handle add as second operand in add
  bool IsAdd = false;
  auto *I = dyn_cast<Instruction>(Operation);
  if (I != nullptr) {
    auto Opcode = I->getOpcode();
    IsAdd = Opcode == Instruction::Add;
  }

  if (not Result.SymbolName.has_value() and IsAdd and Operands[1].hasSymbol())
    Result.SymbolName = Operands[1].symbolName();

  // Ensure no more than one symbol
  auto HasSymbol = [](const MaterializedValue &Value) -> bool {
    return Value.hasSymbol();
  };
  if (not IsAdd and llvm::any_of(skip_front(Operands), HasSymbol))
    return MaterializedValue::invalid();

  // Restrict the operations we can perform on symbols to cast, add,
  // subtraction, masking and bswap
  if (Result.SymbolName.has_value() and I != nullptr) {
    auto *Call = dyn_cast<CallInst>(I);
    // TODO: this is not very nice
    if (not(I->isCast() or I->getOpcode() == Instruction::Add
            or I->getOpcode() == Instruction::Sub
            or I->getOpcode() == Instruction::And
            or I->getOpcode() == Instruction::ZExt
            or I->getOpcode() == Instruction::IntToPtr
            or I->getOpcode() == Instruction::PtrToInt
            or I->getOpcode() == Instruction::Trunc
            or I->getOpcode() == Instruction::BitCast
            or I->getOpcode() == Instruction::Shl
            or (Call != nullptr
                and Call->getIntrinsicID() == Intrinsic::bswap))) {
      return MaterializedValue::invalid();
    }
  }

  //
  // Build operands list
  //
  SmallVector<Constant *, 4> ConstantOperands;
  unsigned Index = 0;
  for (const MaterializedValue &Operand : Operands) {
    auto *OperandType = Operation->getOperand(Index)->getType();
    ConstantOperands.push_back(Operand.toConstant(OperandType));
    ++Index;
  }

  //
  // Perform computation
  //
  const auto &DL = getModule(Operation)->getDataLayout();
  auto *Folded = ConstantFoldInstOperands(Operation, ConstantOperands, DL);
  revng_assert(Folded != nullptr);
  Result.Value = cast<ConstantInt>(Folded)->getValue();

  return Result;
}

MaterializedValue MaterializedValue::fromMutable(llvm::ConstantInt *CI) {
  return MaterializedValue::fromMutable(CI->getValue());
}

MaterializedValue MaterializedValue::fromConstant(llvm::ConstantInt *CI) {
  return MaterializedValue::fromConstant(CI->getValue());
}

llvm::Constant *
MaterializedValue::toConstant(llvm::LLVMContext &Context) const {
  return toConstant(llvm::IntegerType::get(Context, Value.getBitWidth()));
}

llvm::Constant *
MaterializedValue::toConstant(llvm::Type *DestinationType) const {
  using namespace llvm;
  if (DestinationType->isPointerTy()) {
    auto *IntegerValue = toConstant(DestinationType->getContext());
    return ConstantExpr::getIntToPtr(IntegerValue, DestinationType);
  } else if (auto *IT = dyn_cast<IntegerType>(DestinationType)) {
    revng_assert(IT->getBitWidth() == Value.getBitWidth());
    return ConstantInt::get(DestinationType, Value);
  } else {
    revng_abort();
  }
}
