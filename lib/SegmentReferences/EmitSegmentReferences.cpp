//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"

#include "revng/Model/Binary.h"
#include "revng/Model/FunctionTags.h"
#include "revng/SegmentReferences/EmitSegmentReferences.h"
#include "revng/Support/IRBuilder.h"

using namespace llvm;

/// Replace constants in the IR with references to segments.
///
/// This pass does not emit references to global variables, but emits a call to
/// `get_$SEGMENT_ADDRESS()` which can be later inlined.
/// This pass runs early in the pipeline.
class EmitSegmentReferences {
private:
  const model::Binary &Binary;
  MetaAddress CurrentAddress = MetaAddress::invalid();
  revng::IRBuilder B;
  IntegerType *IntPtrType = nullptr;
  llvm::DenseMap<const model::Segment *, Function *> SegmentGlobals;

public:
  EmitSegmentReferences(const model::Binary &Binary, Module &M) :
    Binary(Binary),
    B(M.getContext()),
    IntPtrType(IntegerType::get(M.getContext(), 64)) {

    // Create a global variable for each segment
    auto Pool = FunctionTags::SegmentGlobalGetter.getPool(M);
    for (const model::Segment &Segment : Binary.Segments()) {
      SegmentGlobals[&Segment] = Pool.get(M,
                                          { Segment.StartAddress(),
                                            Segment.VirtualSize() });
    }
  }

  void run(Function &F) {
    for (BasicBlock *BB : ReversePostOrderTraversal(&F)) {
      for (Instruction &I : *BB) {
        if (auto *NewPCCall = getCallTo(&I, "newpc")) {

          // Keep track of the last PC we saw. Since we proceed in RPOT, this
          // should be the address dominating all the instructions we're going
          // to visit.
          using namespace NewPCArguments;
          Value *Argument = NewPCCall->getArgOperand(InstructionID);
          CurrentAddress = BasicBlockID::fromValue(Argument).start();
          CurrentAddress = CurrentAddress.toGeneric();

        } else if (CurrentAddress.isValid()) {

          if (auto *Switch = dyn_cast<SwitchInst>(&I)) {
            // Special case switch to skip case labels
            if (Value *New = handleOperand(I, *Switch->getOperand(0)))
              Switch->setOperand(0, New);
          } else {
            for (Use &Operand : I.operands())
              if (Value *New = handleOperand(I, *Operand))
                Operand.set(New);
          }
        }
      }
    }
  }

private:
  Value *handleOperand(Instruction &I, Value &OperandValue) {
    Type *OperandType = OperandValue.getType();
    bool IsPointer = OperandType->isPointerTy();
    ConstantInt *Constant = getConstant(&OperandValue);
    if (Constant == nullptr)
      return nullptr;

    uint64_t Value = Constant->getLimitedValue();
    auto MaybeAddress = CurrentAddress.replaceAddress(Value);

    // Check if it's a valid address
    if (not MaybeAddress.isValid())
      return nullptr;

    // Check if the address is mapped
    auto [Segment, Offset] = Binary.getSegmentFor(MaybeAddress);
    if (Segment == nullptr)
      return nullptr;

    // OK, we need to replace the constant with get_$SEGMENT_ADDRESS() + offset
    // Note that we need a call here, since this enables to perform
    // optimizations that wouldn't otherwise take place if this was a
    // ConstantExpr. Specifically, in `sub(add(@segment, 1), add(@segment, 2))`,
    // @segment is not simplified.
    B.SetInsertPoint(&I);
    auto *IntType = IsPointer ? IntPtrType : cast<IntegerType>(OperandType);
    auto *SegmentAddress = B.CreateCall(SegmentGlobals[Segment]);
    auto *Add = B.CreateAdd(B.CreateZExtOrTrunc(SegmentAddress, IntType),
                            ConstantInt::get(IntType, Offset));

    if (IsPointer)
      return B.CreateIntToPtr(Add, OperandType);
    else
      return Add;
  }

  ConstantInt *getConstant(Value *Operand) {
    if (auto *CE = dyn_cast<ConstantExpr>(Operand))
      if (CE->getOpcode() == llvm::Instruction::IntToPtr)
        if (auto *CI = dyn_cast<ConstantInt>(CE->getOperand(0)))
          Operand = CI;
    return dyn_cast<ConstantInt>(Operand);
  };
};

namespace revng::pypeline::piperuns {

void EmitSegmentReferences::run() {
  Module &M = ModuleContainer.getModule();
  llvm::Function &F = *M.getFunction("root");
  ::EmitSegmentReferences ESR(Binary, M);
  ESR.run(F);
}

} // namespace revng::pypeline::piperuns
