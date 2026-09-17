//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <optional>

#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"

#include "revng/Model/Binary.h"
#include "revng/Model/FunctionTags.h"
#include "revng/SegmentReferences/EmitSegmentReferences.h"
#include "revng/Support/IRBuilder.h"
#include "revng/Support/NewPC.h"

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

  /// An operand to replace, and the segment and offset to replace it with.
  struct Replacement {
    llvm::Use *Operand;
    const model::Segment *Segment;
    uint64_t Offset;
  };

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
    for (const Replacement &TheReplacement : collect(F))
      apply(TheReplacement);
  }

private:
  llvm::SmallVector<Replacement, 16> collect(Function &F) {
    llvm::SmallVector<Replacement, 16> Replacements;

    for (BasicBlock *BB : ReversePostOrderTraversal(&F)) {
      for (Instruction &I : *BB) {
        if (std::optional NewPCCall = NewPCHelper.getCall(&I)) {

          // Keep track of the last PC we saw. Since we proceed in RPOT, this
          // should be the address dominating all the instructions we're going
          // to visit.
          CurrentAddress = blockIDFromNewPC(*NewPCCall).start();
          CurrentAddress = CurrentAddress.toGeneric();

        } else if (CurrentAddress.isValid()) {

          if (auto *Switch = dyn_cast<SwitchInst>(&I)) {
            // Special case switch to skip case labels
            if (auto R = collectOperand(Switch->getOperandUse(0)))
              Replacements.push_back(*R);
          } else {
            for (Use &Operand : I.operands())
              if (auto R = collectOperand(Operand))
                Replacements.push_back(*R);
          }
        }
      }
    }

    return Replacements;
  }

  std::optional<Replacement> collectOperand(Use &Operand) {
    ConstantInt *Constant = getConstant(Operand.get());
    if (Constant == nullptr)
      return std::nullopt;

    uint64_t Address = Constant->getLimitedValue();
    auto MaybeAddress = CurrentAddress.replaceAddress(Address);

    // Check if it's a valid address
    if (not MaybeAddress.isValid())
      return std::nullopt;

    // Check if the address is mapped
    auto [Segment, Offset] = Binary.getSegmentFor(MaybeAddress);
    if (Segment == nullptr)
      return std::nullopt;

    return Replacement{ &Operand, Segment, Offset };
  }

  void apply(const Replacement &TheReplacement) {
    Use &Operand = *TheReplacement.Operand;
    auto &I = *cast<Instruction>(Operand.getUser());
    Type *OperandType = Operand->getType();
    bool IsPointer = OperandType->isPointerTy();

    // OK, we need to replace the constant with get_$SEGMENT_ADDRESS() + offset
    // Note that we need a call here, since this enables to perform
    // optimizations that wouldn't otherwise take place if this was a
    // ConstantExpr. Specifically, in `sub(add(@segment, 1), add(@segment, 2))`,
    // @segment is not simplified.

    // The replacement has to dominate the use. For a `PHINode` the use is on
    // the edge coming from the incoming block, not at the `PHINode` itself, so
    // emit it at the end of that block. Emitting it before the `PHINode` would
    // leave a non-`PHINode` at the top of the block, and the incoming value
    // would not dominate the end of the incoming block.
    if (auto *Phi = dyn_cast<PHINode>(&I))
      B.SetInsertPoint(Phi->getIncomingBlock(Operand)->getTerminator());
    else
      B.SetInsertPoint(&I);

    auto *IntType = IsPointer ? IntPtrType : cast<IntegerType>(OperandType);
    Value *New = B.CreateCall(SegmentGlobals[TheReplacement.Segment]);
    New = B.CreateZExtOrTrunc(New, IntType);
    New = B.CreateAdd(New, ConstantInt::get(IntType, TheReplacement.Offset));

    if (IsPointer)
      New = B.CreateIntToPtr(New, OperandType);

    Operand.set(New);
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
