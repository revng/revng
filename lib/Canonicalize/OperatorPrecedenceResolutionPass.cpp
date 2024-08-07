//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <array>

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/Value.h"
#include "llvm/Pass.h"

#include "revng/Support/Assert.h"
#include "revng/Support/FunctionTags.h"
#include "revng/Support/OpaqueFunctionsPool.h"

#include "revng-c/Support/FunctionTags.h"
#include "revng-c/Support/IRHelpers.h"

using namespace llvm;

static cl::opt<std::string> LanguageName("language",
                                         cl::desc("Language for operator "
                                                  "precedence table"),
                                         cl::Hidden,
                                         cl::init("c"));

namespace {
constexpr unsigned getInstructionLLVMOpcodeCount() {
#define LAST_OTHER_INST(NR) return NR;
#include "llvm/IR/Instruction.def"
}
} // namespace

enum CustomInstruction : unsigned {
  AddressOf = getInstructionLLVMOpcodeCount() + 1,
  Assignment = getInstructionLLVMOpcodeCount() + 2,
  Cast = getInstructionLLVMOpcodeCount() + 3,
  Indirection = getInstructionLLVMOpcodeCount() + 4,
  MemberAccess = getInstructionLLVMOpcodeCount() + 5,
  LocalVariable = getInstructionLLVMOpcodeCount() + 6,
  Transparent = getInstructionLLVMOpcodeCount() + 7,
  SegmentRef = getInstructionLLVMOpcodeCount() + 8,
  UnaryMinus = getInstructionLLVMOpcodeCount() + 9,
  BinaryNot = getInstructionLLVMOpcodeCount() + 10,
  BooleanNot = getInstructionLLVMOpcodeCount() + 11
};

enum Associativity {
  LeftToRight,
  RightToLeft
};

enum Arity : unsigned {
  Unary,
  Binary,
  Ternary,
  NAry
};

struct OperatorInfo {
  unsigned Opcode;
  int Precedence;
  Associativity Associativity;
  Arity Arity;
};

// Table that maps LLVM opcodes to the equivalent C operator precedence priority
static constexpr std::array<const OperatorInfo, 37>
  LLVMOpcodeToCOpPrecedenceArray{
    // OperatorInfo{ OpCode, Precedence, Associativity, Arity },
    OperatorInfo{ Instruction::Select, 1, RightToLeft, Ternary },
    OperatorInfo{ Instruction::Or, 2, LeftToRight, Binary },
    OperatorInfo{ Instruction::Xor, 3, LeftToRight, Binary },
    OperatorInfo{ Instruction::And, 4, LeftToRight, Binary },
    OperatorInfo{ Instruction::ICmp, 5, LeftToRight, Binary },
    OperatorInfo{ Instruction::Shl, 6, LeftToRight, Binary },
    OperatorInfo{ Instruction::LShr, 6, LeftToRight, Binary },
    OperatorInfo{ Instruction::AShr, 6, LeftToRight, Binary },
    OperatorInfo{ Instruction::Add, 7, LeftToRight, Binary },
    OperatorInfo{ Instruction::Sub, 7, LeftToRight, Binary },
    OperatorInfo{ Instruction::Mul, 8, LeftToRight, Binary },
    OperatorInfo{ Instruction::UDiv, 8, LeftToRight, Binary },
    OperatorInfo{ Instruction::SDiv, 8, LeftToRight, Binary },
    OperatorInfo{ Instruction::URem, 8, LeftToRight, Binary },
    OperatorInfo{ Instruction::SRem, 8, LeftToRight, Binary },
    OperatorInfo{ Instruction::SExt, 9, RightToLeft, Unary },
    OperatorInfo{ Instruction::Trunc, 9, RightToLeft, Unary },
    OperatorInfo{ Instruction::ZExt, 9, RightToLeft, Unary },
    OperatorInfo{ CustomInstruction::AddressOf, 9, RightToLeft, Unary },
    OperatorInfo{ CustomInstruction::Indirection, 9, RightToLeft, Unary },
    OperatorInfo{ CustomInstruction::Cast, 9, RightToLeft, Unary },
    OperatorInfo{ CustomInstruction::UnaryMinus, 9, RightToLeft, Unary },
    OperatorInfo{ CustomInstruction::BinaryNot, 9, RightToLeft, Unary },
    OperatorInfo{ CustomInstruction::BooleanNot, 9, RightToLeft, Unary },
    OperatorInfo{ CustomInstruction::MemberAccess, 10, LeftToRight, Unary },
    OperatorInfo{ Instruction::Call, 10, LeftToRight, NAry },
  };

static constexpr std::array<const OperatorInfo, 37>
  LLVMOpcodeToNopOpPrecedenceArray{
    // OperatorInfo{ OpCode, Precedence, Associativity, Arity },
    OperatorInfo{ Instruction::Select, 0, RightToLeft, Ternary },
    OperatorInfo{ Instruction::Or, 0, LeftToRight, Binary },
    OperatorInfo{ Instruction::Xor, 0, LeftToRight, Binary },
    OperatorInfo{ Instruction::And, 0, LeftToRight, Binary },
    OperatorInfo{ Instruction::ICmp, 0, LeftToRight, Binary },
    OperatorInfo{ Instruction::Shl, 0, LeftToRight, Binary },
    OperatorInfo{ Instruction::LShr, 0, LeftToRight, Binary },
    OperatorInfo{ Instruction::AShr, 0, LeftToRight, Binary },
    OperatorInfo{ Instruction::Add, 0, LeftToRight, Binary },
    OperatorInfo{ Instruction::Sub, 0, LeftToRight, Binary },
    OperatorInfo{ Instruction::Mul, 0, LeftToRight, Binary },
    OperatorInfo{ Instruction::UDiv, 0, LeftToRight, Binary },
    OperatorInfo{ Instruction::SDiv, 0, LeftToRight, Binary },
    OperatorInfo{ Instruction::URem, 0, LeftToRight, Binary },
    OperatorInfo{ Instruction::SRem, 0, LeftToRight, Binary },
    OperatorInfo{ Instruction::SExt, 0, RightToLeft, Unary },
    OperatorInfo{ Instruction::Trunc, 0, RightToLeft, Unary },
    OperatorInfo{ Instruction::ZExt, 0, RightToLeft, Unary },
    OperatorInfo{ CustomInstruction::AddressOf, 0, RightToLeft, Unary },
    OperatorInfo{ CustomInstruction::Indirection, 0, RightToLeft, Unary },
    OperatorInfo{ CustomInstruction::Cast, 0, RightToLeft, Unary },
    OperatorInfo{ CustomInstruction::UnaryMinus, 0, RightToLeft, Unary },
    OperatorInfo{ CustomInstruction::BinaryNot, 0, RightToLeft, Unary },
    OperatorInfo{ CustomInstruction::BooleanNot, 0, RightToLeft, Unary },
    OperatorInfo{ CustomInstruction::MemberAccess, 0, LeftToRight, Unary },
    OperatorInfo{ Instruction::Call, 0, LeftToRight, NAry },
  };

static OperatorInfo
getPrecedence(const std::array<const OperatorInfo, 37> *Table,
              unsigned Opcode) {
  auto It = llvm::find_if(*Table, [&](const auto &Elem) {
    return Elem.Opcode == Opcode;
  });
  revng_assert(It != Table->end());
  return *It;
}

static bool isCustomOpcode(const Value *I) {
  const auto *Call = dyn_cast<CallInst>(I);
  if (nullptr == Call)
    return false;

  const auto *CalledFunc = getCalledFunction(Call);
  if (nullptr == CalledFunc)
    return false;

  if (FunctionTags::AddressOf.isTagOf(CalledFunc)
      or FunctionTags::AllocatesLocalVariable.isTagOf(CalledFunc)
      or FunctionTags::Assign.isTagOf(CalledFunc)
      or FunctionTags::BinaryNot.isTagOf(CalledFunc)
      or FunctionTags::BooleanNot.isTagOf(CalledFunc)
      or FunctionTags::Copy.isTagOf(CalledFunc)
      or FunctionTags::ModelCast.isTagOf(CalledFunc)
      or FunctionTags::ModelGEP.isTagOf(CalledFunc)
      or FunctionTags::ModelGEPRef.isTagOf(CalledFunc)
      or FunctionTags::OpaqueExtractValue.isTagOf(CalledFunc)
      or FunctionTags::SegmentRef.isTagOf(CalledFunc)
      or FunctionTags::UnaryMinus.isTagOf(CalledFunc))
    return true;

  return false;
}

static unsigned getCustomOpcode(const Instruction *I) {
  revng_assert(isCustomOpcode(I));

  auto *CalledFunc = getCalledFunction(cast<CallInst>(I));
  revng_assert(CalledFunc);

  if (FunctionTags::AddressOf.isTagOf(CalledFunc))
    return CustomInstruction::AddressOf;
  else if (FunctionTags::Assign.isTagOf(CalledFunc))
    return CustomInstruction::Assignment;
  else if (FunctionTags::AllocatesLocalVariable.isTagOf(CalledFunc))
    return CustomInstruction::LocalVariable;
  else if (FunctionTags::ModelCast.isTagOf(CalledFunc))
    return CustomInstruction::Cast;
  else if (FunctionTags::ModelGEP.isTagOf(CalledFunc)) {
    auto *Call = cast<CallInst>(I);
    if (Call->arg_size() > 3)
      return CustomInstruction::MemberAccess;
    auto *ConstantArrayIndex = dyn_cast<ConstantInt>(Call->getArgOperand(2));
    if (ConstantArrayIndex and ConstantArrayIndex->isZero())
      return CustomInstruction::Indirection;
    return CustomInstruction::MemberAccess;
  } else if (FunctionTags::ModelGEPRef.isTagOf(CalledFunc)) {
    if (cast<CallInst>(I)->arg_size() > 2)
      return CustomInstruction::MemberAccess;
    return CustomInstruction::Transparent;
  } else if (FunctionTags::OpaqueExtractValue.isTagOf(CalledFunc)) {
    return CustomInstruction::MemberAccess;
  } else if (FunctionTags::Copy.isTagOf(CalledFunc)) {
    return CustomInstruction::Transparent;
  } else if (FunctionTags::SegmentRef.isTagOf(CalledFunc)) {
    return CustomInstruction::SegmentRef;
  } else if (FunctionTags::UnaryMinus.isTagOf(CalledFunc)) {
    return CustomInstruction::UnaryMinus;
  } else if (FunctionTags::BinaryNot.isTagOf(CalledFunc)) {
    return CustomInstruction::BinaryNot;
  } else if (FunctionTags::BooleanNot.isTagOf(CalledFunc)) {
    return CustomInstruction::BooleanNot;
  }

  revng_abort("unhandled custom opcode");
}

static unsigned getOpcode(const Instruction *I) {
  if (isa<CallInst>(I))
    if (isCustomOpcode(I))
      return getCustomOpcode(I);

  return I->getOpcode();
}

static bool isImplicitCast(const Value *V) {
  if (not isCallToTagged(V, FunctionTags::ModelCast))
    return false;

  // If it is an implicit cast, omit the parentheses.
  const llvm::CallInst *ModelCastCall = cast<llvm::CallInst>(V);
  return cast<llvm::ConstantInt>(ModelCastCall->getArgOperand(2))->isOne();
}

static bool isTransparentOpCode(const Value *V) {
  if (isImplicitCast(V))
    return true;

  if (isa<IntToPtrInst>(V) or isa<PtrToIntInst>(V) or isa<BitCastInst>(V)
      or isa<FreezeInst>(V))
    return true;

  const auto *I = dyn_cast<Instruction>(V);
  if (nullptr == I)
    return false;

  return isCustomOpcode(I)
         and getCustomOpcode(I) == CustomInstruction::Transparent;
}

static Value *traverseTransparentOpcodes(Value *I) {
  while (isa<Instruction>(I) and isTransparentOpCode(I)) {
    if (isa<IntToPtrInst>(I) or isa<PtrToIntInst>(I) or isa<BitCastInst>(I)
        or isa<FreezeInst>(I))
      I = cast<Instruction>(I)->getOperand(0);
    else if (auto *CallToCopy = getCallToTagged(I, FunctionTags::Copy))
      I = CallToCopy->getArgOperand(0);
    else if (auto *CallToMGR = getCallToTagged(I, FunctionTags::ModelGEPRef))
      I = CallToMGR->getArgOperand(1);
    else if (isImplicitCast(I))
      I = cast<CallInst>(I)->getArgOperand(1);
    else
      revng_abort("unexpected transparent opcode");
  }
  return I;
}

struct OperatorPrecedenceResolutionPass : public FunctionPass {
private:
  const std::array<const OperatorInfo, 37>
    *LLVMOpcodeToLangOpPrecedenceArray = nullptr;

public:
  static char ID;

  OperatorPrecedenceResolutionPass() : FunctionPass(ID) {
    if (LanguageName == "C" || LanguageName == "c")
      LLVMOpcodeToLangOpPrecedenceArray = &LLVMOpcodeToCOpPrecedenceArray;
    else if (LanguageName == "NOP" || LanguageName == "nop")
      LLVMOpcodeToLangOpPrecedenceArray = &LLVMOpcodeToNopOpPrecedenceArray;
    revng_assert(LLVMOpcodeToLangOpPrecedenceArray);
  }

  bool runOnFunction(Function &F) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
  }

public:
  bool needsParentheses(Instruction *I, Use &U);
};

using OPRP = OperatorPrecedenceResolutionPass;

bool OPRP::needsParentheses(Instruction *I, Use &U) {
  // Control flow instructions never need parentheses around their operands.
  if (isa<BranchInst>(I) or isa<SwitchInst>(I) or isa<ReturnInst>(I))
    return false;

  // The following are transparent in C as we emit it, so we never put
  // parentheses around their operands.
  if (isTransparentOpCode(I))
    return false;

  // The remaining instructions can be divided in 2 categories:
  // - A group of instruction for which we always have to evaluate if we need to
  //   emit parentheses around all their operands, using the operator
  //   precedencence and associativity.
  // - A group of instructions for which we only have to evaluate if we need to
  //   emit parentheses around some of their operands, depending on the specific
  //   instruction. On all the other operands we never need to emit parentheses
  //   and we can bail out early.
  // So we now detect the second group, to bail out early in all the cases where
  // it's necessary, leaving the evaluation of operator precedence and
  // associativity only for later when really necessary.

  if (isa<CallInst>(I) and isCustomOpcode(I)) {
    switch (getCustomOpcode(I)) {
    // These instructions never need parentheses around their operands as well.
    case CustomInstruction::Assignment:
    case CustomInstruction::LocalVariable:
    case CustomInstruction::SegmentRef:
      return false;

    // These should be handled as regular operations, so we just go on below.
    case CustomInstruction::BinaryNot:
    case CustomInstruction::BooleanNot:
    case CustomInstruction::UnaryMinus:
      break;

    case CustomInstruction::MemberAccess: {
      if (isCallToTagged(I, FunctionTags::OpaqueExtractValue)) {
        // For OpaqueExtractValues we only need to evaluate parentheses around
        // the first operand, which is the aggregate, not on the others.
        if (U.getOperandNo() != 0)
          return false;
      } else if (isCallToTagged(I, FunctionTags::ModelGEP)
                 or isCallToTagged(I, FunctionTags::ModelGEPRef)) {
        // For various kinds of ModelGEPs the only operand for which we care
        // about operator precedence is the operand representing the base
        // address. All the others can be ignored
        if (U.getOperandNo() != 1)
          return false;
      } else {
        revng_abort("unexpected MemberAccess opcode");
      }
    } break;

    case CustomInstruction::Indirection:
    case CustomInstruction::AddressOf:
    case CustomInstruction::Cast: {
      // For these operations we only have to consider operator precedence
      // around operand 1, for all the others just return false, since they
      // never need parentheses.
      if (U.getOperandNo() != 1)
        return false;
    } break;

    // This should be already handled above.
    case CustomInstruction::Transparent:
      revng_abort("transparent opcodes should have been handled earlier");

    default:
      revng_abort("Unexpected custom opcode");
    }
  }

  // Here we know by having looked at the instruction that we might need to look
  // at the operand as well, to decide if we have to emit parentheses based on
  // the operator precedence and associativity.
  //
  // However, there are some kinds of operands for which we already know that we
  // never need to emit parentheses, so in those cases we bail out early.

  // Traverse all the transparent opcodes around the operand, until we can
  // really see the operand itself.
  Instruction *Op = dyn_cast<Instruction>(traverseTransparentOpcodes(U.get()));

  // If the operand is not an instruction (e.g. constant, arguments), don't emit
  // parentheses, because in C we always emit it as an identifiers, which never
  // needs parentheses.
  if (nullptr == Op)
    return false;

  // If the operand is a call to qemu helpers or intrinsic we know that we
  // always emit a local variable for it, so we don't have to emit parentheses
  if (isCallToTagged(Op, FunctionTags::QEMU)
      or isCallToTagged(Op, FunctionTags::Helper)
      or isCallToTagged(Op, FunctionTags::Exceptional)
      or isa<IntrinsicInst>(Op))
    return false;

  // If the operand is one of the following custom opcode, there's no need of
  // parentheses around it.
  if (isCustomOpcode(Op)
      and (getCustomOpcode(Op) == CustomInstruction::Assignment
           or getCustomOpcode(Op) == CustomInstruction::LocalVariable
           or getCustomOpcode(Op) == CustomInstruction::SegmentRef))
    return false;

  // For calls that are not custom opcodes, we only have to check the operator
  // precedence for the called operand, not for the arguments.
  if (auto *Call = dyn_cast<CallInst>(I); Call and not isCustomOpcode(Call))
    if (&U != &Call->getCalledOperandUse())
      return false;

  auto [InstructionOpcode,
        InstructionPrecedence,
        InstructionAssociativity,
        InstructionArity] = getPrecedence(LLVMOpcodeToLangOpPrecedenceArray,
                                          getOpcode(I));

  auto [OperandOpcode,
        OperandPrecedence,
        OperandAssociativity,
        OperandArity] = getPrecedence(LLVMOpcodeToLangOpPrecedenceArray,
                                      getOpcode(Op));

  auto Cmp = InstructionPrecedence <=> OperandPrecedence;
  // If the precedence of the instruction and the operand is the same, we have
  // to discriminate by Associativity and by Arity
  if (Cmp == 0) {

    revng_assert(InstructionAssociativity == OperandAssociativity);

    switch (InstructionArity) {
    case NAry: {
      revng_assert(InstructionOpcode == Instruction::Call);
      // Basically this is a call, and we're analyzing the called operand, which
      // has the same precedence (the highest) and associativity of the call.
      // So we can just never emit the parenthesis because associativity will do
      // the work for us.
    } break;

    case Unary: {
      // If the instruction operator is unary, and the precedence is
      // the same, no parentheses are needed, and there's no need to check
      // associativity, since all the operators with the same precedence have
      // the same associativity.
      return false;
    } break;

    case Binary: {
      revng_assert(I->getNumOperands() == 2);
      revng_assert(isa<BinaryOperator>(I) or isa<CmpInst>(I));
      // If there are 2 operands, Instruction can only be a binary operator
      // (arithmetic or bitwise).
      enum Side : unsigned {
        LHS = 0,
        RHS
      };
      Side OpSide = U.getOperandNo() == 0 ? LHS : RHS;
      // If the Instruction associativity leads to preserving semantics we don't
      // need parentheses, otherwise we do.
      if (InstructionAssociativity == LeftToRight and OpSide != RHS)
        return false;

      if (InstructionAssociativity == RightToLeft and OpSide != LHS)
        return false;

      return true;
    } break;

    case Ternary: {
      revng_assert(I->getNumOperands() == 3);
      revng_assert(isa<SelectInst>(I));
      // This is basically a ternary with an operand that is another ternary, so
      // we always emit parentheses to avoid nasty nested things that are hard
      // to read and understand.
      return true;
    } break;

    default:
      revng_abort("unexpected arity");
    }
  }

  // If the precedence of the instruction is different, we only need parentheses
  // when the precedence of the instruction is higher than the precedence of the
  // operand. In this case we never need to check associativity.
  return Cmp > 0;
}

bool OPRP::runOnFunction(Function &F) {
  OpaqueFunctionsPool<Type *> ParenthesesPool(F.getParent(), false);
  initParenthesesPool(ParenthesesPool);

  std::vector<std::pair<Instruction *, Use *>> InstructionsToBeParenthesized;
  for (BasicBlock &BB : F)
    for (Instruction &I : BB)
      for (Use &Op : I.operands())
        if (needsParentheses(&I, Op))
          InstructionsToBeParenthesized.emplace_back(&I, &Op);

  if (InstructionsToBeParenthesized.empty()) {
    // OPRP has executed for this function
    F.setMetadata(ExplicitParenthesesMDName, MDNode::get(F.getContext(), {}));
    return false;
  }

  IRBuilder<> Builder(F.getContext());
  for (const auto &[I, Op] : InstructionsToBeParenthesized) {
    Builder.SetInsertPoint(I);
    Instruction *Ins = cast<Instruction>(Op->get());

    Type *OpToReplaceType = I->getOperand(Op->getOperandNo())->getType();

    auto *ParenthesisFunction = ParenthesesPool.get(OpToReplaceType,
                                                    OpToReplaceType,
                                                    { Ins->getType() },
                                                    "parentheses");
    Value *Call = Builder.CreateCall(ParenthesisFunction, { Ins });
    I->setOperand(Op->getOperandNo(), Call);
  }

  // OPRP has executed for this function
  F.setMetadata(ExplicitParenthesesMDName, MDNode::get(F.getContext(), {}));

  return true;
}

char OPRP::ID = 0;

static RegisterPass<OPRP> X("operatorprecedence-resolution",
                            "A simple pass that resolves precedence of "
                            "operators with parentheses appropriately.",
                            false,
                            false);
