//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <array>

#include "llvm/IR/Function.h"
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
#include "revng/Support/IRHelpers.h"
#include "revng/Support/OpaqueFunctionsPool.h"

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

enum class Associativity {
  LeftToRight,
  RightToLeft
};

enum class Arity : unsigned {
  Unary,
  Binary,
  Ternary,
  NAry
};

struct OperatorInfo {
  uint64_t Precedence;

  // Because people are not especially good when dealing with implicit
  // precedence, it's sometimes better to emit extra parentheses. This is what
  // this value is for. As long as the difference between classes is below this
  // value, the parentheses are going to be emitted even if they are not
  // necessary.
  //
  // TODO: even thought, this simplistic approach is enough for now, we might
  //       want to switch to something more expressive at some point.
  uint64_t NumberOfClassesToForceParenthesesFor;

  Associativity Associativity;
  Arity Arity;
};

static OperatorInfo getPrecedenceImpl(const Instruction &I) {
  // Custom instructions first
  if (auto *Call = llvm::dyn_cast<llvm::CallInst>(&I)) {
    if (auto *CalledFunc = getCalledFunction(Call)) {

      // AddressOf
      if (FunctionTags::AddressOf.isTagOf(CalledFunc))
        return { 2, 0, Associativity::RightToLeft, Arity::Unary };

      // Cast
      else if (FunctionTags::ModelCast.isTagOf(CalledFunc))
        return { 2, 0, Associativity::RightToLeft, Arity::Unary };

      else if (FunctionTags::ModelGEP.isTagOf(CalledFunc)) {
        // Indirection
        if (auto *ArrayIndex = dyn_cast<ConstantInt>(Call->getArgOperand(2)))
          if (Call->arg_size() <= 3 and ArrayIndex->isZero())
            return { 2, 0, Associativity::RightToLeft, Arity::Unary };

        // MemberAccess
        return { 1, 0, Associativity::LeftToRight, Arity::Unary };

      } else if (FunctionTags::ModelGEPRef.isTagOf(CalledFunc)) {
        if (Call->arg_size() > 2) {
          // MemberAccess
          return { 1, 0, Associativity::LeftToRight, Arity::Unary };
        } else {
          revng_abort("How did a transparent instruction got here?");
        }

      } else if (FunctionTags::OpaqueExtractValue.isTagOf(CalledFunc)) {
        // MemberAccess
        return { 1, 0, Associativity::LeftToRight, Arity::Unary };

        // UnaryMinus
      } else if (FunctionTags::UnaryMinus.isTagOf(CalledFunc)) {
        return { 2, 0, Associativity::RightToLeft, Arity::Unary };

        // BinaryNot
      } else if (FunctionTags::BinaryNot.isTagOf(CalledFunc)) {
        return { 2, 0, Associativity::RightToLeft, Arity::Unary };

        // BooleanNot
      } else if (FunctionTags::BooleanNot.isTagOf(CalledFunc)) {
        return { 2, 0, Associativity::RightToLeft, Arity::Unary };
      }
    }

    // Catch all the other calls
    return { 0, 0, Associativity::LeftToRight, Arity::NAry };
  }

  // It's not a call: map it normally
  switch (I.getOpcode()) {
  // Ternary operator (?:)
  case Instruction::Select:
    return { 13, 0, Associativity::RightToLeft, Arity::Ternary };

  // Or (|| and |)
  case Instruction::Or:
    if (I.getType()->isIntegerTy(1))
      return { 12, 0, Associativity::LeftToRight, Arity::Binary };
    else
      return { 10, 7, Associativity::LeftToRight, Arity::Binary };

  // Xor (^)
  case Instruction::Xor:
    return { 9, 6, Associativity::LeftToRight, Arity::Binary };

  // And (&& and &)
  case Instruction::And:
    if (I.getType()->isIntegerTy(1))
      return { 11, 0, Associativity::LeftToRight, Arity::Binary };
    else
      return { 8, 5, Associativity::LeftToRight, Arity::Binary };

  // All the comparisons
  case Instruction::ICmp:
    return { 6, 0, Associativity::LeftToRight, Arity::Binary };

  // Byte-wise shifts
  case Instruction::Shl:
    return { 5, 2, Associativity::LeftToRight, Arity::Binary };
  case Instruction::LShr:
    return { 5, 2, Associativity::LeftToRight, Arity::Binary };
  case Instruction::AShr:
    return { 5, 2, Associativity::LeftToRight, Arity::Binary };

  // Addition and subtraction
  case Instruction::Add:
    return { 4, 0, Associativity::LeftToRight, Arity::Binary };
  case Instruction::Sub:
    return { 4, 0, Associativity::LeftToRight, Arity::Binary };

  // Multiplication, division and remainder
  case Instruction::Mul:
    return { 3, 0, Associativity::LeftToRight, Arity::Binary };
  case Instruction::UDiv:
    return { 3, 0, Associativity::LeftToRight, Arity::Binary };
  case Instruction::SDiv:
    return { 3, 0, Associativity::LeftToRight, Arity::Binary };
  case Instruction::URem:
    return { 3, 0, Associativity::LeftToRight, Arity::Binary };
  case Instruction::SRem:
    return { 3, 0, Associativity::LeftToRight, Arity::Binary };

  // Casts
  case Instruction::SExt:
    return { 2, 0, Associativity::RightToLeft, Arity::Unary };
  case Instruction::Trunc:
    return { 2, 0, Associativity::RightToLeft, Arity::Unary };
  case Instruction::ZExt:
    return { 2, 0, Associativity::RightToLeft, Arity::Unary };

  default:
    revng_abort("unsupported opcode");
  }
}

static OperatorInfo getPrecedence(const Instruction &I) {
  OperatorInfo Result = getPrecedenceImpl(I);
  if (LanguageName == "NOP" || LanguageName == "nop")
    Result.Precedence = 0;
  else
    revng_assert(LanguageName == "C" || LanguageName == "c");

  return Result;
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
public:
  static char ID;

  OperatorPrecedenceResolutionPass() : FunctionPass(ID) {}

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

  auto &&[InstructionPrecedence,
          NumberOfClassesToForceParenthesesFor,
          InstructionAssociativity,
          InstructionArity] = getPrecedence(*I);

  auto &&[OperandPrecedence,
          _,
          OperandAssociativity,
          OperandArity] = getPrecedence(*Op);

  // If the precedence of the instruction and the operand is the same, we have
  // to discriminate by Associativity and by Arity
  if (InstructionPrecedence == OperandPrecedence) {

    revng_assert(InstructionAssociativity == OperandAssociativity);

    switch (InstructionArity) {
    case Arity::NAry:
      revng_assert(llvm::isa<CallInst>(I));
      // Basically this is a call, and we're analyzing the called operand, which
      // has the same precedence (the highest) and associativity of the call.
      // So we can just never emit the parenthesis because associativity will do
      // the work for us.
      return false;

    case Arity::Unary:
      // If the instruction operator is unary, and the precedence is
      // the same, no parentheses are needed, and there's no need to check
      // associativity, since all the operators with the same precedence have
      // the same associativity.
      return false;

    case Arity::Binary: {
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
      if (InstructionAssociativity == Associativity::LeftToRight
          and OpSide != RHS)
        return false;

      if (InstructionAssociativity == Associativity::RightToLeft
          and OpSide != LHS)
        return false;

      return true;
    }

    case Arity::Ternary:
      revng_assert(I->getNumOperands() == 3);
      revng_assert(isa<SelectInst>(I));
      // This is basically a ternary with an operand that is another ternary, so
      // we always emit parentheses to avoid nasty nested things that are hard
      // to read and understand.
      return true;

    default:
      revng_abort("unexpected arity");
    }

  } else if (InstructionPrecedence < OperandPrecedence) {
    // If the instruction takes precedence over the operand, we always need
    // to emit the parentheses: otherwise the expression is not going to be
    // semantically correct.
    return true;

  } else {
    // If the operand takes precedence over the instruction, we only emit
    // parentheses if they help readability.
    uint64_t PrecedenceDifference = InstructionPrecedence - OperandPrecedence;
    return PrecedenceDifference <= NumberOfClassesToForceParenthesesFor;
  }
}

bool OPRP::runOnFunction(Function &F) {
  auto ParenthesesPool = FunctionTags::Parentheses.getPool(*F.getParent());

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
    Builder.SetCurrentDebugLocation(Ins->getDebugLoc());

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
