//
// Copyright rev.ng Srls. See LICENSE.md for details.
//

#include <array>

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Pass.h"

#include "revng/Support/Assert.h"
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

enum Associativity : unsigned { NONE = 0, LEFT_TO_RIGHT, RIGHT_TO_LEFT };

enum CustomInstruction : unsigned {
  AddressOf = getInstructionLLVMOpcodeCount() + 1,
  Assignment = getInstructionLLVMOpcodeCount() + 2,
  Cast = getInstructionLLVMOpcodeCount() + 3,
  Indirection = getInstructionLLVMOpcodeCount() + 4,
  MemberAccess = getInstructionLLVMOpcodeCount() + 5
};

struct InstToOpPrec {
  unsigned InstructionOpcode;
  int Precedence;
  Associativity Associativity;

  constexpr InstToOpPrec(unsigned InstructionOpcode,
                         int Precedence,
                         enum Associativity Associativity) :
    InstructionOpcode(InstructionOpcode),
    Precedence(Precedence),
    Associativity(Associativity) {}
};

// Table that maps LLVM opcodes to the equivalent C operator precedence priority
static constexpr std::array<const InstToOpPrec, 35>
  LLVMOpcodeToCOpPrecedenceArray{
    { { InstToOpPrec(CustomInstruction::Assignment, 0, RIGHT_TO_LEFT) },
      { InstToOpPrec(Instruction::Select, 1, RIGHT_TO_LEFT) },
      { InstToOpPrec(Instruction::Or, 2, LEFT_TO_RIGHT) },
      { InstToOpPrec(Instruction::Xor, 3, LEFT_TO_RIGHT) },
      { InstToOpPrec(Instruction::And, 4, LEFT_TO_RIGHT) },
      { InstToOpPrec(Instruction::ICmp, 5, LEFT_TO_RIGHT) },
      { InstToOpPrec(Instruction::FCmp, 5, LEFT_TO_RIGHT) },
      { InstToOpPrec(Instruction::Shl, 6, LEFT_TO_RIGHT) },
      { InstToOpPrec(Instruction::LShr, 6, LEFT_TO_RIGHT) },
      { InstToOpPrec(Instruction::AShr, 6, LEFT_TO_RIGHT) },
      { InstToOpPrec(Instruction::Add, 7, LEFT_TO_RIGHT) },
      { InstToOpPrec(Instruction::FAdd, 7, LEFT_TO_RIGHT) },
      { InstToOpPrec(Instruction::Sub, 7, LEFT_TO_RIGHT) },
      { InstToOpPrec(Instruction::FSub, 7, LEFT_TO_RIGHT) },
      { InstToOpPrec(Instruction::Mul, 8, LEFT_TO_RIGHT) },
      { InstToOpPrec(Instruction::FMul, 8, LEFT_TO_RIGHT) },
      { InstToOpPrec(Instruction::UDiv, 8, LEFT_TO_RIGHT) },
      { InstToOpPrec(Instruction::SDiv, 8, LEFT_TO_RIGHT) },
      { InstToOpPrec(Instruction::FDiv, 8, LEFT_TO_RIGHT) },
      { InstToOpPrec(Instruction::URem, 8, LEFT_TO_RIGHT) },
      { InstToOpPrec(Instruction::SRem, 8, LEFT_TO_RIGHT) },
      { InstToOpPrec(Instruction::FRem, 8, LEFT_TO_RIGHT) },
      { InstToOpPrec(Instruction::FNeg, 9, RIGHT_TO_LEFT) },
      { InstToOpPrec(Instruction::Load, 9, RIGHT_TO_LEFT) },
      { InstToOpPrec(Instruction::SExt, 9, RIGHT_TO_LEFT) },
      { InstToOpPrec(Instruction::ZExt, 9, RIGHT_TO_LEFT) },
      { InstToOpPrec(Instruction::Trunc, 9, RIGHT_TO_LEFT) },
      { InstToOpPrec(Instruction::IntToPtr, 9, RIGHT_TO_LEFT) },
      { InstToOpPrec(Instruction::PtrToInt, 9, RIGHT_TO_LEFT) },
      { InstToOpPrec(Instruction::BitCast, 9, RIGHT_TO_LEFT) },
      { InstToOpPrec(Instruction::GetElementPtr, 9, RIGHT_TO_LEFT) },
      { InstToOpPrec(CustomInstruction::AddressOf, 9, RIGHT_TO_LEFT) },
      { InstToOpPrec(CustomInstruction::Indirection, 9, RIGHT_TO_LEFT) },
      { InstToOpPrec(CustomInstruction::Cast, 9, RIGHT_TO_LEFT) },
      { InstToOpPrec(CustomInstruction::MemberAccess, 10, LEFT_TO_RIGHT) } }
  };

static constexpr std::array<const InstToOpPrec, 35>
  LLVMOpcodeToNopOpPrecedenceArray{
    { { InstToOpPrec(CustomInstruction::Assignment, 0, RIGHT_TO_LEFT) },
      { InstToOpPrec(Instruction::Select, 0, RIGHT_TO_LEFT) },
      { InstToOpPrec(Instruction::Or, 0, LEFT_TO_RIGHT) },
      { InstToOpPrec(Instruction::Xor, 0, LEFT_TO_RIGHT) },
      { InstToOpPrec(Instruction::And, 0, LEFT_TO_RIGHT) },
      { InstToOpPrec(Instruction::ICmp, 0, LEFT_TO_RIGHT) },
      { InstToOpPrec(Instruction::FCmp, 0, LEFT_TO_RIGHT) },
      { InstToOpPrec(Instruction::Shl, 0, LEFT_TO_RIGHT) },
      { InstToOpPrec(Instruction::LShr, 0, LEFT_TO_RIGHT) },
      { InstToOpPrec(Instruction::AShr, 0, LEFT_TO_RIGHT) },
      { InstToOpPrec(Instruction::Add, 0, LEFT_TO_RIGHT) },
      { InstToOpPrec(Instruction::FAdd, 0, LEFT_TO_RIGHT) },
      { InstToOpPrec(Instruction::Sub, 0, LEFT_TO_RIGHT) },
      { InstToOpPrec(Instruction::FSub, 0, LEFT_TO_RIGHT) },
      { InstToOpPrec(Instruction::Mul, 0, LEFT_TO_RIGHT) },
      { InstToOpPrec(Instruction::FMul, 0, LEFT_TO_RIGHT) },
      { InstToOpPrec(Instruction::UDiv, 0, LEFT_TO_RIGHT) },
      { InstToOpPrec(Instruction::SDiv, 0, LEFT_TO_RIGHT) },
      { InstToOpPrec(Instruction::FDiv, 0, LEFT_TO_RIGHT) },
      { InstToOpPrec(Instruction::URem, 0, LEFT_TO_RIGHT) },
      { InstToOpPrec(Instruction::SRem, 0, LEFT_TO_RIGHT) },
      { InstToOpPrec(Instruction::FRem, 0, LEFT_TO_RIGHT) },
      { InstToOpPrec(Instruction::FNeg, 0, RIGHT_TO_LEFT) },
      { InstToOpPrec(Instruction::Load, 0, RIGHT_TO_LEFT) },
      { InstToOpPrec(Instruction::SExt, 0, RIGHT_TO_LEFT) },
      { InstToOpPrec(Instruction::ZExt, 0, RIGHT_TO_LEFT) },
      { InstToOpPrec(Instruction::Trunc, 0, RIGHT_TO_LEFT) },
      { InstToOpPrec(Instruction::IntToPtr, 0, RIGHT_TO_LEFT) },
      { InstToOpPrec(Instruction::PtrToInt, 0, RIGHT_TO_LEFT) },
      { InstToOpPrec(Instruction::BitCast, 0, RIGHT_TO_LEFT) },
      { InstToOpPrec(Instruction::GetElementPtr, 0, RIGHT_TO_LEFT) },
      { InstToOpPrec(CustomInstruction::AddressOf, 0, RIGHT_TO_LEFT) },
      { InstToOpPrec(CustomInstruction::Indirection, 0, RIGHT_TO_LEFT) },
      { InstToOpPrec(CustomInstruction::Cast, 0, RIGHT_TO_LEFT) },
      { InstToOpPrec(CustomInstruction::MemberAccess, 0, LEFT_TO_RIGHT) } }
  };

static auto
findOpcode(const std::array<const InstToOpPrec, 35> *Table, unsigned Opcode) {
  return find_if(*Table, [&](const auto &Elem) {
    return Elem.InstructionOpcode == Opcode;
  });
}

static bool isCustomOpcode(Instruction *I) {
  auto *CalledFunc = cast<CallInst>(I)->getCalledFunction();
  if (!CalledFunc)
    return false;

  if (FunctionTags::AddressOf.isTagOf(CalledFunc)
      || FunctionTags::AssignmentMarker.isTagOf(CalledFunc)
      || (FunctionTags::ModelCast.isTagOf(CalledFunc))
      || (FunctionTags::ModelGEP.isTagOf(CalledFunc)))
    return true;

  return false;
}

static unsigned getCustomOpcode(Instruction *I) {
  auto *CalledFunc = cast<CallInst>(I)->getCalledFunction();

  if (FunctionTags::AddressOf.isTagOf(CalledFunc))
    return CustomInstruction::AddressOf;
  else if (FunctionTags::AssignmentMarker.isTagOf(CalledFunc))
    return CustomInstruction::Assignment;
  else if (FunctionTags::ModelCast.isTagOf(CalledFunc))
    return CustomInstruction::Cast;
  else if (FunctionTags::ModelGEP.isTagOf(CalledFunc)) {
    if (cast<CallInst>(I)->getNumArgOperands() > 2)
      return CustomInstruction::MemberAccess;
    return CustomInstruction::Indirection;
  }

  revng_abort();
}

static unsigned getOpcode(Instruction *I) {
  if (isa<CallInst>(I))
    if (isCustomOpcode(I))
      return getCustomOpcode(I);

  return I->getOpcode();
}

struct OperatorPrecedenceResolutionPass : public llvm::FunctionPass {
private:
  const std::array<const InstToOpPrec, 35>
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

  bool runOnFunction(llvm::Function &F) override;

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
  }

public:
  bool needsParentheses(Instruction *I, Use &U);
};

enum OperandSide : unsigned { LHS = 0, RHS };

using OPRP = OperatorPrecedenceResolutionPass;

bool OPRP::needsParentheses(Instruction *I, Use &U) {
  // We need to decide when to emit parentheses (by default, always, most
  // conservative choice), yet, there are some cases, e.g. w/ binary
  // operator instructions, in which we can lean on their precedence &
  // associativity to verify whether we need to emit parentheses or
  // not for sure.
  bool VerifyParentheses = false;

  // Isn't the use an instruction? Will not emit parentheses
  if (!isa<Instruction>(U.get()))
    return false;

  // Verify emission of parentheses for binary operators, load and cast
  // instructions. Always emit parentheses when encountering calls.
  if (isa<BinaryOperator>(I) || isa<CmpInst>(I) || isa<ICmpInst>(I)
      || isa<SelectInst>(I) || isa<LoadInst>(I) || isa<CastInst>(I))
    VerifyParentheses = true;
  else if (isa<AllocaInst>(I) || isa<InsertElementInst>(I)
           || isa<ExtractElementInst>(I) || isa<InsertValueInst>(I)
           || isa<ExtractValueInst>(I) || isa<ShuffleVectorInst>(I)
           || isa<StoreInst>(I) || isa<BranchInst>(I) || isa<CallBrInst>(I)
           || isa<IndirectBrInst>(I) || isa<ReturnInst>(I)
           || isa<IntrinsicInst>(I))
    return false;

  // Does the current instruction represent a custom operator?
  if (isa<CallInst>(I) && isCustomOpcode(I)) {
    switch (getCustomOpcode(I)) {
    case CustomInstruction::AddressOf:
    case CustomInstruction::Indirection:
    case CustomInstruction::MemberAccess:
      VerifyParentheses = (U.getOperandNo() == 1);
      break;
    case CustomInstruction::Assignment:
      return false;
    }
  }

  Instruction *Ins = cast<Instruction>(U.get());

  // Skip parenthesizing the expression when the Use is an `Assignment`
  if (isa<CallInst>(Ins) && isCustomOpcode(Ins)
      && getCustomOpcode(Ins) == CustomInstruction::Assignment)
    return false;

  // No need to emit parentheses when the operand is a custom operator or a
  // general instruction of normal function call. This may be subject to change
  // in the future, up to the addition of new operators.
  if (isa<CallInst>(I) && !isCustomOpcode(I)
      && cast<CallInst>(I)->getCalledFunction() && isa<Instruction>(Ins))
    return false;

  if (VerifyParentheses) {
    auto CurrentOpIt = findOpcode(LLVMOpcodeToLangOpPrecedenceArray,
                                  getOpcode(I));
    revng_assert(CurrentOpIt != nullptr);

    auto OperandOpIt = findOpcode(LLVMOpcodeToLangOpPrecedenceArray,
                                  getOpcode(Ins));

    if (OperandOpIt != LLVMOpcodeToLangOpPrecedenceArray->end()) {
      if (CurrentOpIt->Precedence < OperandOpIt->Precedence) {
        return false;
      } else if (CurrentOpIt->Precedence == OperandOpIt->Precedence) {
        OperandSide Side = U.getOperandNo() == 0 ? LHS : RHS;
        // Binary operator?
        if (I->getNumOperands() != 1 && !isa<CallInst>(I)) {
          if (CurrentOpIt->Associativity == LEFT_TO_RIGHT && Side != RHS)
            return false;
          else if (CurrentOpIt->Associativity == RIGHT_TO_LEFT && Side != LHS)
            return false;
        } else {
          // If the instruction operator is unary, and the associativity is
          // alike, no parentheses are needed.
          return false;
        }
      }
    }
  }

  return true;
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

  if (InstructionsToBeParenthesized.empty())
    return false;

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
