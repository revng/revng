//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/STLExtras.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Transforms/Utils/Local.h"

#include "revng/Model/FunctionTags.h"
#include "revng/Support/Debug.h"
#include "revng/Support/OpaqueFunctionsPool.h"

using namespace llvm;

enum Signedness {
  Signed,
  Unsigned
};

enum Operation {
  Add,
  Sub,
  Mul
};

struct OverflowFunctionKey {
public:
  Signedness Signedness;
  Operation Operation;
  unsigned Bits;

  bool operator<(const OverflowFunctionKey &Other) const {
    auto Tie = [](const OverflowFunctionKey &This) {
      return std::tie(This.Signedness, This.Operation, This.Bits);
    };
    return Tie(*this) < Tie(Other);
  }

  std::string name() const {
    std::string Result;

    switch (Operation) {
    case Add:
      Result += "add";
      break;
    case Sub:
      Result += "sub";
      break;
    case Mul:
      Result += "mul";
      break;
    default:
      revng_abort();
    }

    Result += "_overflow_";

    if (Signedness == Unsigned) {
      Result += "u";
    } else {
      Result += "s";
    }

    Result += std::to_string(Bits);

    return Result;
  }
};

class SplitOverflowIntrinsicsPass : public llvm::FunctionPass {
public:
  static char ID;

  SplitOverflowIntrinsicsPass() : llvm::FunctionPass(ID) {}

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {}

  bool runOnFunction(llvm::Function &F) override {
    bool Changed = false;

    // Collect calls to .with.overflow intrinsics
    llvm::SmallVector<WithOverflowInst *, 4> Calls;
    for (BasicBlock &BB : F)
      for (Instruction &I : BB)
        if (auto *Call = dyn_cast<WithOverflowInst>(&I))
          Calls.push_back(Call);

    // Create the function pool computing whether the operation overflows or not
    OpaqueFunctionsPool<std::string> OverflowPool(F.getParent(), false);
    OverflowPool.addFnAttribute(llvm::Attribute::NoUnwind);
    OverflowPool.addFnAttribute(llvm::Attribute::WillReturn);
    OverflowPool.setMemoryEffects(llvm::MemoryEffects::none());

    {
      using namespace FunctionTags;
      OverflowPool.setTags({ &Helper, &BinaryOperationOverflows });
      OverflowPool.initializeFromName(BinaryOperationOverflows);
    }

    // Here we should definitely use the builder that checks the debug info,
    // but since this going to go away soon, let it stay as is.
    revng::NonDebugInfoCheckingIRBuilder Builder(F.getParent()->getContext());

    for (WithOverflowInst *Call : Calls) {
      Changed = true;

      Builder.SetInsertPoint(Call);

      Value *Operand1 = Call->getArgOperand(0);
      Value *Operand2 = Call->getArgOperand(1);
      Value *Result = nullptr;
      Signedness Signedness = Unsigned;
      Operation Operation;

      switch (Call->getIntrinsicID()) {
      case Intrinsic::uadd_with_overflow:
      case Intrinsic::usub_with_overflow:
      case Intrinsic::umul_with_overflow:
        Signedness = Unsigned;
        break;

      case Intrinsic::sadd_with_overflow:
      case Intrinsic::ssub_with_overflow:
      case Intrinsic::smul_with_overflow:
        Signedness = Signed;
        break;

      default:
        revng_abort("Unexpected intrinsic");
      }

      switch (Call->getIntrinsicID()) {
      case Intrinsic::uadd_with_overflow:
      case Intrinsic::sadd_with_overflow:
        Result = Builder.CreateAdd(Operand1, Operand2);
        Operation = Add;
        break;

      case Intrinsic::usub_with_overflow:
      case Intrinsic::ssub_with_overflow:
        Result = Builder.CreateSub(Operand1, Operand2);
        Operation = Sub;
        break;

      case Intrinsic::umul_with_overflow:
      case Intrinsic::smul_with_overflow:
        Result = Builder.CreateMul(Operand1, Operand2);
        Operation = Mul;
        break;

      default:
        revng_abort("Unexpected intrinsic");
      }

      auto *OperandType = cast<IntegerType>(Operand1->getType());
      revng_assert(Operand2->getType() == OperandType);
      auto *FT = FunctionType::get(Builder.getInt1Ty(),
                                   { OperandType, OperandType },
                                   false);
      OverflowFunctionKey Key = { Signedness,
                                  Operation,
                                  OperandType->getBitWidth() };
      std::string Name = Key.name();
      auto *Overflow = Builder.CreateCall(OverflowPool.get(Name, FT, Name),
                                          { Operand1, Operand2 });

      for (User *U : llvm::make_early_inc_range(Call->users())) {
        revng_assert(isCallToTagged(U, FunctionTags::OpaqueExtractValue));
        auto *ExtractValue = cast<CallInst>(U);
        auto *Index = cast<ConstantInt>(ExtractValue->getArgOperand(1));
        Value *ReplaceWith = nullptr;
        if (Index->isZero()) {
          ReplaceWith = Result;
        } else if (Index->isOne()) {
          ReplaceWith = Overflow;
        } else {
          revng_abort();
        }

        ExtractValue->replaceAllUsesWith(ReplaceWith);
        ExtractValue->eraseFromParent();
      }
    }

    return Changed;
  }
};

char SplitOverflowIntrinsicsPass::ID;

using Register = RegisterPass<SplitOverflowIntrinsicsPass>;
static Register R("split-overflow-intrinsics", "", false, false);
