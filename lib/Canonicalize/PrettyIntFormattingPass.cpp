//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <array>
#include <optional>
#include <utility>
#include <vector>

#include "llvm/IR/Attributes.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/Pass.h"

#include "revng/ABI/ModelHelpers.h"
#include "revng/Model/Binary.h"
#include "revng/Model/IRHelpers.h"
#include "revng/Model/LoadModelPass.h"
#include "revng/Support/FunctionTags.h"
#include "revng/Support/IRBuilder.h"
#include "revng/Support/IRHelpers.h"
#include "revng/Support/OpaqueFunctionsPool.h"

enum class IntFormatting : uint32_t {
  NONE, // no formatting
  HEX,
  CHAR,
  BOOL,
  NULLPTR
};

struct FormatInt {
  IntFormatting Formatting;
  llvm::Use *Use;
};

static std::optional<FormatInt>
getIntFormat(llvm::Instruction &I, llvm::Use &U, const model::Binary &Model);

struct PrettyIntFormatting : public llvm::FunctionPass {
public:
  static char ID;

  PrettyIntFormatting() : llvm::FunctionPass(ID) {}

  bool runOnFunction(llvm::Function &F) override;

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<LoadModelWrapperPass>();
  }
};

bool PrettyIntFormatting::runOnFunction(llvm::Function &F) {

  if (not FunctionTags::TagsSet::from(&F).contains(FunctionTags::Isolated))
    return false;

  const model::Binary
    &Model = *getAnalysis<LoadModelWrapperPass>().get().getReadOnlyModel();

  auto &M = *F.getParent();
  auto HexIntegerPool = FunctionTags::HexInteger.getPool(M);
  auto CharIntegerPool = FunctionTags::CharInteger.getPool(M);
  auto BoolIntegerPool = FunctionTags::BoolInteger.getPool(M);
  auto NullPtrPool = FunctionTags::NullPtr.getPool(M);

  std::vector<FormatInt> IntsToBeFormatted;

  for (llvm::Instruction &I : llvm::instructions(F)) {
    for (llvm::Use &U : I.operands()) {
      if (auto formatting = getIntFormat(I, U, Model); formatting) {
        IntsToBeFormatted.push_back(*formatting);
      }
    }
  }

  revng::IRBuilder Builder(F.getContext());
  for (const auto &[Format, Operand] : IntsToBeFormatted) {
    auto *Val = llvm::cast<llvm::ConstantInt>(Operand->get());
    llvm::Type *IntType = Val->getType();

    auto PrettyFunction = [&, Format = Format]() -> llvm::Function * {
      switch (Format) {
      case IntFormatting::HEX:
        return HexIntegerPool.get(IntType, IntType, { IntType }, "print_hex");
      case IntFormatting::CHAR:
        return CharIntegerPool.get(IntType, IntType, { IntType }, "print_char");
      case IntFormatting::BOOL:
        return BoolIntegerPool.get(IntType, IntType, { IntType }, "print_bool");
      case IntFormatting::NULLPTR:
        return NullPtrPool.get(IntType, IntType, { IntType }, "print_nullptr");
      case IntFormatting::NONE:
      default:
        return nullptr;
      }
      return nullptr;
    }();

    if (PrettyFunction) {
      Builder.SetInsertPoint(llvm::cast<llvm::Instruction>(Operand->getUser()));
      llvm::Value *Call = Builder.CreateCall(PrettyFunction, { Val });
      Operand->set(Call);
    }
  }

  return true;
}

std::optional<FormatInt>
getIntFormat(llvm::Instruction &I, llvm::Use &U, const model::Binary &Model) {
  auto &Context = I.getContext();

  // We cannot print properly characters when they are part of switch
  // instruction, because cases in LLVM switch cannot have variables inside.
  if (I.getOpcode() == llvm::Instruction::Switch) {
    return std::nullopt;
  }

  // Some intrinsic calls require ConstantInt as an argument so we are not able
  // to pass there any decorated value.
  if (auto *Intrinsic = llvm::dyn_cast<llvm::IntrinsicInst>(&I)) {
    if (Intrinsic->getIntrinsicID() == llvm::Intrinsic::abs) {
      return std::nullopt;
    }
  }

  // We want to print ints in hex format when they are left operand of shifts or
  // operands of and/or/xor instructions. Look through model implicit casts as
  // well, since they are noops.
  using llvm::ConstantInt;
  llvm::Use *UseToPrettyPrint = &U;
  // Handle multiple layers of calls to implicit ModelCast.
  llvm::Value *ValueToCheckIfModelCast = U.get();
  while (auto *ModelCast = getCallToTagged(ValueToCheckIfModelCast,
                                           FunctionTags::ModelCast)) {
    // Check if it is an implicit cast.
    if (cast<llvm::ConstantInt>(ModelCast->getArgOperand(2))->isOne()) {
      llvm::Value *ValuteToCast = ModelCast->getArgOperand(1);
      if (llvm::isa<ConstantInt>(ValuteToCast)) {
        llvm::ConstantInt
          *IntConstantToCast = cast<llvm::ConstantInt>(ValuteToCast);
        if (not IntConstantToCast->isZero())
          UseToPrettyPrint = &ModelCast->getOperandUse(1);
        break;
      } else {
        ValueToCheckIfModelCast = ModelCast->getArgOperand(1);
      }
    }
    break;
  }

  llvm::ConstantInt
    *IntConstant = dyn_cast<llvm::ConstantInt>(UseToPrettyPrint);
  if (not IntConstant)
    return std::nullopt;

  if (I.getOpcode() == llvm::Instruction::Shl
      || I.getOpcode() == llvm::Instruction::AShr
      || I.getOpcode() == llvm::Instruction::LShr) {
    if (U.getOperandNo() == 0) {
      return FormatInt{ IntFormatting::HEX, UseToPrettyPrint };
    }
  } else if (I.getOpcode() == llvm::Instruction::And
             || I.getOpcode() == llvm::Instruction::Or
             || I.getOpcode() == llvm::Instruction::Xor) {
    return FormatInt{ IntFormatting::HEX, UseToPrettyPrint };
  } else if (auto *Call = getCallToTagged(&I, FunctionTags::ModelCast)) {
    if (Call->getArgOperandNo(&U) == 1) {
      if (IntConstant->isZero()) {
        // If it's a ModelCast casting a zero constant to a pointer, then we
        // decorate the constant so that it's printed as NULL.
        auto Type = fromLLVMString(Call->getArgOperand(0), Model);
        if (Type->isPointer())
          return FormatInt{ IntFormatting::NULLPTR, &U };
      }
    }
    return std::nullopt;
  }

  using llvm::IntegerType;
  if (UseToPrettyPrint->get()->getType() == IntegerType::getInt8Ty(Context)) {
    return FormatInt{ IntFormatting::CHAR, UseToPrettyPrint };
  }

  if (UseToPrettyPrint->get()->getType() == IntegerType::getInt1Ty(Context)) {
    return FormatInt{ IntFormatting::BOOL, UseToPrettyPrint };
  }

  return std::nullopt;
}

char PrettyIntFormatting::ID = 0;

llvm::RegisterPass<PrettyIntFormatting> X("pretty-int-formatting",
                                          "Wraps integers with decorator "
                                          "functions which informs backend "
                                          "about literal type that should be "
                                          "used",
                                          false,
                                          false);
