/// \file PostLiftVerifyPass.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ModuleSlotTracker.h"

#include "revng/Support/Assert.h"

#include "PostLiftVerifyPass.h"

using namespace llvm;

bool PostLiftVerifyPass::runOnModule(Module &M) {
  Function *RootFunction = M.getFunction("root");
  revng_assert(RootFunction != nullptr);

  llvm::ModuleSlotTracker MST(&M, false);

  for (BasicBlock &BB : *RootFunction) {
    // Ignore some special basic blocks
    if (BB.getName() == "dispatcher.default"
        or BB.getName() == "serialize_and_jump_out"
        or BB.getName() == "return_from_external" or BB.getName() == "setjmp"
        or BB.getName() == "dispatcher.external")
      continue;

    for (Instruction &I : BB) {
      bool Good = false;

      switch (I.getOpcode()) {
      case Instruction::Store:
      case Instruction::Load:
        Good = true;
        break;

      case Instruction::IntToPtr:
      case Instruction::Add:
      case Instruction::Sub:
      case Instruction::Mul:
      case Instruction::UDiv:
      case Instruction::SDiv:
      case Instruction::URem:
      case Instruction::SRem:
      case Instruction::And:
      case Instruction::Or:
      case Instruction::Xor:
      case Instruction::ZExt:
      case Instruction::Trunc:
      case Instruction::SExt:
      case Instruction::ICmp:
      case Instruction::LShr:
      case Instruction::AShr:
      case Instruction::Shl:
      case Instruction::Select:
        Good = true;
        break;

      case Instruction::Br:
      case Instruction::Switch:
      case Instruction::Unreachable:
        Good = true;
        break;

      case Instruction::PHI:
        Good = true;
        break;

      case Instruction::ExtractValue:
        Good = true;
        break;

      case Instruction::Call:
        // Make further checks
        auto *Call = cast<CallInst>(&I);
        Value *CalledOperand = Call->getCalledOperand();
        Function *Callee = dyn_cast_or_null<Function>(CalledOperand);
        StringRef CalleeName;
        if (Callee != nullptr)
          CalleeName = Callee->getName();

        Good = (CalleeName == "newpc" or CalleeName == "jump_to_symbol"
                or CalleeName.startswith("helper_")
                or CalleeName == "function_call"
                or CalleeName == "helper_initialize_env"
                or CalleeName == "revng_abort");

        switch (Callee->getIntrinsicID()) {
        case Intrinsic::fshl:
        case Intrinsic::fshr:
        case Intrinsic::bswap:
        case Intrinsic::abs:
        case Intrinsic::umin:
        case Intrinsic::umax:
        case Intrinsic::smin:
        case Intrinsic::smax:
        case Intrinsic::ctlz:
        case Intrinsic::cttz:
          Good = true;
        }

        break;
      }

      if (not Good) {
        std::string Buffer;
        {
          llvm::raw_string_ostream Stream(Buffer);
          Stream << "Unexpected instruction: ";
          I.print(Stream, MST);
          Stream << "\n";
        }
        revng_abort(Buffer.c_str());
      }
    }
  }

  return false;
}

char PostLiftVerifyPass::ID;
