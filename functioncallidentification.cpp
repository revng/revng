/// \file functioncallidentification.cpp
/// \brief Implementation of the FunctionCallIdentification pass, which
///        identifies function calls.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Local includes
#include "debug.h"
#include "functioncallidentification.h"

using namespace llvm;

char FunctionCallIdentification::ID = 0;
static RegisterPass<FunctionCallIdentification> X("fci",
                                                  "Function Call "
                                                  "Identification",
                                                  true,
                                                  true);

bool FunctionCallIdentification::runOnFunction(llvm::Function &F) {
  DBG("passes", { dbg << "Starting FunctionCallIdentification\n"; });

  auto &GCBI = getAnalysis<GeneratedCodeBasicInfo>();

  // Create function call marker
  // TODO: we could factor this out
  Module *M = F.getParent();
  LLVMContext &C = M->getContext();
  Type *Int8PtrTy = Type::getInt8PtrTy(C);
  auto *Int32Ty = IntegerType::get(C, 32);
  auto *FunctionCallFT = FunctionType::get(Type::getVoidTy(C),
                                           { Int8PtrTy, Int8PtrTy, Int32Ty },
                                           false);
  FunctionCall = cast<Function>(M->getOrInsertFunction("function_call",
                                                       FunctionCallFT));

  // Initialize the function, if necessary
  if (FunctionCall->empty()) {
    FunctionCall->setLinkage(GlobalValue::InternalLinkage);
    auto *EntryBB = BasicBlock::Create(C, "", FunctionCall);
    ReturnInst::Create(C, EntryBB);
    assert(FunctionCall->user_begin() == FunctionCall->user_end());
  }

  // Collect function calls
  for (BasicBlock &BB : F) {

    // Consider the basic block only if it's terminator is an actual jump, it's
    // not an unreachable instruction and it hasn't been already marked as a
    // function call
    TerminatorInst *Terminator = BB.getTerminator();
    if (!GCBI.isJump(Terminator)
        || isa<UnreachableInst>(Terminator)
        || isCall(Terminator))
      continue;

    // To be a function call we need to find:
    //
    // * a call to "newpc"
    // * a store of the next PC
    // * a store to the PC
    //
    // TODO: the function call detection criteria in reachingdefinitions.cpp
    //       is probably more elegant, import it.
    bool SaveRAFound = false;
    bool StorePCFound = false;
    uint64_t ReturnPC = GCBI.getNextPC(Terminator);
    uint64_t LastPC = ReturnPC;

    // We can meet up calls to newpc up to (1 + "size of the delay slot")
    // times
    unsigned NewPCLeft = 1 + GCBI.delaySlotSize();

    auto Visitor = [&GCBI,
                    &NewPCLeft,
                    &SaveRAFound,
                    ReturnPC,
                    &LastPC,
                    &StorePCFound] (RBasicBlockRange R) {
      for (Instruction &I : R) {
        if (auto *Store = dyn_cast<StoreInst>(&I)) {
          Value *V = Store->getValueOperand();
          auto *D = dyn_cast<GlobalVariable>(Store->getPointerOperand());
          if (GCBI.isPCReg(D)) {
            StorePCFound = true;
          } else if (auto *Constant = dyn_cast<ConstantInt>(V)) {
            // Note that we willingly ignore stores to the PC here
            if (Constant->getLimitedValue() == ReturnPC) {
              assert(!SaveRAFound);
              SaveRAFound = true;
            }
          }
        } else if (auto *Call = dyn_cast<CallInst>(&I)) {
          auto *Callee = Call->getCalledFunction();
          if (Callee != nullptr && Callee->getName() == "newpc") {
            assert(NewPCLeft > 0);

            uint64_t ProgramCounter = getLimitedValue(Call->getOperand(0));
            uint64_t InstructionSize = getLimitedValue(Call->getOperand(1));

            // Check that, w.r.t. to the last newpc, we're looking at the
            // immediately preceeding instruction, if not fail.
            if (ProgramCounter + InstructionSize != LastPC)
              return StopNow;

            // Update the last seen PC
            LastPC = ProgramCounter;

            NewPCLeft--;
            if (NewPCLeft == 0)
              return StopNow;
          }
        }
      }

      return Continue;
    };

    // TODO: adapt visitPredecessors from visitSuccessors
    GCBI.visitPredecessors(Terminator, Visitor);

    BasicBlock *ReturnBB = GCBI.getBlockAt(ReturnPC);
    if (SaveRAFound && StorePCFound && NewPCLeft == 0 && ReturnBB != nullptr) {
      // It's a function call, register it

      // Emit a call to "function_call" with two parameters: the first is the
      // callee basic block, the second the return basic block
      unsigned SuccessorCount = 0;
      for (BasicBlock *Successor : successors(Terminator->getParent()))
        if (GCBI.isJumpTarget(Successor))
          SuccessorCount++;

      assert(SuccessorCount <= 1 && "Multiple successors are not supported");
      Value *Args[3] = {
        BlockAddress::get(Terminator->getSuccessor(0)),
        BlockAddress::get(ReturnBB),
        ConstantInt::get(Int32Ty, ReturnPC)
      };
      CallInst::Create(FunctionCall, Args, "", Terminator);
    }
  }

  DBG("passes", { dbg << "Ending FunctionCallIdentification\n"; });

  return false;
}
