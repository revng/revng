/// \file functioncallidentification.cpp
/// \brief Implementation of the FunctionCallIdentification pass, which
///        identifies function calls.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Local libraries includes
#include "revng/BasicAnalyses/FunctionCallIdentification.h"
#include "revng/Support/Debug.h"

using namespace llvm;

char FunctionCallIdentification::ID = 0;
static RegisterPass<FunctionCallIdentification> X("fci",
                                                  "Function Call "
                                                  "Identification",
                                                  true,
                                                  true);

bool FunctionCallIdentification::runOnFunction(llvm::Function &F) {
  revng_log(PassesLog, "Starting FunctionCallIdentification");

  auto &GCBI = getAnalysis<GeneratedCodeBasicInfo>();

  FallthroughAddresses.clear();

  // Create function call marker
  // TODO: we could factor this out
  Module *M = F.getParent();
  LLVMContext &C = M->getContext();
  PointerType *Int8PtrTy = Type::getInt8PtrTy(C);
  auto *Int8NullPtr = ConstantPointerNull::get(Int8PtrTy);
  auto *PCTy = IntegerType::get(C, GCBI.pcRegSize() * 8);
  auto *PCPtrTy = cast<PointerType>(GCBI.pcReg()->getType());
  std::initializer_list<Type *> FunctionArgsTy = {
    Int8PtrTy, Int8PtrTy, PCTy, PCPtrTy, Int8PtrTy
  };
  using FT = FunctionType;
  auto *Ty = FT::get(Type::getVoidTy(C), FunctionArgsTy, false);
  Constant *FunctionCallC = M->getOrInsertFunction("function_call", Ty);
  FunctionCall = cast<Function>(FunctionCallC);

  // Initialize the function, if necessary
  if (FunctionCall->empty()) {
    FunctionCall->setLinkage(GlobalValue::InternalLinkage);
    auto *EntryBB = BasicBlock::Create(C, "", FunctionCall);
    ReturnInst::Create(C, EntryBB);
    revng_assert(FunctionCall->user_begin() == FunctionCall->user_end());
  }

  // Collect function calls
  for (BasicBlock &BB : F) {
    if (!GCBI.isTranslated(&BB))
      continue;

    // Consider the basic block only if it's terminator is an actual jump and it
    // hasn't been already marked as a function call
    TerminatorInst *Terminator = BB.getTerminator();
    if (BB.empty() or not GCBI.isJump(Terminator) or isCall(Terminator))
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
    Constant *LinkRegister = nullptr;

    // We can meet up calls to newpc up to (1 + "size of the delay slot")
    // times
    unsigned NewPCLeft = 1;

    auto Visitor = [&BB,
                    &GCBI,
                    &NewPCLeft,
                    &SaveRAFound,
                    ReturnPC,
                    &LastPC,
                    &StorePCFound,
                    &LinkRegister,
                    &PCPtrTy](RBasicBlockRange R) {
      for (Instruction &I : R) {
        if (auto *Store = dyn_cast<StoreInst>(&I)) {
          Value *V = Store->getValueOperand();
          Value *Pointer = Store->getPointerOperand();
          auto *TargetCSV = dyn_cast<GlobalVariable>(Pointer);
          if (GCBI.isPCReg(TargetCSV)) {
            if (TargetCSV != nullptr)
              StorePCFound = true;
          } else if (auto *Constant = dyn_cast<ConstantInt>(V)) {
            // Note that we willingly ignore stores to the PC here
            if (Constant->getLimitedValue() == ReturnPC) {
              if (SaveRAFound) {
                SaveRAFound = false;
                return StopNow;
              }
              SaveRAFound = true;

              // Find where the return address is being stored
              revng_assert(LinkRegister == nullptr);
              if (TargetCSV != nullptr) {
                // The return address is being written to a register
                LinkRegister = TargetCSV;
              } else {
                // The return address is likely being written on the stack, we
                // have to check the last value on the stack and check if we're
                // writing there. This should cover basically all the cases,
                // and, if not, expanding this should be straightforward

                // Reference example:
                //
                // %1 = load i64, i64* @rsp
                // %2 = sub i64 %1, 8
                // %3 = inttoptr i64 %2 to i64*
                // store i64 4194694, i64* %3
                // store i64 %2, i64* @rsp
                // store i64 4194704, i64* @pc

                // Find the last write to the stack pointer
                Value *LastStackPointer = nullptr;
                for (Instruction &I : make_range(BB.rbegin(), BB.rend())) {
                  if (auto *S = dyn_cast<StoreInst>(&I)) {
                    auto *P = dyn_cast<GlobalVariable>(S->getPointerOperand());
                    if (P != nullptr && GCBI.isSPReg(P)) {
                      LastStackPointer = Store->getPointerOperand();
                      break;
                    }
                  }
                }
                revng_assert(LastStackPointer != nullptr);
                revng_assert(skipCasts(LastStackPointer) == skipCasts(Pointer));

                // If LinkRegister is nullptr it means the return address is
                // being pushed on the top of the stack
                LinkRegister = ConstantPointerNull::get(PCPtrTy);
              }
            }
          }
        } else if (auto *Call = dyn_cast<CallInst>(&I)) {
          auto *Callee = Call->getCalledFunction();
          if (Callee != nullptr && Callee->getName() == "newpc") {
            revng_assert(NewPCLeft > 0);

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

      // Emit a call to "function_call" with three parameters: the first is the
      // callee basic block, the second the return basic block and the third is
      // the return address

      // If there is a single successor it can be anypc or an actual callee
      // basic block, both cases are fine. If there's more than one successor,
      // we want to register only the default successor of the switch statement
      // (typically anypc).
      // TODO: register in the call to function_call multiple call targets
      unsigned SuccessorsCount = Terminator->getNumSuccessors();
      Value *Callee = nullptr;

      if (SuccessorsCount == 0) {
        Callee = Int8NullPtr;
      } else if (SuccessorsCount == 1) {
        Callee = BlockAddress::get(Terminator->getSuccessor(0));
      } else {
        // If there are multiple successors, at least one should not be a jump
        // target
        bool Found = false;
        for (BasicBlock *Successor : successors(Terminator->getParent())) {
          if (!GCBI.isJumpTarget(Successor)) {
            // There should be only one non-jump target successor (i.e., anypc
            // or unepxectedpc).
            revng_assert(!Found);
            Found = true;
          }
        }
        revng_assert(Found);

        // It's an indirect call
        Callee = Int8NullPtr;
      }

      const std::initializer_list<Value *> Args{ Callee,
                                                 BlockAddress::get(ReturnBB),
                                                 ConstantInt::get(PCTy,
                                                                  ReturnPC),
                                                 LinkRegister,
                                                 Int8NullPtr };

      FallthroughAddresses.insert(ReturnPC);

      // If the instruction before the terminator is a call to exitTB, inject
      // the call to function_call before it, so it doesn't get purged
      auto It = Terminator->getIterator();
      if (It != Terminator->getParent()->begin()) {
        auto PrevIt = It;
        PrevIt--;
        if (isCallTo(&*PrevIt, "exitTB"))
          It = PrevIt;
      }

      CallInst::Create(FunctionCall, Args, "", &*It);
    }
  }

  revng_log(PassesLog, "Ending FunctionCallIdentification");

  return false;
}
