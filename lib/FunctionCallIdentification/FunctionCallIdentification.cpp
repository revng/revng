/// \file functioncallidentification.cpp
/// \brief Implementation of the FunctionCallIdentification pass, which
///        identifies function calls.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Local libraries includes
#include "revng/FunctionCallIdentification/FunctionCallIdentification.h"
#include "revng/Support/Debug.h"

using namespace llvm;

char FunctionCallIdentification::ID = 0;
static RegisterPass<FunctionCallIdentification> X("fci",
                                                  "Function Call "
                                                  "Identification",
                                                  true,
                                                  true);

static Logger<> FilteredCFGLog("filtered-cfg");

bool FunctionCallIdentification::runOnModule(llvm::Module &M) {
  revng_log(PassesLog, "Starting FunctionCallIdentification");

  llvm::Function &F = *M.getFunction("root");
  auto &GCBI = getAnalysis<GeneratedCodeBasicInfo>();

  FallthroughAddresses.clear();

  // Create function call marker
  // TODO: we could factor this out
  LLVMContext &C = M.getContext();
  PointerType *Int8PtrTy = Type::getInt8PtrTy(C);
  auto *Int8NullPtr = ConstantPointerNull::get(Int8PtrTy);
  auto *PCTy = IntegerType::get(C, GCBI.pcRegSize() * 8);
  auto *PCPtrTy = cast<PointerType>(GCBI.pcReg()->getType());
  std::initializer_list<Type *> FunctionArgsTy = {
    Int8PtrTy, Int8PtrTy, PCTy, PCPtrTy, Int8PtrTy
  };
  using FT = FunctionType;
  auto *Ty = FT::get(Type::getVoidTy(C), FunctionArgsTy, false);
  Constant *FunctionCallC = M.getOrInsertFunction("function_call", Ty);
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

    if (BB.empty() or not GCBI.isTranslated(&BB))
      continue;

    // Consider the basic block only if it's terminator is an actual jump and it
    // hasn't been already marked as a function call
    TerminatorInst *Terminator = BB.getTerminator();

    if (Terminator != nullptr) {
      if (CallInst *Call = getCall(Terminator)) {
        FallthroughAddresses.insert(getLimitedValue(Call->getOperand(2)));
        continue;
      }
    }

    if (not GCBI.isJump(Terminator))
      continue;

    // To be a function call we need to find:
    //
    // * a call to "newpc"
    // * a store of the next PC
    // * a store to the PC
    //
    // TODO: the function call detection criteria in reachingdefinitions.cpp
    //       is probably more elegant, import it.
    struct Visitor
      : public BFSVisitorBase<false, Visitor, SmallVector<BasicBlock *, 4>> {
    public:
      using SuccessorsType = SmallVector<BasicBlock *, 4>;

    public:
      BasicBlock *BB;
      const GeneratedCodeBasicInfo &GCBI;
      bool SaveRAFound;
      bool StorePCFound;
      Constant *LinkRegister;
      const uint64_t ReturnPC;
      uint64_t LastPC;

      // We can meet calls up to newpc up to (1 + "size of the delay slot")
      // times
      uint64_t NewPCLeft;
      PointerType *PCPtrTy;

    public:
      Visitor(BasicBlock *BB,
              const GeneratedCodeBasicInfo &GCBI,
              uint64_t ReturnPC,
              PointerType *PCPtrTy) :
        BB(BB),
        GCBI(GCBI),
        SaveRAFound(false),
        StorePCFound(false),
        LinkRegister(nullptr),
        ReturnPC(ReturnPC),
        LastPC(ReturnPC),
        NewPCLeft(1),
        PCPtrTy(PCPtrTy) {}

    public:
      VisitAction visit(instruction_range Range) {
        for (Instruction &I : Range) {
          if (auto *Store = dyn_cast<StoreInst>(&I)) {
            Value *V = Store->getValueOperand();
            Value *Pointer = skipCasts(Store->getPointerOperand());
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
                  // have to check the last value on the stack and check if
                  // we're writing there. This should cover basically all the
                  // cases, and, if not, expanding this should be
                  // straightforward

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
                  for (Instruction &I : make_range(BB->rbegin(), BB->rend())) {
                    if (auto *S = dyn_cast<StoreInst>(&I)) {
                      Value *Pointer = skipCasts(S->getPointerOperand());
                      auto *P = dyn_cast<GlobalVariable>(Pointer);
                      if (P != nullptr && GCBI.isSPReg(P)) {
                        LastStackPointer = Store->getPointerOperand();
                        break;
                      }
                    }
                  }
                  revng_assert(LastStackPointer != nullptr);
                  revng_assert(skipCasts(LastStackPointer) == Pointer);

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
      }

      SuccessorsType successors(BasicBlock *BB) {
        SuccessorsType Successors;
        for (BasicBlock *Successor : make_range(pred_begin(BB), pred_end(BB)))
          if (not BB->empty() and GCBI.isTranslated(Successor))
            Successors.push_back(Successor);
        return Successors;
      }
    };

    uint64_t ReturnPC = GCBI.getNextPC(Terminator);
    Visitor V(&BB, GCBI, ReturnPC, PCPtrTy);
    V.run(Terminator);

    BasicBlock *ReturnBB = GCBI.getBlockAt(ReturnPC);
    if (V.SaveRAFound and V.StorePCFound and V.NewPCLeft == 0
        and ReturnBB != nullptr) {
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
                                                 V.LinkRegister,
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

  buildFilteredCFG(F);

  revng_log(PassesLog, "Ending FunctionCallIdentification");

  return false;
}

void FunctionCallIdentification::buildFilteredCFG(llvm::Function &F) {
  auto &GCBI = getAnalysis<GeneratedCodeBasicInfo>();

  // We have to create a view on the CFG where:
  //
  // * We only have translate basic blocks
  // * Function call edges proceed towards the falltrough basic block
  for (BasicBlock &BB : F) {

    if (BB.empty() or not GCBI.isTranslated(&BB))
      continue;

    CustomCFGNode *Node = FilteredCFG.getNode(&BB);

    // Is this a function call?
    if (CallInst *Call = GCBI.getFunctionCall(&BB)) {

      Value *SecondArgument = Call->getArgOperand(1);
      auto *Fallthrough = cast<BlockAddress>(SecondArgument)->getBasicBlock();
      Node->addSuccessor(FilteredCFG.getNode(Fallthrough));

    } else {

      unsigned SuccessorsCount = succ_end(&BB) - succ_begin(&BB);
      bool IsReturn = SuccessorsCount > 0;
      bool AllZero = true;
      auto SuccessorsRange = make_range(succ_begin(&BB), succ_end(&BB));
      for (llvm::BasicBlock *Successor : SuccessorsRange) {

        if (Successor->empty())
          continue;

        uint64_t Address = getBasicBlockPC(Successor);
        AllZero = AllZero and (Address == 0);
        IsReturn = IsReturn and (Address == 0 or isFallthrough(Address));
      }
      IsReturn = IsReturn and not AllZero;

      if (not IsReturn) {
        for (BasicBlock *Successor : SuccessorsRange) {

          if (Successor->empty() or not GCBI.isTranslated(Successor))
            continue;

          Node->addSuccessor(FilteredCFG.getNode(Successor));
        }
      }
    }
  }

  FilteredCFG.buildBackLinks();

  if (FilteredCFGLog.isEnabled())
    FilteredCFG.dump(FilteredCFGLog);
}
