/// \file CPULoopExitPass.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"

#include "revng/HelperArgumentsAnalysis/CPULoopExitPass.h"
#include "revng/Support/Debug.h"
#include "revng/Support/IRHelpers.h"

using namespace llvm;

char CPULoopExitPass::ID = 0;

static void purgeNoReturn(Function *F) {
  auto &Context = F->getParent()->getContext();

  if (F->hasFnAttribute(Attribute::NoReturn))
    F->removeFnAttr(Attribute::NoReturn);

  for (User *U : F->users()) {
    if (auto *Call = dyn_cast<CallInst>(U)) {
      if (Call->hasFnAttr(Attribute::NoReturn)) {
        auto OldAttr = Call->getAttributes();
        auto NewAttr = OldAttr.removeFnAttribute(Context, Attribute::NoReturn);
        Call->setAttributes(NewAttr);
      }
    }
  }
}

static ReturnInst *createRet(Instruction *Position) {
  Function *F = Position->getParent()->getParent();
  purgeNoReturn(F);

  Type *ReturnType = F->getFunctionType()->getReturnType();
  if (ReturnType->isVoidTy()) {
    return ReturnInst::Create(F->getParent()->getContext(), nullptr, Position);
  } else if (ReturnType->isIntegerTy() or ReturnType->isPointerTy()) {
    auto *Null = Constant::getNullValue(ReturnType);
    return ReturnInst::Create(F->getParent()->getContext(), Null, Position);
  } else if (ReturnType->isStructTy()) {
    auto *StructTy = cast<StructType>(ReturnType);
    auto *Null = ConstantAggregateZero::get(StructTy);
    return ReturnInst::Create(F->getParent()->getContext(), Null, Position);
  }

  revng_abort("Return type not supported");
}

/// Find all calls to cpu_loop_exit and replace them with:
///
/// * call invoke_handle_exception
/// * set cpu_loop_exiting = true
/// * return
///
/// Then look for all the callers of the function calling cpu_loop_exit and make
/// them check whether they should return immediately (cpu_loop_exiting == true)
/// or not.
/// Then when we reach the root function, set cpu_loop_exiting to false after
/// the call.
bool CPULoopExitPass::runOnModule(llvm::Module &M) {
  LLVMContext &Context = M.getContext();

  // Replace uses of cpu_loop_exit_restore with cpu_loop_exit, some targets
  // e.g. mips only use cpu_loop_exit_restore
  Function *CpuLoopExitRestore = M.getFunction("cpu_loop_exit_restore");
  if (CpuLoopExitRestore != nullptr) {
    Type *FirstArgumentType = CpuLoopExitRestore->getArg(0)->getType();
    Type *Void = Type::getVoidTy(Context);
    auto CpuLoopExitCallee = M.getOrInsertFunction("cpu_loop_exit",
                                                   Void,
                                                   FirstArgumentType);
    auto *CpuLoopExit = cast<Function>(CpuLoopExitCallee.getCallee());
    SmallVector<Value *, 8> ToErase;
    for (User *U : CpuLoopExitRestore->users()) {
      auto *Call = cast<CallInst>(U);
      Value *CpuStateArg = Call->getArgOperand(0);
      IRBuilder<> Builder(Call);
      Value *NewCall = Builder.CreateCall(CpuLoopExit, { CpuStateArg });
      Call->replaceAllUsesWith(NewCall);
      ToErase.push_back(Call);
    }

    for (auto &V : ToErase) {
      eraseFromParent(V);
    }
  }

  Function *CpuLoopExit = M.getFunction("cpu_loop_exit");
  // Nothing to do here
  if (CpuLoopExit == nullptr)
    return false;

  purgeNoReturn(CpuLoopExit);

  IntegerType *BoolType = Type::getInt1Ty(Context);
  std::set<Function *> FixedCallers;
  GlobalVariable *CpuLoopExitingVariable = nullptr;
  CpuLoopExitingVariable = new GlobalVariable(M,
                                              BoolType,
                                              false,
                                              GlobalValue::CommonLinkage,
                                              ConstantInt::getFalse(BoolType),
                                              StringRef("cpu_loop_exiting"));

  Function *InvokeHandleException = M.getFunction("invoke_handle_exception");
  revng_assert(InvokeHandleException != nullptr);

  for (User *U : to_vector(CpuLoopExit->users())) {
    auto *Call = cast<CallInst>(U);
    revng_assert(Call->getCalledFunction() == CpuLoopExit);
    Function *Caller = Call->getParent()->getParent();

    // Call handle_exception
    auto *CallCpuLoop = CallInst::Create(InvokeHandleException,
                                         { Call->getArgOperand(0) },
                                         "",
                                         Call);

    // In recent versions of LLVM you can no longer inject a CallInst in a
    // Function with debug location if the call itself has not a debug location
    // as well, otherwise module verification will fail
    CallCpuLoop->setDebugLoc(Call->getDebugLoc());

    // Set cpu_loop_exiting to true
    new StoreInst(ConstantInt::getTrue(BoolType), CpuLoopExitingVariable, Call);

    // Return immediately
    createRet(Call);
    auto *Unreach = cast<UnreachableInst>(&*(++Call->getIterator()));
    eraseFromParent(Unreach);

    // Remove the call to cpu_loop_exit
    eraseFromParent(Call);

    if (FixedCallers.contains(Caller))
      continue;
    FixedCallers.insert(Caller);

    std::queue<Value *> WorkList;
    WorkList.push(Caller);

    while (!WorkList.empty()) {
      Value *V = WorkList.front();
      WorkList.pop();

      if (auto *F = dyn_cast<Function>(V))
        F->setMetadata("revng.cpu_loop_exits", MDTuple::get(Context, {}));

      for (User *User : V->users()) {
        auto *Call = dyn_cast<CallInst>(User);
        if (Call == nullptr) {
          if (auto *Cast = dyn_cast<ConstantExpr>(User)) {
            revng_assert(Cast->getOperand(0) == V && Cast->isCast());
            WorkList.push(Cast);
            continue;
          } else if (isa<Constant>(User)) {
            continue;
          } else if (auto *Store = dyn_cast<StoreInst>(User)) {
            // We're leaking a pointer to a function that we're instrumenting,
            // fail at run-time.
            CallInst::Create(M.getFunction("abort"), "", Store);
            continue;
          } else {
            revng_abort("Unexpected user");
          }
        }

        Function *RecCaller = Call->getParent()->getParent();

        // TODO: make this more reliable than using function name
        // If the caller is a QEMU helper function make it check
        // cpu_loop_exiting and if it's true, make it return

        // Split BB
        BasicBlock *OldBB = Call->getParent();
        BasicBlock::iterator SplitPoint = ++Call->getIterator();
        revng_assert(SplitPoint != OldBB->end());
        BasicBlock *NewBB = OldBB->splitBasicBlock(SplitPoint);

        // Add a BB with a ret
        BasicBlock *QuitBB = BasicBlock::Create(Context,
                                                "cpu_loop_exit_return",
                                                RecCaller,
                                                NewBB);
        UnreachableInst *Temp = new UnreachableInst(Context, QuitBB);
        createRet(Temp);
        eraseFromParent(Temp);

        // Check value of cpu_loop_exiting
        auto *Branch = cast<BranchInst>(&*++(Call->getIterator()));
        auto *PointeeTy = CpuLoopExitingVariable->getValueType();
        auto *Compare = new ICmpInst(Branch,
                                     CmpInst::ICMP_EQ,
                                     new LoadInst(PointeeTy,
                                                  CpuLoopExitingVariable,
                                                  "",
                                                  Branch),
                                     ConstantInt::getTrue(BoolType));

        BranchInst::Create(QuitBB, NewBB, Compare, Branch);
        eraseFromParent(Branch);

        // Add to the work list only if it hasn't been fixed already
        if (!FixedCallers.contains(RecCaller)) {
          FixedCallers.insert(RecCaller);
          WorkList.push(RecCaller);
        }
      }
    }
  }

  return true;
}

using RegisterCLE = RegisterPass<CPULoopExitPass>;
static RegisterCLE Z("cpu-loop-exit", "CPULoopExit Pass", false, false);
