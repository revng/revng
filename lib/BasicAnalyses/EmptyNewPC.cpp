/// \file EmptyNewPC.cpp
/// A simple pass to given an empty body to the `newpc` function so that it can
/// be optimized away.

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"

#include "revng/BasicAnalyses/EmptyNewPC.h"
#include "revng/Support/IRHelpers.h"

using namespace llvm;

char EmptyNewPC::ID = 0;
using Register = RegisterPass<EmptyNewPC>;
static Register X("empty-newpc", "Create an empty newpc function", true, true);

bool EmptyNewPC::runOnModule(llvm::Module &M) {
  LLVMContext &Context = getContext(&M);
  Function *NewPCFunction = M.getFunction("newpc");
  NewPCFunction->deleteBody();
  ReturnInst::Create(Context, BasicBlock::Create(Context, "", NewPCFunction));
  return false;
}
