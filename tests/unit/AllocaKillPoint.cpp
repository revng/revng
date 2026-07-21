//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#define BOOST_TEST_MODULE AllocaKillPoint
bool init_unit_test();
#include "boost/test/unit_test.hpp"

#include "llvm/Analysis/PostDominators.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"

#include "revng/Canonicalize/AllocaKillPoint.h"
#include "revng/UnitTestHelpers/LLVMTestHelpers.h"
#include "revng/UnitTestHelpers/UnitTestHelpers.h"

using namespace llvm;

static AllocaInst *allocaByName(Function *F, const char *Name) {
  return cast<AllocaInst>(instructionByName(F, Name));
}

// Straight line: the value loaded from %local is consumed in %consume, one
// block past the load's own block %use. The kill point is therefore %consume -
// later than %use, which the alloca's direct uses alone would have yielded -
// and only blocks strictly downstream of %consume are past it.
BOOST_AUTO_TEST_CASE(StraightLine) {
  const char *Body = R"LLVM(
  %local = alloca i64
  store i64 5, ptr %local
  br label %use
use:
  %v = load i64, ptr %local
  br label %consume
consume:
  %c = add i64 %v, 1
  br label %after
after:
  %past = add i64 0, 0
  ret void
)LLVM";
  LLVMContext C;
  std::unique_ptr<Module> M = loadModule(C, Body);
  Function *F = M->getFunction("main");
  PostDominatorTree PDT(*F);
  AllocaInst *Local = allocaByName(F, "local");

  const BasicBlock *PostDom = commonPostDominatorOfTransitiveUses(PDT, Local);
  BOOST_CHECK(PostDom == basicBlockByName(F, "consume"));

  // Up to and including the consumer we are not past the kill point.
  BOOST_CHECK(not isPastCommonPostDominator(PDT,
                                            PostDom,
                                            instructionByName(F, "v")));
  BOOST_CHECK(not isPastCommonPostDominator(PDT,
                                            PostDom,
                                            instructionByName(F, "c")));
  // Strictly downstream of the consumer we are past it.
  BOOST_CHECK(isPastCommonPostDominator(PDT,
                                        PostDom,
                                        instructionByName(F, "past")));
}

// Diamond: the value loaded on the %then branch flows through a PHI in %merge
// and is consumed in %tail. The kill point is %tail - past the join, and later
// than the direct uses (which would stop at %merge). The sibling branch and the
// merge are not past it; only the block after the consumer is.
BOOST_AUTO_TEST_CASE(Diamond) {
  const char *Body = R"LLVM(
  %local = alloca i64
  store i64 5, ptr %local
  %cv = load i64, ptr @rdi
  %cond = icmp eq i64 %cv, 0
  br i1 %cond, label %then, label %else
then:
  %v = load i64, ptr %local
  br label %merge
else:
  %elsev = add i64 0, 0
  br label %merge
merge:
  %phi = phi i64 [ %v, %then ], [ 0, %else ]
  br label %tail
tail:
  %use = add i64 %phi, 1
  br label %end
end:
  %past = add i64 0, 0
  ret void
)LLVM";
  LLVMContext C;
  std::unique_ptr<Module> M = loadModule(C, Body);
  Function *F = M->getFunction("main");
  PostDominatorTree PDT(*F);
  AllocaInst *Local = allocaByName(F, "local");

  const BasicBlock *PostDom = commonPostDominatorOfTransitiveUses(PDT, Local);
  BOOST_CHECK(PostDom == basicBlockByName(F, "tail"));

  // The reading branch, the sibling branch, the merge and the consumer are all
  // before or at the kill point.
  BOOST_CHECK(not isPastCommonPostDominator(PDT,
                                            PostDom,
                                            instructionByName(F, "v")));
  BOOST_CHECK(not isPastCommonPostDominator(PDT,
                                            PostDom,
                                            instructionByName(F, "elsev")));
  BOOST_CHECK(not isPastCommonPostDominator(PDT,
                                            PostDom,
                                            instructionByName(F, "phi")));
  BOOST_CHECK(not isPastCommonPostDominator(PDT,
                                            PostDom,
                                            instructionByName(F, "use")));
  // Past the consumer we are past the kill point.
  BOOST_CHECK(isPastCommonPostDominator(PDT,
                                        PostDom,
                                        instructionByName(F, "past")));
}

// Loop: %v is loaded in the body and accumulated into %acc around a PHI cycle,
// and is never used after the loop. The kill point is the loop header (the
// common post-dominator of the body and the entry), so nothing inside the loop
// is pruned - the block-level test is loop-safe - and only the post-loop block
// is past it. The PHI cycle also exercises the closure's visited set.
BOOST_AUTO_TEST_CASE(LoopBody) {
  const char *Body = R"LLVM(
  %local = alloca i64
  store i64 5, ptr %local
  br label %header
header:
  %acc = phi i64 [ 0, %initial_block ], [ %accnext, %body ]
  %i = phi i64 [ 0, %initial_block ], [ %inext, %body ]
  %cond = icmp slt i64 %i, 10
  br i1 %cond, label %body, label %exitb
body:
  %v = load i64, ptr %local
  %accnext = add i64 %acc, %v
  %inext = add i64 %i, 1
  br label %header
exitb:
  %past = add i64 0, 0
  ret void
)LLVM";
  LLVMContext C;
  std::unique_ptr<Module> M = loadModule(C, Body);
  Function *F = M->getFunction("main");
  PostDominatorTree PDT(*F);
  AllocaInst *Local = allocaByName(F, "local");

  const BasicBlock *PostDom = commonPostDominatorOfTransitiveUses(PDT, Local);
  BOOST_CHECK(PostDom == basicBlockByName(F, "header"));

  // The in-loop load and its accumulation must not be pruned.
  BOOST_CHECK(not isPastCommonPostDominator(PDT,
                                            PostDom,
                                            instructionByName(F, "v")));
  BOOST_CHECK(not isPastCommonPostDominator(PDT,
                                            PostDom,
                                            instructionByName(F, "accnext")));
  // Nor may we prune at the loop header (the kill point itself), which runs
  // before the body's load on every iteration.
  BOOST_CHECK(not isPastCommonPostDominator(PDT,
                                            PostDom,
                                            instructionByName(F, "acc")));
  // After the loop we are past it.
  BOOST_CHECK(isPastCommonPostDominator(PDT,
                                        PostDom,
                                        instructionByName(F, "past")));
}

// Multiple exits: the loaded value's uses have no common post-dominator (the
// function returns on two separate paths), so there is no kill point and
// nothing is ever pruned.
BOOST_AUTO_TEST_CASE(MultipleExits) {
  const char *Body = R"LLVM(
  %local = alloca i64
  store i64 5, ptr %local
  %cv = load i64, ptr @rdi
  %cond = icmp eq i64 %cv, 0
  br i1 %cond, label %left, label %right
left:
  %v = load i64, ptr %local
  %luse = add i64 %v, 1
  ret void
right:
  ret void
)LLVM";
  LLVMContext C;
  std::unique_ptr<Module> M = loadModule(C, Body);
  Function *F = M->getFunction("main");
  PostDominatorTree PDT(*F);
  AllocaInst *Local = allocaByName(F, "local");

  const BasicBlock *PostDom = commonPostDominatorOfTransitiveUses(PDT, Local);
  BOOST_CHECK(PostDom == nullptr);

  // With no common post-dominator nothing is ever past it.
  BOOST_CHECK(not isPastCommonPostDominator(PDT,
                                            PostDom,
                                            instructionByName(F, "v")));
  BOOST_CHECK(not isPastCommonPostDominator(PDT,
                                            PostDom,
                                            instructionByName(F, "luse")));
}
