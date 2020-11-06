/// \file ReachabilityPass.cpp
/// \brief Tests for ReachabilityPass

//
// Copyright (c) rev.ng Srls 2017-2020.
//

// Boost includes
#define BOOST_TEST_MODULE ReachabilityPass
bool init_unit_test();
#include <boost/test/unit_test.hpp>

// LLVM includes
#include "llvm/IR/Dominators.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/SourceMgr.h"

// revng includes
#include "revng/Support/Debug.h"

// Local libraries includes
#include "Reachability/ReachabilityPass.h"

using namespace llvm;

static const char *ModuleBegin = R"LLVM(
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

@rax = internal global i64 0
@rdi = internal global i64 0
@rsi = internal global i64 0
@rbx = internal global i64 0
@rcx = internal global i64 0

define void @main() {
)LLVM";

static const char *ModuleEnd = "\n}\n";

static std::string buildModule(const char *Body) {
  std::string Result;
  Result += ModuleBegin;
  Result += Body;
  Result += ModuleEnd;
  return Result;
}

static BasicBlock *basicBlockByName(Function *F, const char *Name) {
  revng_assert(F != nullptr);

  for (BasicBlock &BB : *F)
    if (BB.hasName() and BB.getName() == Name)
      return &BB;

  revng_abort("Couldn't find a Value with the requested name");
}

static std::unique_ptr<Module> loadModule(LLVMContext &C, const char *Body) {
  std::string ModuleText = buildModule(Body);
  SMDiagnostic Diagnostic;
  using MB = MemoryBuffer;
  std::unique_ptr<MB> Buffer = MB::getMemBuffer(StringRef(ModuleText));
  std::unique_ptr<Module> M = parseIR(Buffer.get()->getMemBufferRef(),
                                      Diagnostic,
                                      C);

  if (M.get() == nullptr) {
    Diagnostic.print("revamb", dbgs());
    revng_abort();
  }

  return M;
}

class Test {
public:
  enum Type { Regular, Conditional, Both };

private:
  LLVMContext Context;

public:
  Test() {}

  void
  test(const char *Body,
       std::vector<std::pair<const char *, std::vector<const char *>>> Checks) {

    std::unique_ptr<Module> M = loadModule(Context, Body);
    Function *F = M->getFunction("main");

    ReachabilityPass *Reachability = new ReachabilityPass();
    Reachability->runOnFunction(*F);

    for (auto &P : Checks) {
      BasicBlock *SourceBB = basicBlockByName(F, P.first);

      for (const char *TargetBlockName : P.second) {
        BasicBlock *TargetBB = basicBlockByName(F, TargetBlockName);
        revng_assert(Reachability->existsPath(SourceBB, TargetBB));
      }
    }
  }
};

BOOST_AUTO_TEST_CASE(SelfReachable) {
  Test X;

  //
  // Check that a block is reachable from themselves.
  //
  const char *Body = R"LLVM(
  bb.main:
  ret void

  bb.main2:
  ret void
)LLVM";

  X.test(Body,
         { { "bb.main", { "bb.main" } }, { "bb.main2", { "bb.main2" } } });
}

BOOST_AUTO_TEST_CASE(SimpleSuccessor) {
  Test X;

  //
  // Check that a block with a single successor can reach it.
  //
  const char *Body = R"LLVM(
  bb.main:
  br label %bb.main2

  bb.main2:
  ret void
)LLVM";

  X.test(Body, { { "bb.main", { "bb.main2" } } });
}

BOOST_AUTO_TEST_CASE(SimpleCycle) {
  Test X;

  //
  // Check that we can follow backward jumps.
  //
  const char *Body = R"LLVM(
  bb.main:
  br label %bb.main2

  bb.main2:
  br label %bb.main
)LLVM";

  X.test(Body, { { "bb.main2", { "bb.main" } } });
}

BOOST_AUTO_TEST_CASE(SimpleConditional) {
  Test X;

  //
  // Check that we can reach multiple successors.
  //
  const char *Body = R"LLVM(
  bb.main:
  br i1 0, label %bb.main2, label %bb.main3

  bb.main2:
  ret void

  bb.main3:
  ret void
)LLVM";

  X.test(Body,
         { { "bb.main", { "bb.main2" } }, { "bb.main", { "bb.main3" } } });
}

BOOST_AUTO_TEST_CASE(CyclicConditional) {
  Test X;

  //
  // Check that with a conditional jump and a backward jump we are able to
  // reach the successors.
  //
  const char *Body = R"LLVM(
  bb.main:
  br i1 0, label %bb.main2, label %bb.main3

  bb.main2:
  br label %bb.main

  bb.main3:
  ret void
)LLVM";

  X.test(Body,
         { { "bb.main", { "bb.main2" } },
           { "bb.main", { "bb.main3" } },
           { "bb.main2", { "bb.main3" } } });
}
