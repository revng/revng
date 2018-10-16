/// \file ReachingDefinitionsPass.cpp
/// \brief Tests for ReachingDefinitionsPass

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Boost includes
#define BOOST_TEST_MODULE ReachingDefinitionsPass
bool init_unit_test();
#include <boost/test/unit_test.hpp>

// LLVM includes
#include "llvm/IR/Dominators.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/SourceMgr.h"

// Local libraries includes
#include "revng/BasicAnalyses/ReachingDefinitionsAnalysisImpl.h"

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

static Instruction *instructionByName(Function *F, const char *Name) {
  if (StringRef(Name).startswith("s:")) {
    Name = Name + 2;
    for (BasicBlock &BB : *F)
      for (Instruction &I : BB)
        if (auto *Store = dyn_cast<StoreInst>(&I))
          if (Store->getValueOperand()->hasName()
              and Store->getValueOperand()->getName() == Name)
            return &I;
  } else {
    for (BasicBlock &BB : *F)
      for (Instruction &I : BB)
        if (I.hasName() and I.getName() == Name)
          return &I;
  }

  revng_abort("Couldn't find a Value with the requested name");
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

template<typename T, typename B>
static void assertReachers(Function *F,
                           const RDA::Analysis<T, B> &A,
                           const char *InstructionName,
                           std::vector<const char *> ExpectedNames) {
  auto *I = cast<LoadInst>(instructionByName(F, InstructionName));
  std::set<Instruction *> Expected;
  for (const char *Name : ExpectedNames)
    Expected.insert(instructionByName(F, Name));

  std::set<Instruction *> Actual;
  for (Instruction *Reacher : A.getReachers(I))
    Actual.insert(Reacher);

  if (Expected != Actual) {
    dbg << "Unexpected result:\n";
    dbg << "Expected:\n";
    for (Instruction *I : Expected)
      I->dump();
    dbg << "Actual:\n";
    for (Instruction *I : Actual)
      I->dump();
    revng_abort();
  }
}

using ColorMap = std::map<BasicBlock *, RDA::ColorsList>;

namespace RDA {

template<>
struct ColorsProviderTraits<ColorMap> {
  static ColorsList &Empty;
  static const ColorsList &getBlockColors(const ColorMap &CP, BasicBlock *BB) {
    auto It = CP.find(BB);
    if (It == CP.end())
      return EmptyColorsList;
    else
      return It->second;
  }

  static int32_t getEdgeColor(const ColorMap &CP,
                              BasicBlock *Source,
                              BasicBlock *Destination) {
    if (auto *Branch = dyn_cast<BranchInst>(Source->getTerminator())) {

      if (Branch->isUnconditional())
        return 0;

      bool First = Source->getTerminator()->getSuccessor(0) == Destination;
      int32_t Pointer = reinterpret_cast<intptr_t>(Branch->getCondition());
      return Pointer * (First ? 1 : -1);
    } else {
      return 0;
    }
  }

  static const llvm::SmallVector<int32_t, 4> &
  getResetColors(const ColorMap &CNP, llvm::BasicBlock *BB) {
    static llvm::SmallVector<int32_t, 4> ResultVector;
    std::set<int32_t> Result;

    // Find all instructions used as a condition in a conditional branch
    for (Instruction &I : *BB)
      for (Use &U : I.uses())
        if (auto *B = dyn_cast<BranchInst>(U.getUser()))
          if (B->isConditional() and U.getOperandNo() == 0)
            Result.insert(reinterpret_cast<intptr_t>(&I));

    ResultVector.clear();
    std::copy(Result.begin(), Result.end(), std::back_inserter(ResultVector));
    return ResultVector;
  }
};

} // namespace RDA

class Test {
public:
  enum Type { Regular, Conditional, Both };

private:
  LLVMContext &Context;

public:
  Test() : Context(getGlobalContext()) {}

  void
  test(const char *Body,
       std::vector<std::pair<const char *, std::vector<const char *>>> Checks,
       std::vector<const char *> BlackList = {},
       Type T = Both) {

    std::unique_ptr<Module> M = loadModule(Context, Body);
    Function *F = M->getFunction("main");

    std::set<BasicBlock *> BasicBlockBlackList;
    for (const char *Name : BlackList)
      BasicBlockBlackList.insert(basicBlockByName(F, Name));

    if (T == Regular || T == Both) {
      using Analysis = RDA::Analysis<RDA::NullColorsProvider,
                                     std::set<BasicBlock *>>;
      Analysis A(F, RDA::NullColorsProvider(), BasicBlockBlackList);
      A.registerExtremal(&F->getEntryBlock());
      A.initialize();
      A.run();

      for (auto &P : Checks)
        assertReachers(F, A, P.first, P.second);
    }

    if (T == Conditional || T == Both) {

      highlightConditionEdges(*F);

      // Compute the dominator tree
      // TODO: in more recent LLVM versions we don't need to recompute the
      //       dominator tree but we'll be able to update it
      DominatorTree DT(*F);

      ColorMap Colors;

      // Perform a light version of the ConditionNumberingPass
      std::map<Value *, int> ConditionsMap;
      for (BasicBlock &BB : *F) {
        auto *T = dyn_cast<BranchInst>(BB.getTerminator());
        if (T == nullptr or T->isUnconditional())
          continue;

        int32_t ConditionIndex = reinterpret_cast<intptr_t>(T->getCondition());

        // ConditionIndex at the first iteration will be positive, at the second
        // negative
        std::array<BasicBlock *, 2> Successors{ T->getSuccessor(0),
                                                T->getSuccessor(1) };
        for (BasicBlock *Successor : Successors) {
          revng_assert(Successor->getSinglePredecessor() == &BB);

          SmallVector<BasicBlock *, 6> Descendants;
          DT.getDescendants(Successor, Descendants);
          for (BasicBlock *Descendant : Descendants)
            Colors[Descendant].push_back(ConditionIndex);

          ConditionIndex = -ConditionIndex;
        }
      }

      using Analysis = RDA::Analysis<ColorMap, std::set<BasicBlock *>>;
      Analysis CA(F, Colors, BasicBlockBlackList);
      CA.registerExtremal(&F->getEntryBlock());
      CA.initialize();
      CA.run();

      for (auto &P : Checks)
        assertReachers(F, CA, P.first, P.second);
    }
  }
};

BOOST_AUTO_TEST_CASE(OneStoreOneLoad) {
  Test X;

  //
  // One store, one load
  //
  const char *Body = R"LLVM(
  %zero = add i64 0, 0
  store i64 %zero, i64* @rax
  %load_rax = load i64, i64* @rax
  ret void
)LLVM";

  X.test(Body, { { "load_rax", { "s:zero" } } });
}

BOOST_AUTO_TEST_CASE(StoreToDifferentCSV) {
  Test X;

  //
  // Store to a different CSV
  //
  const char *Body = R"LLVM(
  %zero = add i64 0, 0
  store i64 %zero, i64* @rax
  %one = add i64 0, 0
  store i64 %one, i64* @rbx
  %load_rax = load i64, i64* @rax
  ret void
)LLVM";

  X.test(Body, { { "load_rax", { "s:zero" } } });
}

BOOST_AUTO_TEST_CASE(ClobberingStore) {
  Test X;

  //
  // Store clobbering a previous store
  //
  const char *Body = R"LLVM(
  %zero = add i64 0, 0
  store i64 %zero, i64* @rax
  %one = add i64 1, 0
  store i64 %one, i64* @rax
  %load_rax = load i64, i64* @rax
  ret void
)LLVM";

  X.test(Body, { { "load_rax", { "s:one" } } });
}

BOOST_AUTO_TEST_CASE(LoadReachingAnotherLoad) {
  Test X;

  //
  // Load reaching another load
  //
  const char *Body = R"LLVM(
  %load_rax1 = load i64, i64* @rax
  %load_rax2 = load i64, i64* @rax
  ret void
)LLVM";

  X.test(Body, { { "load_rax2", { "load_rax1" } } });
}

BOOST_AUTO_TEST_CASE(MultipleLoadsReachingAnotherLoad) {
  Test X;

  //
  // Multiple loads reaching another load
  //
  const char *Body = R"LLVM(
  %load_rax1 = load i64, i64* @rax
  %load_rax2 = load i64, i64* @rax
  %load_rax3 = load i64, i64* @rax
  ret void
)LLVM";

  X.test(Body, { { "load_rax3", { "load_rax1" } } });
}

BOOST_AUTO_TEST_CASE(IfStatement) {
  Test X;

  //
  // If statement
  //
  const char *If = R"LLVM(
  %storezero = add i64 0, 0
  store i64 %storezero, i64* @rax
  br i1 0, label %one, label %two

one:
  %storeone = add i64 0, 0
  store i64 %storeone, i64* @rax
  br label %end

two:
  %storetwo = add i64 0, 0
  store i64 %storetwo, i64* @rax
  br label %end

end:
  %load_rax = load i64, i64* @rax
  ret void
)LLVM";

  X.test(If, { { "load_rax", { "s:storeone", "s:storetwo" } } });

  // Now try again but inhibiting propgation to the end basic block
  X.test(If, { { "load_rax", {} } }, { "end" });
}

BOOST_AUTO_TEST_CASE(Loop) {
  Test X;

  //
  // Loop
  //
  const char *Body = R"LLVM(
  %storeone = add i64 0, 0
  store i64 %storeone, i64* @rax
  br label %head

head:
  %load_rax = load i64, i64* @rax
  %storetwo = add i64 0, 0
  store i64 %storetwo, i64* @rax
  br i1 0, label %end, label %head

end:
  ret void
)LLVM";

  X.test(Body, { { "load_rax", { "s:storeone", "s:storetwo" } } });
}

BOOST_AUTO_TEST_CASE(SelfReachingLoad) {
  Test X;

  //
  // Self-reaching load
  //
  const char *Body = R"LLVM(
  br label %head

head:
  %load_rax = load i64, i64* @rax
  br i1 0, label %end, label %head

end:
  ret void
)LLVM";

  X.test(Body, { { "load_rax", {} } });
}

BOOST_AUTO_TEST_CASE(RepeatedIfStatement) {
  Test X;

  //
  // Repeated if statement
  //
  const char *RepeatedIf = R"LLVM(
  %storezero = add i64 0, 0
  store i64 %storezero, i64* @rax
  br i1 0, label %one, label %two

one:
  %storeone = add i64 0, 0
  store i64 %storeone, i64* @rax
  br label %secondif

two:
  %storetwo = add i64 0, 0
  store i64 %storetwo, i64* @rax
  br label %secondif

secondif:
  br i1 0, label %three, label %four

three:
  %load_three = load i64, i64* @rax
  br label %end

four:
  %load_four = load i64, i64* @rax
  br label %end

end:
  ret void
)LLVM";

  X.test(RepeatedIf,
         { { "load_three", { "s:storeone", "s:storetwo" } },
           { "load_four", { "s:storeone", "s:storetwo" } } },
         {},
         Test::Regular);

  X.test(RepeatedIf,
         { { "load_three", { "s:storeone" } },
           { "load_four", { "s:storetwo" } } },
         {},
         Test::Conditional);
}

BOOST_AUTO_TEST_CASE(ConditionalDefinition) {
  Test X;

  //
  // Conditional definition
  //
  const char *ConditionalDefinition = R"LLVM(
  %storezero = add i64 0, 0
  store i64 %storezero, i64* @rax
  br i1 0, label %one, label %secondif

one:
  %storeone = add i64 0, 0
  store i64 %storeone, i64* @rax
  br label %secondif

secondif:
  br i1 0, label %three, label %four

three:
  %load_one = load i64, i64* @rax
  br label %end

four:
  %load_two = load i64, i64* @rax
  br label %end

end:
  ret void
)LLVM";

  X.test(ConditionalDefinition,
         { { "load_one", { "s:storeone" } },
           { "load_two", { "s:storezero" } } },
         {},
         Test::Conditional);
}

BOOST_AUTO_TEST_CASE(LoopClobbering) {
  Test X;

  //
  // Conditional definition
  //
  const char *ConditionalDefinition = R"LLVM(
  %variable = alloca i1
  br label %head

head:
  %variable_read = load i1, i1 *%variable
  br i1 %variable_read, label %one, label %two

one:
  %storezero = add i64 0, 0
  store i64 %storezero, i64* @rax
  br label %head

two:
  %load_one = load i64, i64 *@rax
  br label %end

end:
  ret void
)LLVM";

  X.test(ConditionalDefinition,
         { { "load_one", { "s:storezero" } } },
         {},
         Test::Conditional);
}
