//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"

#include "revng/ADT/Queue.h"
#include "revng/BasicAnalyses/GeneratedCodeBasicInfo.h"
#include "revng/HelperInliningAnalyses/DetectUninlinableHelpers.h"
#include "revng/Support/Assert.h"
#include "revng/Support/Debug.h"
#include "revng/Support/IRHelpers.h"

using namespace llvm;

static Logger Log("detect-uninlinable-helpers");

namespace DetectUninlinableHelpers {

bool isPointerToConstantGlobal(const Value *Pointer) {
  const Value *Stripped = Pointer->stripPointerCasts();
  const auto *Global = dyn_cast<GlobalVariable>(Stripped);
  return Global != nullptr and Global->isConstant();
}

std::optional<BitVector> computeCriticalArgumentsFor(const Function &Helper) {

  // A `Function` declaration has no body to inline, so it cannot be inlined at
  // any call site.
  if (Helper.isDeclaration())
    return std::nullopt;

  // Seed the backward worklist with every critical operand: the condition of
  // every `switch` and the index operands of every `getelementptr`. Done in a
  // single pass over the function.
  OnceQueue<const Value *> Worklist;
  for (const BasicBlock &BB : Helper) {
    for (const Instruction &I : BB) {
      if (const auto *GEP = dyn_cast<GetElementPtrInst>(&I))
        for (const auto &Index : GEP->indices())
          Worklist.insert(Index);
    }

    if (const auto *Switch = dyn_cast<SwitchInst>(BB.getTerminator()))
      Worklist.insert(Switch->getCondition());
  }

  BitVector Critical(Helper.arg_size(), false);

  // Backward dataflow walk: for every `Value` reached, classify it.
  while (not Worklist.empty()) {
    const Value *V = Worklist.pop();

    if (const auto *Arg = dyn_cast<Argument>(V)) {

      // A formal parameter has been reached, so this argument is critical.
      Critical.set(Arg->getArgNo());
    } else if (const auto *Load = dyn_cast<LoadInst>(V)) {

      // Loads from a constant global are fine: the loaded value is statically
      // known. Anything else is a load from runtime memory and we have to
      // give up — the helper cannot be inlined at any call site.
      if (not isPointerToConstantGlobal(Load->getPointerOperand()))
        return std::nullopt;
    } else if (const auto *I = dyn_cast<Instruction>(V)) {

      // Any other instruction. Enqueue its operands to keep walking back.
      for (const Use &U : I->operands())
        Worklist.insert(U.get());
    }
  }

  return Critical;
}

} // namespace DetectUninlinableHelpers

namespace {

/// True if any instruction in `F`'s body is an `insertvalue`, which the
/// downstream decompile path does not currently support.
bool containsInsertValue(const Function &F) {
  for (const BasicBlock &BB : F)
    for (const Instruction &I : BB)
      if (isa<InsertValueInst>(&I))
        return true;
  return false;
}

/// Tests condition under which `F` must not be inlined at any call site.
bool mustNotInline(const Function &F,
                   const SmallPtrSetImpl<const Function *> &Recursive) {
  if (Recursive.contains(&F)) {
    revng_log(Log, "demote " << F.getName() << " reason=recursive-scc");
    return true;
  }
  if (containsInsertValue(F)) {
    revng_log(Log, "demote " << F.getName() << " reason=insertvalue");
    return true;
  }
  return false;
}

/// Pass that tags every inlinable `revng_inline` helper with its
/// `!revng.inline.policy` metadata (which represents whose arguments needs to
/// be `Constant` at the call site for meeting the inlining condition) and
/// strips the `revng_inline` section attribute from those that must not be
/// inlined.
class DetectUninlinableHelpersPass : public ModulePass {
public:
  static char ID;

  DetectUninlinableHelpersPass() : ModulePass(ID) {}

  bool runOnModule(Module &M) override {
    using namespace DetectUninlinableHelpers;

    // Collect every function in a recursive SCC on the call graph; these
    // cannot be inlined without an unbounded fixed-point loop.
    SmallPtrSet<const Function *, 8> Recursive;
    CallGraph CG(M);
    for (auto It = scc_begin(&CG), End = scc_end(&CG); It != End; ++It)
      if (It.hasCycle())
        for (auto &Node : *It)
          if (const Function *F = Node->getFunction())
            Recursive.insert(F);

    for (Function &F : M) {
      if (F.getSection() != InlineHelpersSection)
        continue;

      if (mustNotInline(F, Recursive)) {
        F.setSection("");
        continue;
      }

      auto Result = computeCriticalArgumentsFor(F);
      if (not Result.has_value()) {
        revng_log(Log,
                  "demote " << F.getName()
                            << " reason=critical-args-runtime-load");
        F.setSection("");
        continue;
      }

      serializeInliningPolicy(F, *Result);
      revng_log(Log,
                "keep " << F.getName()
                        << " critical-arguments-count=" << Result->count());
    }

    return true;
  }
};

char DetectUninlinableHelpersPass::ID = 0;

using Register = RegisterPass<DetectUninlinableHelpersPass>;
static Register X("detect-uninlinable-helpers",
                  "Detect which `revng_inline` helpers can be inlined at lift "
                  "time; "
                  "those that can are tagged with `!revng.inline.policy` "
                  "metadata, those "
                  "that cannot have their `revng_inline` section attribute "
                  "stripped",
                  true,
                  true);

} // namespace
