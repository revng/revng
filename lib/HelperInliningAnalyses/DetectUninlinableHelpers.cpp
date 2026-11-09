//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"

#include "revng/ADT/Queue.h"
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

static std::optional<BitVector> walkBack(const Function &Helper,
                                         OnceQueue<const Value *> Worklist);

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

  return walkBack(Helper, std::move(Worklist));
}

/// \return the arguments of \p Helper the values in \p Worklist depend on, or
///         `std::nullopt` if they do not become constants through inlining.
static std::optional<BitVector> walkBack(const Function &Helper,
                                         OnceQueue<const Value *> Worklist) {
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
    } else if (isa<PHINode>(V)) {

      // Unlike a `select`, a `phi` also depends on the branches leading to it,
      // which the walk does not follow: give up.
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
bool mustNotInline(const Function &F) {
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

    // Demote every function in a recursive SCC on the call graph; these
    // cannot be inlined without an unbounded fixed-point loop.
    CallGraph CG(M);
    for (auto It = scc_begin(&CG), End = scc_end(&CG); It != End; ++It) {
      if (It.hasCycle()) {
        for (auto &Node : *It) {
          Function *F = Node->getFunction();
          if (F != nullptr and isInlineHelper(*F)) {
            revng_log(Log,
                      "demote " << F->getName() << " reason=recursive-scc");
            F->setSection("");
          }
        }
      }
    }

    // Visit callees before their callers, so that they have a policy when
    // visiting their call sites. Functions the call graph does not reach have
    // no callers.
    DenseMap<const Function *, BitVector> CriticalArguments;
    for (CallGraphNode *Node : post_order(&CG)) {
      if (Node->getFunction() == nullptr
          or not isInlineHelper(*Node->getFunction()))
        continue;

      Function &F = *Node->getFunction();
      if (mustNotInline(F)) {
        F.setSection("");
        continue;
      }

      auto Result = computeCriticalArgumentsFor(F);
      if (not Result.has_value()) {
        revng_log(Log,
                  "demote " << F.getName()
                            << " reason=critical-args-not-constant");
        F.setSection("");
        continue;
      }

      // Calls to inline helpers are expected to be inlined: the actual
      // arguments in their critical positions have to become constants too.
      // Keep the call if they depend on a runtime value or on an argument that
      // is not critical already: whether to inline it then depends on the
      // actual arguments of the caller.
      for (auto &[CallSite, CalleeNode] : *CG[&F]) {
        auto It = CriticalArguments.find(CalleeNode->getFunction());
        if (It == CriticalArguments.end())
          continue;

        auto *Call = cast<CallBase>(*CallSite);
        OnceQueue<const Value *> Worklist;
        for (unsigned Index : It->second.set_bits())
          Worklist.insert(Call->getArgOperand(Index));

        auto Reached = walkBack(F, std::move(Worklist));
        if (not Reached.has_value() or Reached->test(*Result)) {
          Call->addFnAttr(Attribute::NoInline);
          revng_log(Log,
                    "keep call " << F.getName() << " -> "
                                 << It->first->getName());
        }
      }

      CriticalArguments[&F] = *Result;
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
