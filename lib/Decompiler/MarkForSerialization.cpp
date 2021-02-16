//
// Copyright rev.ng Srls. See LICENSE.md for details.
//

/// \brief Dataflow analysis to identify which Instructions must be serialized

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/Casting.h"

#include "revng/Model/LoadModelPass.h"
#include "revng/Support/IRHelpers.h"

#include "revng-c/Decompiler/MarkForSerialization.h"
#include "revng-c/IsolatedFunctions/IsolatedFunctions.h"
#include "revng-c/RestructureCFGPass/BasicBlockNode.h"
#include "revng-c/RestructureCFGPass/RegionCFGTree.h"
#include "revng-c/RestructureCFGPass/RestructureCFG.h"
#include "revng-c/TargetFunctionOption/TargetFunctionOption.h"

#include "MarkAnalysis.h"

Logger<> MarkLog("mark-serialization");

namespace MarkAnalysis {

static bool isPure(const llvm::Instruction & /*Call*/) {
  return false;
}

static bool
haveInterferingSideEffects(const llvm::Instruction & /*InstrWithSideEffects*/,
                           const llvm::Instruction & /*Other*/) {
  return true;
}

Analysis::InterruptType Analysis::transfer(const llvm::BasicBlock *BB) {
  using namespace llvm;
  revng_log(MarkLog,
            "transfer: BB in Function: " << BB->getParent()->getName() << '\n'
                                         << BB);

  LatticeElement Pending = this->State[BB].copy();

  size_t NBBDuplicates = NDuplicates.at(BB);
  for (const Instruction &I : *BB) {
    revng_log(MarkLog, "Analyzing Instr: '" << &I << "': " << dumpToString(&I));

    // Operands are removed from pending
    revng_log(MarkLog, "Remove operands from pending.");

    MarkLog.indent();
    revng_log(MarkLog, "Operands:");
    for (auto &TheUse : I.operands()) {
      Value *V = TheUse.get();
      revng_log(MarkLog, "Op: '" << V << "': " << dumpToString(V));

      MarkLog.indent();
      if (auto *UsedInstr = dyn_cast<Instruction>(V)) {
        revng_log(MarkLog, "Op is Instruction: erase it from pending");
        Pending.erase(UsedInstr);
      } else {
        revng_log(MarkLog, "Op is NOT Instruction: leave it in pending");
        revng_assert(isa<Argument>(V) or isa<Constant>(V) or isa<BasicBlock>(V)
                     or isa<MetadataAsValue>(V));
      }
      MarkLog.unindent();
    }
    MarkLog.unindent();

    // PHINodes are never serialized directly in the BB they are.
    if (isa<PHINode>(I))
      continue;

    // Skip branching instructions.
    // Branch instructions are never serialized directly, because it's only
    // after building an AST and matching ifs, loops, switches and others that
    // we really know what kind of C statement we want to emit for a given
    // branch.
    if (isa<BranchInst>(I) or isa<SwitchInst>(I))
      continue;

    if (isa<InsertValueInst>(I)) {
      // InsertValueInst are serialized in C as:
      //   struct x = { .designated = 0xDEAD, .initializers = 0xBEEF };
      //   x.designated = value_that_overrides_0xDEAD;
      // The second statement is always necessary.
      ToSerialize[&I].set(NeedsManyStatements);
      revng_log(MarkLog, "Instr NeedsManyStatements");
    }

    if (isa<InsertValueInst>(I) or isa<AllocaInst>(I)) {
      // As noted in the comment above, InsertValueInst always need a local
      // variable (x in the example above) for the computation of the expression
      // that represents the result of Instruction itself.
      // This is the local variable in C that will be used by x's users.
      // Also AllocaInst always need a local variable, which is the variable
      // allocated by the alloca.
      ToSerialize[&I].set(NeedsLocalVarToComputeExpr);
      revng_log(MarkLog, "Instr NeedsLocalVarToComputeExpr");
    }

    if (isa<StoreInst>(&I) or (isa<CallInst>(&I) and not isPure(I))) {
      // StoreInst and CallInst that are not pure always have side effects.
      ToSerialize[&I].set(HasSideEffects);
      revng_log(MarkLog, "Instr HasSideEffects");

      // Also, force calls to revng_init_local_sp to behave like if they had
      // many uses, so that they generate a local variable.
      if (auto *Call = dyn_cast<CallInst>(&I)) {
        llvm::StringRef CalleeName = Call->getCalledFunction()->getName();
        if (CalleeName == "revng_init_local_sp") {
          ToSerialize[&I].set(HasManyUses);
          revng_log(MarkLog, "Instr HasManyUses");
        }
      }
    }

    switch (I.getNumUses()) {

    case 1: {
      User *U = I.uses().begin()->getUser();
      Instruction *UserI = cast<Instruction>(U);
      BasicBlock *UserBB = UserI->getParent();
      auto UserNDuplicates = NDuplicates.at(UserBB);
      if (NBBDuplicates < UserNDuplicates) {
        ToSerialize[&I].set(HasDuplicatedUses);
        revng_log(MarkLog, "Instr HasDuplicatedUses");
      }
    } break;

    case 0: {
      // Do nothing
      ToSerialize[&I].set(AlwaysSerialize);
      revng_log(MarkLog, "Instr AlwaysSerialize");
    } break;

    default: {
      // Instructions with more than one use are always serialized.
      ToSerialize[&I].set(HasManyUses);
      revng_log(MarkLog, "Instr HasManyUses");
    } break;
    }

    if (ToSerialize.count(&I)) {
      revng_log(MarkLog, "Serialize Pending");
      // We also have to serialize all the instructions that are still pending
      // and have interfering side effects.
      for (auto PendingIt = Pending.begin(); PendingIt != Pending.end();) {
        const auto *PendingInstr = PendingIt->first;
        revng_log(MarkLog,
                  "Pending: '" << PendingInstr
                               << "': " << dumpToString(PendingInstr));
        if (haveInterferingSideEffects(I, *PendingInstr)) {
          ToSerialize[PendingInstr].set(HasInterferingSideEffects);
          revng_log(MarkLog, "HasInterferingSideEffects");

          PendingIt = Pending.erase(PendingIt);
        } else {
          ++PendingIt;
        }
      }
    } else {
      Pending.insert(&I);
      revng_log(MarkLog,
                "Add to pending: '" << &I << "': " << dumpToString(&I));
    }
  }

  return InterruptType::createInterrupt(std::move(Pending));
}

void Analysis::initialize() {
  Base::initialize();
  LivenessAnalysis::Analysis Liveness(F);
  Liveness.initialize();
  Liveness.run();
  LiveIn = Liveness.extractLiveIn();
}

} // namespace MarkAnalysis

void MarkForSerializationPass::getAnalysisUsage(llvm::AnalysisUsage &AU) const {
  AU.addRequired<LoadModelPass>();
  AU.addRequired<RestructureCFG>();
  AU.setPreservesAll();
}

bool MarkForSerializationPass::runOnFunction(llvm::Function &F) {

  // Skip non-isolated functions
  const model::Binary &Model = getAnalysis<LoadModelPass>().getReadOnlyModel();
  if (not hasIsolatedFunction(Model, F))
    return false;

  // If the `-single-decompilation` option was passed from command line, skip
  // decompilation for all the functions that are not the selected one.
  if (not TargetFunction.empty())
    if (not F.getName().equals(TargetFunction.c_str()))
      return false;

  // Compute the number of duplicates for each BasicBlock.
  const auto &RestructurePass = getAnalysis<RestructureCFG>();
  using MarkAnalysis::DuplicationMap;
  const DuplicationMap &NDuplicates = RestructurePass.getNDuplicates();

  // Mark instructions for serialization, and write the results in ToSerialize
  ToSerialize = {};
  MarkAnalysis::Analysis Mark(F, NDuplicates, ToSerialize);
  Mark.initialize();
  Mark.run();

  return true;
}

char MarkForSerializationPass::ID = 0;

using Register = llvm::RegisterPass<MarkForSerializationPass>;
static Register X("mark-for-serialization",
                  "Pass that marks Instructions for serialization in C",
                  false,
                  false);
