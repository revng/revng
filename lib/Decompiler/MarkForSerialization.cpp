//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

/// \brief Dataflow analysis to identify which Instructions must be serialized

// LLVM includes
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Instructions.h>

// local libraries includes
#include "revng-c/RestructureCFGPass/BasicBlockNode.h"
#include "revng-c/RestructureCFGPass/RegionCFGTree.h"

// local includes
#include "MarkForSerialization.h"

static Logger<> MarkLog("mark-serialization");

using namespace llvm;

namespace MarkForSerialization {

static bool isPure(const Instruction & /*Call*/) {
  return false;
}

void Analysis::initialize() {
  Base::initialize();
  ToSerialize.clear();

  {
    // one empty set for each BB
    using InstrSet = std::set<Instruction *>;
    ToSerializeInBB = std::vector<InstrSet>(F.size(), InstrSet{});
  }

  size_t I = 0;
  for (BasicBlock &BB : F) {
    // map BasicBlock pointer to the position in ToSerializeInBB
    BBToIdMap[&BB] = I++;
  }
}

Analysis::InterruptType Analysis::transfer(BasicBlock *BB) {
  revng_log(MarkLog,
            "transfer: BB in Function: " << BB->getParent()->getName() << '\n'
                                         << BB);

  LatticeElement Pending = this->State[BB].copy();

  size_t NumDuplicatesOfBB = NDuplicates.at(BB);
  for (Instruction &I : *BB) {
    revng_log(MarkLog, "Analyzing: '" << &I << "': " << dumpToString(&I));
    // PHINodes are never serialized
    if (isa<PHINode>(&I))
      continue;
    // Skip this for now. We'll need to change it if we ever want to emit code
    // with goto statements
    // Branch instructions are never serialized directly, because it's only
    // after building an AST and matching ifs, loops, switches and others that
    // we really know what kind of C statement we want to emit for a given
    // branch.
    if (isa<BranchInst>(&I))
      continue;
    Pending.insert(&I);
    revng_log(MarkLog, "Add to pending: '" << &I << "': " << dumpToString(&I));
    revng_log(MarkLog, "Operands:");
    MarkLog.indent();
    for (auto &TheUse : I.operands()) {
      Value *V = TheUse.get();
      revng_log(MarkLog, "Op: '" << V << "': " << dumpToString(V));
      MarkLog.indent();
      if (auto *UseInstr = dyn_cast<Instruction>(V)) {
        revng_log(MarkLog, "Op is Instruction");
        Pending.erase(UseInstr);
      } else {
        revng_log(MarkLog, "Op is NOT Instruction");
        revng_assert(isa<Argument>(V) or isa<Constant>(V) or isa<BasicBlock>(V)
                     or isa<MetadataAsValue>(V));
      }
      MarkLog.unindent();
    }
    MarkLog.unindent();

    bool HasSideEffects = isa<AllocaInst>(&I) or isa<StoreInst>(&I)
                          or isa<InsertValueInst>(&I)
                          or isa<ExtractValueInst>(&I)
                          or (isa<CallInst>(&I) and not isPure(I));
    if (HasSideEffects) {
      revng_log(MarkLog, "Serialize Pending");
      markSetToSerialize(Pending);
      Pending = LatticeElement::top(); // empty set
      markValueToSerialize(&I);
    } else {
      switch (I.getNumUses()) {
      case 1: {
        User *U = I.uses().begin()->getUser();
        if (not isa<Instruction>(U)) {
          Pending.insert(&I);
          continue;
        }
        Instruction *UserI = cast<Instruction>(U);
        if (NumDuplicatesOfBB != NDuplicates.at(UserI->getParent())) {
          revng_assert(NumDuplicatesOfBB < NDuplicates.at(UserI->getParent()));
          markValueToSerialize(&I);
        } else {
          Pending.insert(&I);
        }

      } break;
      default:
        revng_log(MarkLog, "Mark this to serialize");
        markValueToSerialize(&I);
        break;
      }
    }
  }

  return InterruptType::createInterrupt(std::move(Pending));
}

} // namespace MarkForSerialization
