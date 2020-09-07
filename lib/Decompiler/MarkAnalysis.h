#pragma once

//
// Copyright (c) rev.ng Srls. See LICENSE.md for details.
//

/// \brief Analysis that marks instructions to be serialized in C

#include <map>

#include "revng/Support/MonotoneFramework.h"

#include "revng-c/Decompiler/MarkForSerialization.h"
#include "revng-c/Liveness/LivenessAnalysis.h"

namespace llvm {

class BasicBlock;
class Function;
class Instruction;

} // end namespace llvm

namespace MarkAnalysis {

using DuplicationMap = std::map<const llvm::BasicBlock *, size_t>;

using LatticeElement = IntersectionMonotoneSet<const llvm::Instruction *>;

using SuccVector = llvm::SmallVector<const llvm::BasicBlock *, 2>;

class Analysis : public MonotoneFramework<Analysis,
                                          const llvm::BasicBlock *,
                                          LatticeElement,
                                          VisitType::ReversePostOrder,
                                          SuccVector> {
private:
  const llvm::Function &F;
  SerializationMap &ToSerialize;
  const DuplicationMap &NDuplicates;
  LivenessAnalysis::LivenessMap LiveIn;

public:
  using Base = MonotoneFramework<Analysis,
                                 const llvm::BasicBlock *,
                                 LatticeElement,
                                 VisitType::ReversePostOrder,
                                 SuccVector>;

  void assertLowerThanOrEqual(const LatticeElement &A,
                              const LatticeElement &B) const {
    revng_assert(A.lowerThanOrEqual(B));
  }

  Analysis(const llvm::Function &F,
           const DuplicationMap &NDuplicates,
           SerializationMap &ToSerialize) :
    Base(&F.getEntryBlock()),
    F(F),
    ToSerialize(ToSerialize),
    NDuplicates(NDuplicates),
    LiveIn() {
    Base::registerExtremal(&F.getEntryBlock());
  }

  [[noreturn]] void dumpFinalState() const { revng_abort(); }

  SuccVector successors(const llvm::BasicBlock *BB, InterruptType &) const {
    SuccVector Result;
    for (const llvm::BasicBlock *Successor :
         make_range(succ_begin(BB), succ_end(BB)))
      Result.push_back(Successor);
    return Result;
  }

  llvm::Optional<LatticeElement>
  handleEdge(const LatticeElement &Original,
             const llvm::BasicBlock * /*Source*/,
             const llvm::BasicBlock *Destination) const {
    llvm::Optional<LatticeElement> Result{ llvm::None };
    auto LiveInIt = LiveIn.find(Destination);
    if (LiveInIt == LiveIn.end())
      return Result;

    const auto &LiveInSet = LiveInIt->second;

    Result = Original.copy();

    const auto IsDead = [&LiveInSet](const llvm::Instruction *I) {
      return not LiveInSet.contains(I);
    };

    for (auto ResultsIt = Result->begin(); ResultsIt != Result->end();) {
      if (IsDead(*ResultsIt)) {
        ResultsIt = Result->erase(ResultsIt);
      } else {
        ++ResultsIt;
      }
    }

    return Result;
  }

  size_t successor_size(const llvm::BasicBlock *BB, InterruptType &) const {
    return succ_size(BB);
  }

  static LatticeElement extremalValue(const llvm::BasicBlock *) {
    return LatticeElement::top();
  }

  void initialize() {
    Base::initialize();
    LivenessAnalysis::Analysis Liveness(F);
    Liveness.initialize();
    Liveness.run();
    LiveIn = Liveness.extractLiveIn();
  }

  InterruptType transfer(const llvm::BasicBlock *);
};

} // namespace MarkAnalysis
