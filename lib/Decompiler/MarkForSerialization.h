#ifndef REVNGC_MARKFORSERIALIZATION_H
#define REVNGC_MARKFORSERIALIZATION_H
//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

/// \brief Dataflow analysis to identify which Instructions must be serialized

// std includes
#include <set>

// LLVM includes
#include <llvm/IR/Function.h>
#include <revng/Support/IRHelpers.h>

// revng includes
#include <revng/Support/MonotoneFramework.h>

class RegionCFG;

namespace MarkForSerialization {

using LatticeElement = IntersectionMonotoneSet<llvm::Instruction *>;

class Analysis
  : public MonotoneFramework<Analysis,
                             llvm::BasicBlock *,
                             LatticeElement,
                             VisitType::ReversePostOrder,
                             llvm::SmallVector<llvm::BasicBlock *, 2>> {
protected:
  llvm::Function &F;
  RegionCFG &RCFG;
  std::set<llvm::Instruction *> ToSerialize;
  std::vector<std::set<llvm::Instruction *>> ToSerializeInBB;
  std::map<llvm::BasicBlock *, size_t> BBToIdMap;
  std::map<llvm::BasicBlock *, size_t> NDuplicates;

public:
  using Base = MonotoneFramework<Analysis,
                                 llvm::BasicBlock *,
                                 LatticeElement,
                                 VisitType::ReversePostOrder,
                                 llvm::SmallVector<llvm::BasicBlock *, 2>>;

  void assertLowerThanOrEqual(const LatticeElement &A,
                              const LatticeElement &B) const {
    revng_abort();
  }

  Analysis(llvm::Function &F, RegionCFG &RCFG) :
    Base(&F.getEntryBlock()),
    F(F),
    RCFG(RCFG),
    ToSerialize(),
    ToSerializeInBB(),
    BBToIdMap(),
    NDuplicates() {
      Base::registerExtremal(&F.getEntryBlock());
  }

  void dumpFinalState() const { revng_abort(); }

  llvm::SmallVector<llvm::BasicBlock *, 2>
  successors(llvm::BasicBlock *BB, InterruptType &) const {
    llvm::SmallVector<llvm::BasicBlock *, 2> Result;
    for (llvm::BasicBlock *Successor : make_range(succ_begin(BB), succ_end(BB)))
      Result.push_back(Successor);
    return Result;
  }

  llvm::Optional<LatticeElement>
  handleEdge(const LatticeElement & /*Original*/,
             llvm::BasicBlock * /*Source*/,
             llvm::BasicBlock * /*Destination*/) const {
    return llvm::Optional<LatticeElement>();
  }

  size_t successor_size(llvm::BasicBlock *BB, InterruptType &) const {
    return succ_end(BB) - succ_begin(BB);
  }

  static LatticeElement extremalValue(llvm::BasicBlock *) {
    return LatticeElement::top();
  }

  InterruptType transfer(llvm::BasicBlock *);

  void initialize();

  const std::set<llvm::Instruction *> &getToSerialize() const {
    return ToSerialize;
  }

  const std::set<llvm::Instruction *> &
  getToSerialize(llvm::BasicBlock *B) const {
    return ToSerializeInBB.at(BBToIdMap.at(B));
  }

protected:
  void markValueToSerialize(llvm::Instruction *I) {
    ToSerialize.insert(I);
    ToSerializeInBB.at(BBToIdMap.at(I->getParent())).insert(I);
  }

  void markSetToSerialize(const LatticeElement &S) {
    for (llvm::Instruction *I : S)
      markValueToSerialize(I);
  }
};

} // end namespace MarkForSerialization

#endif /* ifndef REVNGC_MARKFORSERIALIZATION_H */
