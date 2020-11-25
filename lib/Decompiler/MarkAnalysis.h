#pragma once

//
// Copyright (c) rev.ng Srls. See LICENSE.md for details.
//

/// \brief Analysis that marks instructions to be serialized in C

#include <map>

#include "revng/ADT/ZipMapIterator.h"
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

class IntersectionMonotoneSetWithTaint;

using LatticeElement = IntersectionMonotoneSetWithTaint;

class IntersectionMonotoneSetWithTaint {
public:
  using Instruction = llvm::Instruction;
  using TaintMap = std::map<const Instruction *, std::set<const Instruction *>>;
  using const_iterator = typename TaintMap::const_iterator;
  using iterator = typename TaintMap::iterator;
  using size_type = typename TaintMap::size_type;

protected:
  TaintMap TaintedPending;
  bool IsBottom;

protected:
  IntersectionMonotoneSetWithTaint(const IntersectionMonotoneSetWithTaint &) =
    default;

public:
  IntersectionMonotoneSetWithTaint() : TaintedPending(), IsBottom(true){};

  IntersectionMonotoneSetWithTaint copy() const { return *this; }
  IntersectionMonotoneSetWithTaint &
  operator=(const IntersectionMonotoneSetWithTaint &) = default;

  IntersectionMonotoneSetWithTaint(IntersectionMonotoneSetWithTaint &&) =
    default;
  IntersectionMonotoneSetWithTaint &
  operator=(IntersectionMonotoneSetWithTaint &&) = default;

public:
  const_iterator begin() const { return TaintedPending.begin(); }
  const_iterator end() const { return TaintedPending.end(); }

  void dump() const { dump(dbg); }

  template<typename O>
  void dump(O &Output) const {
    Output << "{ ";
    for (const auto &[Instr, _] : TaintedPending)
      Output << Instr << " ";
    Output << " }";
  }

public:
  size_type size() const {
    revng_assert(not IsBottom);
    return TaintedPending.size();
  }

  void insert(const Instruction *Key) {
    revng_assert(not IsBottom);
    TaintedPending[Key];
  }

  size_type erase(const Instruction *El) {
    revng_assert(not IsBottom);
    return TaintedPending.erase(El);
  }

  const_iterator erase(const_iterator It) {
    revng_assert(not IsBottom);
    return this->TaintedPending.erase(It);
  }

  bool isPending(const Instruction *Key) const {
    revng_assert(not IsBottom);
    return TaintedPending.count(Key);
  }

public:
  static IntersectionMonotoneSetWithTaint bottom() {
    return IntersectionMonotoneSetWithTaint();
  }

  static IntersectionMonotoneSetWithTaint top() {
    IntersectionMonotoneSetWithTaint Res = {};
    Res.IsBottom = false;
    return Res;
  }

public:
  void combine(const IntersectionMonotoneSetWithTaint &Other) {
    // Simply intersects the sets
    if (Other.IsBottom)
      return;

    if (IsBottom) {
      this->TaintedPending = Other.TaintedPending;
      IsBottom = false;
      return;
    }

    std::vector<iterator> ToDrop;

    auto OtherCopy = Other.copy();
    const_iterator OtherEnd = OtherCopy.end();

    iterator SetIt = this->TaintedPending.begin();
    iterator SetEnd = this->TaintedPending.end();
    for (; SetIt != SetEnd; ++SetIt) {
      iterator OtherIt = OtherCopy.TaintedPending.find(SetIt->first);
      if (OtherIt == OtherEnd)
        ToDrop.push_back(SetIt);
      else
        SetIt->second.merge(OtherIt->second);
    }

    for (iterator I : ToDrop)
      this->TaintedPending.erase(I);
  }

  bool lowerThanOrEqual(const IntersectionMonotoneSetWithTaint &Other) const {
    if (IsBottom)
      return true;

    if (Other.IsBottom)
      return false;

    if (size() < Other.size())
      return false;

    const auto &Zip = zipmap_range(TaintedPending, Other.TaintedPending);
    for (const auto &PtrPair : Zip) {

      if (nullptr == PtrPair.first)
        return false;

      if (nullptr == PtrPair.second)
        continue;

      revng_assert(PtrPair.first->first == PtrPair.second->first);
      const auto &ThisTaintSet = PtrPair.first->second;
      const auto &OtherTaintSet = PtrPair.second->second;

      if (not std::includes(OtherTaintSet.begin(),
                            OtherTaintSet.end(),
                            ThisTaintSet.begin(),
                            ThisTaintSet.end())) {
        return false;
      }
    }
    return true;
  }
};

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

    auto LiveInIt = LiveIn.find(Destination);
    if (LiveInIt == LiveIn.end())
      return llvm::None;

    const auto &LiveInSet = LiveInIt->second;

    LatticeElement Result = Original.copy();

    const auto IsDead = [&LiveInSet](const llvm::Instruction *I) {
      return not LiveInSet.contains(I);
    };

    for (auto ResultsIt = Result.begin(); ResultsIt != Result.end();) {
      if (IsDead(ResultsIt->first)) {
        ResultsIt = Result.erase(ResultsIt);
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

  void initialize();

  InterruptType transfer(const llvm::BasicBlock *);
};

} // namespace MarkAnalysis
