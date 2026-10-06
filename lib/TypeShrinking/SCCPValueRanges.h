#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <functional>

#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/ValueLattice.h"
#include "llvm/IR/ConstantRange.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Module.h"
#include "llvm/Transforms/Utils/SCCPSolver.h"

namespace TypeShrinking {

/// Wrapper around SCCPSolver to easily provide ranges for a `Value`
class SCCPValueRanges {
private:
  using LibraryInfo = llvm::TargetLibraryInfo;
  using LibraryInfoGetter = std::function<
    const LibraryInfo &(llvm::Function &)>;

private:
  llvm::SCCPSolver Solver;

public:
  SCCPValueRanges(llvm::Function &F, const LibraryInfo &TLI) :
    Solver(F.getParent()->getDataLayout(),
           getLibraryInfo(TLI),
           F.getContext()) {
    Solver.markBlockExecutable(&F.getEntryBlock());

    for (auto &Argument : F.args())
      Solver.markOverdefined(&Argument);

    do {
      Solver.solve();
    } while (Solver.resolvedUndefsIn(F));
  }

public:
  /// Return a nonempty range at the original scalar integer width, valid at
  /// every use. Unknown values have the full range.
  llvm::ConstantRange range(llvm::Value *V) const {
    if (auto *C = llvm::dyn_cast<llvm::ConstantInt>(V))
      return llvm::ConstantRange(C->getValue());

    // Other constants, including undef and poison, provide no usable bound.
    if (llvm::isa<llvm::Constant>(V))
      return fullRange(V);

    // getLatticeValueFor asserts on a value SCCP never recorded a state for.
    // A block can be reachable in the CFG and still not be executable, when a
    // constant branch condition proves its incoming edges are never taken.
    if (auto *I = llvm::dyn_cast<llvm::Instruction>(V)) {
      if (not Solver.isBlockExecutable(I->getParent()))
        return fullRange(V);
    }

    // getConstantRange asserts on any other lattice state.
    const auto &State = Solver.getLatticeValueFor(V);
    if (State.isConstantRange(false)) {
      llvm::ConstantRange Range = State.getConstantRange(false);
      if (not Range.isEmptySet())
        return Range;
    }
    return fullRange(V);
  }

private:
  static LibraryInfoGetter getLibraryInfo(const LibraryInfo &TLI) {
    return [&TLI](llvm::Function &) -> const LibraryInfo & { return TLI; };
  }

  static llvm::ConstantRange fullRange(const llvm::Value *V) {
    return llvm::ConstantRange::getFull(V->getType()->getIntegerBitWidth());
  }
};

} // namespace TypeShrinking
