#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/BitVector.h"

#include "revng/MFP/MFP.h"
#include "revng/RegisterUsageAnalyses/Function.h"

namespace rua {

class Liveness {
private:
  using Set = llvm::BitVector;
  using RegisterSet = Set;

public:
  using LatticeElement = Set;
  using GraphType = llvm::Inverse<const Function *>;
  using Label = const BlockNode *;

private:
  Set Default;

public:
  Liveness(const Function &F) {
    uint8_t Max = 0;
    for (const Block *Block : F.nodes()) {
      for (const Operation &Operation : Block->Operations) {
        Max = std::max(Max, Operation.Target);
      }
    }

    Default.resize(Max + 1);
  }

public:
  Set defaultValue() const { return Default; }

public:
  Set combineValues(const Set &LHS, const Set &RHS) const {
    Set Result = LHS;
    Result |= RHS;
    return Result;
  }

  bool isLessOrEqual(const Set &LHS, const Set &RHS) const {
    // RHS must contain or be equal to LHS
    Set Intersection = RHS;
    Intersection &= LHS;
    return Intersection == LHS;
  }

  RegisterSet applyTransferFunction(const BlockNode *Block,
                                    const RegisterSet &InitialState) const {
    RegisterSet Result = InitialState;

    for (const Operation &Operation :
         llvm::make_range(Block->rbegin(), Block->rend())) {

      switch (Operation.Type) {
      case OperationType::Read:
        Result.set(Operation.Target);
        break;

      case OperationType::Write:
      case OperationType::Clobber:
        Result.reset(Operation.Target);
        break;

      case OperationType::Invalid:
        revng_abort();
        break;
      }
    }

    return Result;
  }
};

static_assert(MFP::MonotoneFrameworkInstance<Liveness>);

} // namespace rua
