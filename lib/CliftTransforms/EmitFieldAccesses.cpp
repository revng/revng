//
// This file is distributed under the MIT License. See LICENSE.md for details.
//
#include <compare>
#include <cstdint>
#include <memory>

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"

#include "revng/Clift/CliftEnums.h"
#include "revng/CliftTransforms/Passes.h"

#include "BestTraversal.h"
#include "EmitFieldAccesses.h"
#include "FieldAccessReplacement.h"
#include "PointerArithmetic.h"

namespace clift {
#define GEN_PASS_DEF_CLIFTEMITFIELDACCESSES
#include "revng/CliftTransforms/Passes.h.inc"
} // namespace clift

using namespace clift;

namespace {

struct EmitFieldAccessesPass
  : clift::impl::CliftEmitFieldAccessesBase<EmitFieldAccessesPass> {
  void runOnOperation() override {
    if (emitFieldAccesses(getOperation()).failed()) {
      signalPassFailure();
    }
  }
};

/// Holds the planned `Replacement` for a single expression
struct PlannedReplacement {
  clift::ExpressionOpInterface Op;
  PointerArithmetic PA;
  Traversal BestTraversal;
};

/// Implementation of the high level `emitFieldAccesses` phases inside the
/// anonymous namespace in this translation unit
mlir::LogicalResult emitFieldAccessesImpl(clift::FunctionOp Function) {

  bool PropagatedThroughIndirection = false;

  do {
    PropagatedThroughIndirection = false;

    // `TraversalInfoMap` cache. Even if not elegant, we store the data
    // computed at the pass level so that we can cache it instead of
    // recomputing it every time
    TraversalInfoMap TraversalMap;

    // Phase 1-2: Collect all planned `Replacement`s without modifying the IR
    llvm::SmallVector<PlannedReplacement> Replacements;
    Function->walk([&TraversalMap,
                    &Replacements](clift::ExpressionOpInterface Op) {
      // 1. We inspect all the `ExpressionOp`s in the current `Function`
      std::optional<PointerArithmetic> PA = computePointerArithmetic(Op);

      // The `PointerArithmetic` returned object could be empty at the moment
      // of return, and this is a signal that we could not compute a
      // `PointerArithmetic` for the current `ExpressionOpInterface`

      // 2. We proceed with the computation of the `BestTraversal` for the
      //    current `PointerArithmetic`
      if (PA) {
        std::optional<Traversal> BT = computeBestTraversal(Op,
                                                           *PA,
                                                           TraversalMap);

        if (BT) {
          Replacements.push_back({ Op, std::move(*PA), std::move(*BT) });
        }
      }
    });

    // Phase 3: Apply all replacements. If any replacement propagates a type
    // through an indirection, we rerun the whole EFA to discover new
    // rewriting opportunities enabled by the newly typed pointers.
    for (const auto &R : Replacements) {
      if (replaceFieldAccess(R.Op, R.PA, R.BestTraversal))
        PropagatedThroughIndirection = true;
    }
  } while (PropagatedThroughIndirection);

  // The IR is always in a valid state, regardless of whether we performed an
  // operation rewrite or not.
  return mlir::success();
}

} // namespace

/// `emitFieldAccesses` driver that can be called by importing the header
mlir::LogicalResult emitFieldAccesses(clift::FunctionOp Function) {
  return emitFieldAccessesImpl(Function);
}

clift::PassPtr<clift::FunctionOp> clift::createEmitFieldAccessesPass() {
  return std::make_unique<EmitFieldAccessesPass>();
}
