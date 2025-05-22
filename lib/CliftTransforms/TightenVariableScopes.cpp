//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <algorithm>
#include <ranges>

#include "llvm/ADT/SmallVector.h"

#include "mlir/Pass/Pass.h"

#include "revng/Clift/CliftOpHelpers.h"
#include "revng/CliftTransforms/Passes.h"

namespace clift {
#define GEN_PASS_DEF_CLIFTTIGHTENVARIABLESCOPES
#include "revng/CliftTransforms/Passes.h.inc"
} // namespace clift

using namespace clift;

namespace {

// Custom walk function that provides nesting level.
template<typename CallbackT>
static void walkWithNestingLevel(mlir::Region *Region,
                                 CallbackT Callback,
                                 unsigned NestingLevel = 0) {
  for (mlir::Block &Block : *Region) {
    for (mlir::Operation &Op : Block) {
      // Post-order walk through the nested regions, incrementing the nesting
      // level for inner region.
      for (mlir::Region &NestedRegion : Op.getRegions())
        walkWithNestingLevel(&NestedRegion, Callback, NestingLevel + 1);

      // Then call the callback on the operation itself.
      Callback(&Op, NestingLevel);
    }
  }
}

// Stores the position and nesting level for each local variable definition.
struct LocalVariableLocation {
  BlockPosition Position;
  unsigned NestingLevel = 0;
};

// Merges the current location for a variable definition with the new use
// position.
static void updateVariableLocation(LocalVariableLocation &VarLoc,
                                   BlockPosition NewPosition,
                                   unsigned NewLevel) {
  mlir::Region *CurrentRegion = VarLoc.Position.Block->getParent();
  mlir::Region *NewPosRegion = NewPosition.Block->getParent();

  unsigned CurrentLevel = VarLoc.NestingLevel;

  // If the new position is at a lower nesting level, we surely need to go up
  // from the current region at least until the levels are equal, in order to
  // find the common ancestor.
  while (CurrentLevel > NewLevel) {
    CurrentRegion = CurrentRegion->getParentRegion();
    --CurrentLevel;
  }

  // Conversely, if the new position is at a higher nesting level, we need to go
  // up from it until we reach the current region's level.
  while (NewLevel > CurrentLevel) {
    NewPosRegion = NewPosRegion->getParentRegion();
    --NewLevel;
  }

  // We can now check if the current region is identical to the parent region of
  // the new position.
  while (CurrentRegion != NewPosRegion) {
    // If they are not the same, we need to go up from both of them until we
    // reach the common ancestor region.
    CurrentRegion = CurrentRegion->getParentRegion();
    NewPosRegion = NewPosRegion->getParentRegion();
    --CurrentLevel;
  }

  // Now we can find the common ancestor operation that is the closest to the
  // current variable location by walking up the tree until we match the nesting
  // level.
  mlir::Operation *CurrentOp = VarLoc.Position.getOperation();
  for (unsigned i = 0; i < (VarLoc.NestingLevel - CurrentLevel); ++i)
    CurrentOp = CurrentOp->getParentOp();

  revng_assert(not mlir::isa<mlir::ModuleOp>(CurrentOp));

  // Update the variable location to the newfound common ancestor.
  VarLoc.Position = BlockPosition::get(CurrentOp);
  VarLoc.NestingLevel = CurrentLevel;
}

struct TightenVariableScopePass
  : clift::impl::CliftTightenVariableScopesBase<TightenVariableScopePass> {
  void runOnOperation() override {
    FunctionOp Function = getOperation();

    // Store the function's local variables in a map, associated with their
    // optimal position in the MLIR tree.
    llvm::MapVector<LocalVariableOp, LocalVariableLocation> Locals;

    auto WalkCallback = [&](mlir::Operation *Op, unsigned OpNestingLevel) {
      // For each operand of the operation, we check if it is a local variable.
      for (mlir::Value Operand : Op->getOperands()) {
        auto Local = Operand.getDefiningOp<LocalVariableOp>();

        // If the defining op is not a local variable, we can skip it.
        if (not Local)
          continue;

        // Local variables with an initializer cannot be moved.
        if (not Local.getInitializer().empty())
          continue;

        // TODO: Once LLVM has been upgraded, this could use try_emplace.
        auto [Iterator,
              Inserted] = Locals.insert({ Local, LocalVariableLocation() });

        if (Inserted) {
          // If the local variable was not already in the map, mark its optimal
          // position as right before the the current user, or right before the
          // parent of the current user if it is an expression operation.
          if (mlir::isa<ExpressionOpInterface>(Op)) {
            Iterator->second.Position = BlockPosition::get(Op->getParentOp());
            Iterator->second.NestingLevel = OpNestingLevel - 1;
          } else {
            Iterator->second.Position = BlockPosition::get(Op);
            Iterator->second.NestingLevel = OpNestingLevel;
          }
        } else {
          // If the local variable is already in the map, update its location to
          // find the common ancestor position that covers the new user.
          auto OpPosition = BlockPosition::get(Op);
          updateVariableLocation(Iterator->second, OpPosition, OpNestingLevel);
        }
      }
    };

    // Walk the function body to identify local variable uses, along with their
    // nesting levels.
    walkWithNestingLevel(&Function.getBody(), WalkCallback);

    // Move each local variable to its optimal position.
    for (const auto &[Local, Pair] : Locals) {
      // Get the target operation where we want to insert the variable.
      mlir::Operation *TargetOp = Pair.Position.getOperation();

      revng_assert(not mlir::isa<FunctionOp>(TargetOp));

      // Check if moving would be a no-op (already in the right place), and
      // otherwise Move the local variable declaration to the new position.
      if (TargetOp->getPrevNode() != Local)
        Local->moveBefore(TargetOp);
    }
  }
};

} // namespace

PassPtr<FunctionOp> clift::createVariableScopeTighteningPass() {
  return std::make_unique<TightenVariableScopePass>();
}
