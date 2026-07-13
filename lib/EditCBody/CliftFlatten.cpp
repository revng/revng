//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <vector>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include "revng/Clift/Clift.h"
#include "revng/Clift/CliftOpHelpers.h"
#include "revng/Clift/Helpers.h"
#include "revng/Support/Assert.h"

#include "CliftFlatten.h"
#include "Statements.h"

using namespace revng::editcbody;

namespace {

StatementKind cliftStatementKind(mlir::Operation *Op) {
  using namespace clift;
  if (mlir::isa<LocalVariableOp>(Op))
    return StatementKind::LocalVariableDeclaration;
  else if (mlir::isa<ExpressionStatementOp>(Op))
    return StatementKind::Expression;
  else if (mlir::isa<ReturnOp>(Op))
    return StatementKind::Return;
  else if (mlir::isa<IfOp>(Op))
    return StatementKind::If;
  else if (mlir::isa<WhileOp>(Op))
    return StatementKind::While;
  else if (mlir::isa<DoWhileOp>(Op))
    return StatementKind::DoWhile;
  else if (mlir::isa<ForOp>(Op))
    return StatementKind::For;
  else if (mlir::isa<SwitchOp>(Op))
    return StatementKind::Switch;
  // TODO: once clift emits plain `break` and `continue`, handle them.
  else if (mlir::isa<GotoOp, BreakToOp, ContinueToOp>(Op))
    return StatementKind::Goto;
  else if (mlir::isa<AssignLabelOp>(Op))
    return StatementKind::Label;
  else
    revng_abort("Unsupported Clift statement");
}

bool hasFallthrough(mlir::Region &Region) {
  return not clift::getLastNoFallthroughStatement(Region);
}

} // namespace

/// Flatten a Clift region into an ordered list of statements matching the
/// emitted C, iteratively (without recursion) via an explicit work list. Each
/// item is either a region to expand into its statements, an operation to turn
/// into a node plus its nested regions, or a synthesized marker.
void revng::editcbody::flattenCliftRegion(mlir::Region &Root,
                                          std::vector<CliftStatement> &Output) {
  using namespace clift;

  // A region item has a non-null Region; an operation item has a non-null
  // Operation; a marker item has neither and carries only its Kind.
  struct WorkItem {
    mlir::Region *Region = nullptr;
    mlir::Operation *Operation = nullptr;
    StatementKind Marker = {};
  };

  auto RegionItem = [](mlir::Region &Region) {
    return WorkItem{ .Region = &Region };
  };
  auto MarkerItem = [](StatementKind Marker) {
    return WorkItem{ .Marker = Marker };
  };

  llvm::SmallVector<WorkItem> Stack = { RegionItem(Root) };

  while (not Stack.empty()) {
    WorkItem Item = Stack.pop_back_val();

    if (Item.Region != nullptr) {
      // Expand a region into its visible statements, preserving their order by
      // pushing them onto the stack in reverse.
      llvm::SmallVector<mlir::Operation *> Operations;
      for (mlir::Operation &Operation : Item.Region->getOps())
        if (not mlir::isa<MakeLabelOp, RequireOp>(&Operation))
          Operations.push_back(&Operation);
      for (mlir::Operation *Operation : llvm::reverse(Operations))
        Stack.push_back({ .Operation = Operation });
    } else if (Item.Operation == nullptr) {
      // A synthesized marker becomes a node with no operation.
      Output.push_back({ Item.Marker, nullptr });
    } else if (auto Block = mlir::dyn_cast<BlockStatementOp>(Item.Operation)) {
      // Blocks are transparent: expand them without producing a node.
      Stack.push_back(RegionItem(Block.getBlock()));
    } else {
      mlir::Operation *Operation = Item.Operation;
      Output.push_back({ cliftStatementKind(Operation), Operation });

      // Collect the op's nested work in emission order, then push it in
      // reverse.
      llvm::SmallVector<WorkItem> Children;
      if (auto If = mlir::dyn_cast<IfOp>(Operation)) {
        Children.push_back(RegionItem(If.getThen()));
        if (not If.getElse().empty())
          Children.push_back(RegionItem(If.getElse()));
      } else if (auto Switch = mlir::dyn_cast<SwitchOp>(Operation)) {
        auto AppendCase = [&](StatementKind Marker, mlir::Region &Region) {
          Children.push_back(MarkerItem(Marker));
          Children.push_back(RegionItem(Region));
          if (hasFallthrough(Region))
            Children.push_back(MarkerItem(StatementKind::Break));
        };
        for (unsigned I = 0, Count = Switch.getNumCases(); I < Count; ++I)
          AppendCase(StatementKind::Case, Switch.getCaseRegion(I));
        if (Switch.hasDefaultCase())
          AppendCase(StatementKind::Default, Switch.getDefaultCaseRegion());
      } else if (auto Loop = mlir::dyn_cast<LoopOpInterface>(Operation)) {
        Children.push_back(RegionItem(Loop.getBody()));
        // The continue label closes the body; the break label follows the loop.
        if (Loop.getContinueLabel())
          Children.push_back(MarkerItem(StatementKind::Label));
        if (Loop.getBreakLabel())
          Children.push_back(MarkerItem(StatementKind::Label));
      }

      for (const WorkItem &Child : llvm::reverse(Children))
        Stack.push_back(Child);
    }
  }
}
