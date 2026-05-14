//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/RegionUtils.h"

#include "revng/Clift/Clift.h"
#include "revng/Clift/CliftOpHelpers.h"
#include "revng/CliftTransforms/Passes.h"

namespace clift {
#define GEN_PASS_DEF_CLIFTVARIABLEINITIALIZERHOISTING
#include "revng/CliftTransforms/Passes.h.inc"
} // namespace clift

using namespace clift;

namespace {

static bool visitExpressionStatement(ExpressionStatementOp Statement,
                                     bool HoistInitializer) {
  mlir::Region &Expression = Statement.getExpression();
  auto Yield = getYieldOp(Expression);

  auto Assignment = Yield.getValue().getDefiningOp<AssignOp>();
  if (not Assignment)
    return false;

  auto Local = Assignment.getLhs().getDefiningOp<LocalVariableOp>();
  if (not Local)
    return false;

  mlir::Region &Initializer = Local.getInitializer();
  if (not Initializer.empty())
    return false;

  mlir::Block *Block = Statement->getBlock();
  if (Local->getBlock() != Block)
    return false;

  bool HasIntercedingLocals = false;

  // Scan from the local variable until the assignment or the end of the block,
  // whichever comes first. If the end of the block or any non-local-variable
  // operation is reached, or if the interceding local variable operation
  // already has a non-empty initializer, then hoisting is not possible.
  for (auto I = Local->getIterator(); I != Statement->getIterator(); ++I) {
    if (I == Block->end())
      return false;

    auto IntercedingLocal = mlir::dyn_cast<LocalVariableOp>(&*I);
    if (not IntercedingLocal or not IntercedingLocal.getInitializer().empty())
      return false;

    HasIntercedingLocals = true;
  }

  if (HoistInitializer) {
    // If there are interceding local variables between the assignment and the
    // assigned local, move the assigned local to just before the assignment.
    if (HasIntercedingLocals)
      Local->moveBefore(Statement);

    // Move expression statement block into the variable initializer.
    Initializer.getBlocks().splice(Initializer.getBlocks().end(),
                                   Expression.getBlocks());

    // Replace the expression root with the assigned value operand.
    Yield->setOperand(0, Assignment.getRhs());

    // The assignment operation is no longer needed.
    Assignment->erase();

    // The expression statement operation is no longer needed.
    Statement->erase();

    // Add an initializer block argument in case it's needed by the assignment.
    if (Initializer.args_empty())
      Initializer.addArgument(Local.getType(), Local->getLoc());

    auto Argument = Initializer.getArgument(0);

    // Replace all uses of the variable within the initialiser region with the
    // initialiser block argument.
    mlir::replaceAllUsesInRegionWith(Local, Argument, Initializer);

    // If the initialiser block argument is still unused, it can be removed.
    if (Argument.use_empty())
      Initializer.eraseArgument(0);
  }

  return true;
}

template<typename T>
using PassBase = clift::impl::CliftVariableInitializerHoistingBase<T>;

struct VariableInitializerHoistingPass
  : PassBase<VariableInitializerHoistingPass> {

  void runOnOperation() override {
    llvm::SmallVector<ExpressionStatementOp> Statements;

    getOperation()->walk([&](ExpressionStatementOp Statement) {
      if (visitExpressionStatement(Statement, /*HoistInitializer=*/false))
        Statements.push_back(Statement);
    });

    for (ExpressionStatementOp Statement : Statements) {
      bool Hoisted = visitExpressionStatement(Statement,
                                              /*HoistInitializer=*/true);
      revng_assert(Hoisted);
    }
  }
};

} // namespace

PassPtr<FunctionOp> clift::createVariableInitializerHoistingPass() {
  return std::make_unique<VariableInitializerHoistingPass>();
}
