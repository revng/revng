#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Clift/CliftOpHelpers.h"
#include "revng/CliftTransforms/PatternRewriter.h"

namespace clift {

inline IntegerType getBooleanType(mlir::MLIRContext *Context) {
  return IntegerType::get(Context, IntegerKind::Signed, /*Size=*/1);
}

/// Transform the expression tree in \p Region with a new expression tree.
///
/// \p Transform is used to compute the new expression tree root. It is invoked
/// with the expression tree root value of \p Region.
void transformExpression(mlir::PatternRewriter &Rewriter,
                         mlir::Region &Region,
                         llvm::function_ref<mlir::Value(mlir::Value)>
                           Transform);

/// Move the expression tree from \p SourceRegion into \p TargetRegion and
/// merge the two expressions together into a single expression at the root of
/// the resulting expression tree in \p TargetRegion.
///
/// \p Merge is used to compute the new expression tree root. It is invoked with
/// 1. the expression tree root value of \p SourceRegion, and
/// 2. the expression tree root value of \p TargetRegion.
void mergeExpressionInto(mlir::PatternRewriter &Rewriter,
                         mlir::Region &SourceRegion,
                         mlir::Region &TargetRegion,
                         llvm::function_ref<mlir::Value(mlir::Value,
                                                        mlir::Value)> Merge);

/// Returns true if the condition described the specified region is a non-zero
/// ImmediateOp.
bool isTriviallyTrue(mlir::Region &Condition);

/// Inverts the condition expression in the specified region.
void invertBooleanExpression(mlir::PatternRewriter &Rewriter,
                             mlir::Location Loc,
                             mlir::Region &Region);

/// Inverts the condition and the two statement regions of the specified if
/// operation.
void invertIfStatement(mlir::PatternRewriter &Rewriter, IfOp If);

/// Returns the position of the next statement executed starting from the
/// specified position, or the end of a block if the end of the function or of a
/// loop is encountered.
BlockPosition getFallthroughTarget(BlockPosition Position);

} // namespace clift
