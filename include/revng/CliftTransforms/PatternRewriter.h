#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "mlir/IR/PatternMatch.h"

namespace clift {

//===------------------ Future PatternRewriter functions ------------------===//

void inlineBlockBefore(mlir::PatternRewriter &Rewriter,
                       mlir::Block *Src,
                       mlir::Block *Dst,
                       mlir::Block::iterator Pos);

//===------------------------------- Helpers ------------------------------===//

inline void inlineRegionAtEnd(mlir::PatternRewriter &Rewriter,
                              mlir::Region &Source,
                              mlir::Region &Destination) {
  Rewriter.inlineRegionBefore(Source, Destination, Destination.end());
}

inline void clearRegion(mlir::PatternRewriter &Rewriter, mlir::Region &Region) {
  while (not Region.empty())
    Rewriter.eraseBlock(&Region.front());
}

inline void setOperandValue(mlir::PatternRewriter &Rewriter,
                            mlir::OpOperand &Operand,
                            mlir::Value Value) {
  Rewriter.updateRootInPlace(Operand.getOwner(), [&]() { Operand.set(Value); });
}

} // namespace clift
