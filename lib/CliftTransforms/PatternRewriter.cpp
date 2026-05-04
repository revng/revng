//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/CliftTransforms/PatternRewriter.h"

void clift::inlineBlockBefore(mlir::PatternRewriter &Rewriter,
                              mlir::Block *Src,
                              mlir::Block *Dst,
                              mlir::Block::iterator Pos) {
  mlir::Operation *SrcOp = Src->getParentOp();
  mlir::Operation *DstOp = Dst->getParentOp();

  bool UpdateSrcOp = SrcOp != nullptr;
  bool UpdateDstOp = DstOp != nullptr and DstOp != SrcOp;

  if (UpdateSrcOp)
    Rewriter.startRootUpdate(SrcOp);
  if (UpdateDstOp)
    Rewriter.startRootUpdate(DstOp);

  Dst->getOperations().splice(Pos, Src->getOperations());

  if (UpdateDstOp)
    Rewriter.finalizeRootUpdate(DstOp);
  if (UpdateSrcOp)
    Rewriter.finalizeRootUpdate(SrcOp);
}
