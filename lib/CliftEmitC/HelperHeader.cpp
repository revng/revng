//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "mlir/IR/BuiltinOps.h"

#include "revng/CliftEmitC/CEmitter.h"
#include "revng/CliftEmitC/Headers.h"
#include "revng/PTML/CTokenEmitter.h"
#include "revng/Pipeline/Location.h"
#include "revng/Pipes/Ranks.h"

#include "PrintTypeDependencyGraph.h"
#include "TypeDependencyGraph.h"

static void printHelperPrototype(mlir::clift::CEmitter &Emitter,
                                 mlir::clift::FunctionOp Function) {
  Emitter.tokenEmitter().emitComment("TODO: print "
                                     + Function.getHandle().str());
}

void mlir::clift::emitHelpers(mlir::clift::CEmitter &Emitter,
                              const mlir::ModuleOp &Module) {
  ptml::CTokenEmitter &PTML = Emitter.tokenEmitter();
  PTML.emitComment("Types", ptml::CTokenEmitter::CommentKind::Category);
  PTML.emitNewline();

  // WIP: this might not be a good idea after all, look into this again and
  // possibly drop this.
  auto Graph = TypeDependencyGraph::makeHelperGraph(Module,
                                                    Emitter.Target.PointerSize);

  // In order to improve the printing order, do the visit it in two parts:
  // first only start from nodes without any successors (real roots),
  // only then, resolve potential loops by starting from arbitrary nodes.
  std::set<const TypeDependencyNode *> Emitted;
  for (const auto *Root : Graph.nodes())
    if (not Root->predecessorCount())
      emitTypeTree(Emitter, *Root, Emitted);
  for (const auto *Root : Graph.nodes())
    emitTypeTree(Emitter, *Root, Emitted);
  revng_assert(Graph.size() == Emitted.size());

  PTML.emitNewline();
  PTML.emitComment("Functions", ptml::CTokenEmitter::CommentKind::Category);
  PTML.emitNewline();

  Module->walk([&Emitter](mlir::clift::FunctionOp Function) {
    if (pipeline::locationFromString(revng::ranks::HelperFunction,
                                     Function.getHandle()))
      printHelperPrototype(Emitter, Function);
  });

  PTML.emitNewline();
}
