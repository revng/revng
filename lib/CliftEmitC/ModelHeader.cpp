//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "mlir/IR/BuiltinOps.h"

#include "revng/Clift/ModuleVisitor.h"
#include "revng/CliftEmitC/CEmitter.h"
#include "revng/CliftEmitC/Headers.h"
#include "revng/PTML/CTokenEmitter.h"
#include "revng/Pipeline/Location.h"
#include "revng/Pipes/Ranks.h"

#include "PrintTypeDependencyGraph.h"
#include "TypeDependencyGraph.h"

void mlir::clift::emitModelTypes(mlir::clift::CEmitter &Emitter,
                                 const mlir::ModuleOp &Module) {
  ptml::CTokenEmitter &PTML = Emitter.tokenEmitter();
  PTML.emitComment("Types", ptml::CTokenEmitter::CommentKind::Category);
  PTML.emitNewline();

  auto Graph = TypeDependencyGraph::makeModelGraph(Module,
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
}

void mlir::clift::emitModelFunctions(mlir::clift::CEmitter &Emitter,
                                     const mlir::ModuleOp &Module) {
  bool FirstFunction = true;

  Module->walk([&Emitter, &FirstFunction](mlir::clift::FunctionOp Function) {
    ptml::CTokenEmitter &PTML = Emitter.tokenEmitter();

    if (pipeline::locationFromString(revng::ranks::Function,
                                     Function.getHandle())) {
      if (FirstFunction) {
        PTML.emitComment("Functions",
                         ptml::CTokenEmitter::CommentKind::Category);
        PTML.emitNewline();
        FirstFunction = false;
      }

      Emitter.emitFunctionPrototype(Function);
      PTML.emitPunctuator(ptml::CTokenEmitter::Punctuator::Semicolon);
      PTML.emitNewline();
      PTML.emitNewline();
    }
  });
}

void mlir::clift::emitDynamicModelFunctions(mlir::clift::CEmitter &Emitter,
                                            const mlir::ModuleOp &Module) {
  bool FirstFunction = true;

  Module->walk([&Emitter, &FirstFunction](mlir::clift::FunctionOp Function) {
    ptml::CTokenEmitter &PTML = Emitter.tokenEmitter();

    if (pipeline::locationFromString(revng::ranks::DynamicFunction,
                                     Function.getHandle())) {
      if (FirstFunction) {
        PTML.emitComment("Imported Dynamic Functions",
                         ptml::CTokenEmitter::CommentKind::Category);
        PTML.emitNewline();
        FirstFunction = false;
      }

      Emitter.emitFunctionPrototype(Function);
      PTML.emitPunctuator(ptml::CTokenEmitter::Punctuator::Semicolon);
      PTML.emitNewline();
      PTML.emitNewline();
    }
  });
}

void mlir::clift::emitSegments(mlir::clift::CEmitter &Emitter,
                               const mlir::ModuleOp &Module) {
  ptml::CTokenEmitter &PTML = Emitter.tokenEmitter();

  PTML.emitComment("Segments", ptml::CTokenEmitter::CommentKind::Category);
  PTML.emitNewline();
  PTML.emitComment("WIP");
}
