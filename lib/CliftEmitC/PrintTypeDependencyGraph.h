//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/PostOrderIterator.h"

#include "revng/CliftEmitC/CEmitter.h"

#include "TypeDependencyGraph.h"

inline Logger TypePrinterLog{ "clift-type-definition-printer" };

inline void
emitTypeTree(mlir::clift::CEmitter &Emitter,
             const mlir::clift::TypeDependencyNode &Root,
             std::set<const mlir::clift::TypeDependencyNode *> &Emitted) {
  revng_log(TypePrinterLog,
            "Starting a post order visit from:" << getNodeLabel(&Root));

  size_t NodesEmittedBefore = Emitted.size();
  for (const auto *Node : llvm::post_order_ext(&Root, Emitted)) {
    LoggerIndent PostOrderIndent{ TypePrinterLog };
    revng_log(TypePrinterLog, "visiting: " << getNodeLabel(Node));

    mlir::clift::DefinedType Definition = Node->T;

    if (Node->isDeclaration()) {
      revng_log(TypePrinterLog, "Declaration");
      Emitter.emitTypeDeclaration(Definition);
    } else {
      revng_log(TypePrinterLog, "Definition");
      revng_assert(Node->isDefinition());
      revng_assert(not Emitter.isDeclarationTheSameAsDefinition(Definition));
      Emitter.emitTypeDefinition(Definition);
    }
  }

  if (NodesEmittedBefore != Emitted.size())
    Emitter.tokenEmitter().emitNewline();

  revng_log(TypePrinterLog, "Root is done: " << getNodeLabel(&Root));
}
