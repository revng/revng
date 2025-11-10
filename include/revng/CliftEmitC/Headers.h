#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/PTML/CTokenEmitter.h"

namespace mlir {

class ModuleOp;

namespace clift {

class CEmitter;

// TODO: should this header become a class?

ptml::CTokenEmitter::Scope emitHeaderPrologue(ptml::CTokenEmitter &PTML);
ptml::CTokenEmitter::Scope emitHeaderPrologue(mlir::clift::CEmitter &Emitter);
void emitIncludes();

void emitAttributes(ptml::CTokenEmitter &PTML);
void emitPrimitiveTypes(ptml::CTokenEmitter &PTML);

void emitModelTypes(CEmitter &Emitter, const mlir::ModuleOp &Module);
void emitModelFunctions(CEmitter &PTML, const mlir::ModuleOp &Module);
void emitDynamicModelFunctions(CEmitter &PTML, const mlir::ModuleOp &Module);
void emitSegments(CEmitter &PTML, const mlir::ModuleOp &Module);
void emitHelpers(CEmitter &PTML, const mlir::ModuleOp &Module);

inline void emitAttributeHeader(ptml::CTokenEmitter &PTML) {
  auto Scope = emitHeaderPrologue(PTML);

  emitAttributes(PTML);
}
inline void emitPrimitiveHeader(ptml::CTokenEmitter &PTML) {
  auto Scope = emitHeaderPrologue(PTML);

  emitPrimitiveTypes(PTML);
}
inline void emitModelHeader(mlir::clift::CEmitter &Emitter,
                            const mlir::ModuleOp &Module) {
  auto Scope = emitHeaderPrologue(Emitter);

  // TODO: split the following into separate headers.

  emitModelTypes(Emitter, Module);
  emitModelFunctions(Emitter, Module);
  emitDynamicModelFunctions(Emitter, Module);
  emitSegments(Emitter, Module);
}
inline void emitHelperHeader(mlir::clift::CEmitter &Emitter,
                             const mlir::ModuleOp &Module) {
  auto Scope = emitHeaderPrologue(Emitter);

  emitHelpers(Emitter, Module);
}

} // namespace clift
} // namespace mlir
