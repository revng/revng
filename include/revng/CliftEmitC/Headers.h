#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "mlir/IR/BuiltinOps.h"

#include "revng/CliftEmitC/Configuration.h"
#include "revng/PTML/CTokenEmitter.h"

namespace mlir::clift {

void emitHeaderPrologue(ptml::CTokenEmitter &Tokens);

void emitCommonIncludes(ptml::CTokenEmitter &Tokens);
void emitTypes(ptml::CTokenEmitter &Tokens,
               const TargetCImplementation &Target,
               mlir::ModuleOp Module,
               TypeEmitterConfiguration Configuration);
void emitFunctions(ptml::CTokenEmitter &Tokens,
                   const TargetCImplementation &Target,
                   mlir::ModuleOp Module);
void emitDynamicFunctions(ptml::CTokenEmitter &Tokens,
                          const TargetCImplementation &Target,
                          mlir::ModuleOp Module);
void emitSegments(ptml::CTokenEmitter &Tokens,
                  const TargetCImplementation &Target,
                  mlir::ModuleOp Module);
inline void emitTypeAndGlobalHeader(ptml::CTokenEmitter &Tokens,
                                    const TargetCImplementation &Target,
                                    mlir::ModuleOp Module,
                                    TypeEmitterConfiguration Configuration) {
  // TODO: emit header location definition on the scope tag so that ctrl+click
  //       on includes (references) leads to this file.
  ptml::CTokenEmitter::Scope
    Scope = Tokens.enterScope(ptml::CTokenEmitter::ScopeKind::Basic, 0);

  emitHeaderPrologue(Tokens);

  emitCommonIncludes(Tokens);

  // TODO: split the following into separate headers.

  emitTypes(Tokens, Target, Module, Configuration);
  emitFunctions(Tokens, Target, Module);
  emitDynamicFunctions(Tokens, Target, Module);
  emitSegments(Tokens, Target, Module);
}

void emitHelpers(ptml::CTokenEmitter &Tokens,
                 const TargetCImplementation &Target,
                 std::vector<mlir::ModuleOp> &Modules);
inline void emitHelperHeader(ptml::CTokenEmitter &Tokens,
                             const TargetCImplementation &Target,
                             std::vector<mlir::ModuleOp> &Modules) {
  // TODO: emit header location definition on the scope tag so that ctrl+click
  //       on includes (references) leads to this file.
  ptml::CTokenEmitter::Scope
    Scope = Tokens.enterScope(ptml::CTokenEmitter::ScopeKind::Basic, 0);

  emitHeaderPrologue(Tokens);
  emitHelpers(Tokens, Target, Modules);
}

} // namespace mlir::clift
