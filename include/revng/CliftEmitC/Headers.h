#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "mlir/IR/BuiltinOps.h"

#include "revng/Clift/CliftTypeInterfaces.h"
#include "revng/CliftEmitC/Configuration.h"
#include "revng/PTML/CTokenEmitter.h"

namespace model {
class Binary;
}

void emitCommonIncludes(ptml::CTokenEmitter &Tokens,
                        const CDataModel &DataModel);

void emitTypes(ptml::CTokenEmitter &Tokens,
               mlir::ModuleOp Module,
               TypeEmitterConfiguration Configuration);

void emitTypeAndGlobalHeader(ptml::CTokenEmitter &Tokens,
                             mlir::ModuleOp Module,
                             TypeEmitterConfiguration Configuration);

void emitHelperHeader(ptml::CTokenEmitter &Tokens,
                      llvm::ArrayRef<mlir::ModuleOp> Modules,
                      const model::Binary &Binary);

void emitSingleTypeDefinition(ptml::CTokenEmitter &Tokens,
                              const CDataModel &DataModel,
                              clift::DefinedType Type,
                              TypeEmitterConfiguration Configuration = {});
