#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "mlir/Bytecode/BytecodeImplementation.h"

#include "revng/Clift/CliftTypes.h"

namespace mlir::clift {

mlir::Attribute readAttr(mlir::DialectBytecodeReader &Reader);

mlir::LogicalResult writeAttr(mlir::Attribute Attr,
                              mlir::DialectBytecodeWriter &Writer);

mlir::Type readType(mlir::DialectBytecodeReader &Reader);

mlir::LogicalResult writeType(mlir::Type Type,
                              mlir::DialectBytecodeWriter &Writer);

clift::StructType readStructDefinition(mlir::DialectBytecodeReader &Reader);

void writeStructDefinition(clift::StructType Type,
                           mlir::DialectBytecodeWriter &Writer);

clift::UnionType readUnionDefinition(mlir::DialectBytecodeReader &Reader);

void writeUnionDefinition(clift::UnionType Type,
                          mlir::DialectBytecodeWriter &Writer);

} // namespace mlir::clift
