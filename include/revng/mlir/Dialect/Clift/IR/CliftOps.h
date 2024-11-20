#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "revng/mlir/Dialect/Clift/IR/Clift.h"
#include "revng/mlir/Dialect/Clift/IR/CliftAttributes.h"
#include "revng/mlir/Dialect/Clift/IR/CliftEnums.h"
#include "revng/mlir/Dialect/Clift/IR/CliftInterfaces.h"
#include "revng/mlir/Dialect/Clift/IR/CliftOpTraits.h"
#include "revng/mlir/Dialect/Clift/IR/CliftTypes.h"

namespace mlir {
namespace clift::impl {

bool verifyStatementRegion(Region &R);
bool verifyExpressionRegion(Region &R, bool Required);

bool verifyPrimitiveTypeOf(ValueType Type, PrimitiveKind Kind);

mlir::Type removeCliftConst(mlir::Type Type);

ParseResult parseCliftOpTypes(OpAsmParser &Parser,
                              Type *Result,
                              llvm::ArrayRef<Type *> Arguments);

void printCliftOpTypes(OpAsmPrinter &Printer,
                       Type Result,
                       llvm::ArrayRef<Type> Arguments);

mlir::LogicalResult verifyUnaryIntegerMutationOp(Operation *Op);

} // namespace clift::impl

template<std::same_as<Type>... Ts>
ParseResult
parseCliftOpTypes(OpAsmParser &Parser, Type &Result, Ts &...Arguments) {
  static_assert(sizeof...(Ts) > 0);
  return clift::impl::parseCliftOpTypes(Parser, &Result, { &Arguments... });
}

template<std::same_as<Type>... Ts>
ParseResult parseCliftOpOperandTypes(OpAsmParser &Parser, Ts &...Arguments) {
  static_assert(sizeof...(Ts) > 0);
  return clift::impl::parseCliftOpTypes(Parser, nullptr, { &Arguments... });
}

template<std::same_as<Type>... Ts>
void printCliftOpTypes(OpAsmPrinter &Printer,
                       Operation *Op,
                       Type Result,
                       Ts... Arguments) {
  static_assert(sizeof...(Ts) > 0);
  clift::impl::printCliftOpTypes(Printer, Result, { Arguments... });
}

template<std::same_as<Type>... Ts>
void printCliftOpOperandTypes(OpAsmPrinter &Printer,
                              Operation *Op,
                              Ts... Arguments) {
  static_assert(sizeof...(Ts) > 0);
  clift::impl::printCliftOpTypes(Printer, nullptr, { Arguments... });
}

} // namespace mlir

// This include should stay here for correct build procedure
#define GET_OP_CLASSES
#include "revng/mlir/Dialect/Clift/IR/CliftOps.h.inc"
