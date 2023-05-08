//
// Copyright (c) rev.ng Labs Srl. See LICENSE.md for details.
//

#include "revng-c/mlir/Dialect/Clift/IR/CliftTypes.h"

#define GET_TYPEDEF_CLASSES
#include "revng-c/mlir/Dialect/Clift/IR/CliftOpsTypes.cpp.inc"

void mlir::clift::CliftDialect::registerTypes() {
  addTypes</* Include the auto-generated clift types */
#define GET_TYPEDEF_LIST
#include "revng-c/mlir/Dialect/Clift/IR/CliftOpsTypes.cpp.inc"
           /* End of types list */>();
}

/// Parse a type registered to this dialect
::mlir::Type
mlir::clift::CliftDialect::parseType(::mlir::DialectAsmParser &Parser) const {
  ::llvm::SMLoc typeLoc = Parser.getCurrentLocation();
  ::llvm::StringRef Mnemonic;
  ::mlir::Type GenType;
  auto ParseResult = generatedTypeParser(Parser, &Mnemonic, GenType);
  if (ParseResult.has_value())
    return GenType;

  Parser.emitError(typeLoc) << "unknown  type `" << Mnemonic << "` in dialect `"
                            << getNamespace() << "`";
  return {};
}

/// Print a type registered to this dialect
void mlir::clift::CliftDialect::printType(::mlir::Type Type,
                                          ::mlir::DialectAsmPrinter &Printer)
  const {
  if (::mlir::succeeded(generatedTypePrinter(Type, Printer)))
    return;
}
