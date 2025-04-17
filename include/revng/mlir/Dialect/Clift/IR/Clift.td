//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#ifndef MLIR_CLIFT
#define MLIR_CLIFT
// Include the definition of the necessary tablegen constructs for defining our
// dialect
include "mlir/IR/DialectBase.td"

// Here is a simple definition of a dialect
def Clift_Dialect : Dialect {
  let summary = "Clift dialect.";
  let description = [{
    very important
  }];

  // This is the namespace of the dialect. It is used to encapsulate the
  // sub-components of the dialect, such as operations ("my_dialect.foo")
  let name = "clift";

  // The C++ namespace that the dialect, and its sub-components, get placed in
  let cppNamespace = "mlir::clift";
  let useDefaultTypePrinterParser = 0;
  let useDefaultAttributePrinterParser = 0;
  let useFoldAPI = kEmitFoldAdaptorFolder;

  let extraClassDeclaration = [{
    void registerTypes();
    void registerAttributes();
    void registerOperations();

    mlir::Attribute parseAttribute(mlir::DialectAsmParser &Parser,
                                   mlir::Type Type) const override;

    void printAttribute(mlir::Attribute Attr,
                        mlir::DialectAsmPrinter &Printer) const override;

    mlir::Type parseType(mlir::DialectAsmParser &Parser) const override;

    void printType(mlir::Type Type,
                   mlir::DialectAsmPrinter &Printer) const override;
  }];
}
#endif
