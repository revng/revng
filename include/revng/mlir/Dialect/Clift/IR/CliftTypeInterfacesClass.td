//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#ifndef MLIR_CLIFT_TYPE_INTERFACES_CLASS
#define MLIR_CLIFT_TYPE_INTERFACES_CLASS

include "revng/mlir/Dialect/Clift/IR/CliftTypeInterfacesDefined.td"

def Clift_ClassType : Clift_TypeInterface<"ClassType", [Clift_DefinedType]> {
  let methods = [
    InterfaceMethod<
      "Returns true if the type is not incomplete.",
      "bool", "isComplete", (ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return $_type.isComplete();
      }]
    >,
    InterfaceMethod<
      "Returns the field attributes of this class type",
      "llvm::ArrayRef<mlir::clift::FieldAttr>", "getFields", (ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return $_type.getFields();
      }]
    >,
  ];
}

#endif

