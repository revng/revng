//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#ifndef MLIR_CLIFT_TYPE_INTERFACES
#define MLIR_CLIFT_TYPE_INTERFACES

// The MLIR tablegen tool emits declarations in alphabetical order, causing
// incorrect emission of interface hierarchies. The type interface declarations
// are split up into separate files in order to emit in the correct order:
//
// Tablegen: ValueType -> DefinedType -> ClassType
// Emission: ClassType    DefinedType    ValueType

include "revng/Clift/CliftTypeInterfacesClass.td"

#endif
