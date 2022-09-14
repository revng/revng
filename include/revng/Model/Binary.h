#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/SmallString.h"

#include "revng/ADT/MutableSet.h"
#include "revng/ADT/SortedVector.h"
#include "revng/ADT/UpcastablePointer.h"
#include "revng/ADT/UpcastablePointer/YAMLTraits.h"
#include "revng/Model/ABI.h"
#include "revng/Model/DynamicFunction.h"
#include "revng/Model/Function.h"
#include "revng/Model/FunctionAttribute.h"
#include "revng/Model/Register.h"
#include "revng/Model/Segment.h"
#include "revng/Model/Type.h"
#include "revng/Model/VerifyHelper.h"
#include "revng/Support/MetaAddress.h"
#include "revng/Support/MetaAddress/YAMLTraits.h"
#include "revng/Support/YAMLTraits.h"
#include "revng/TupleTree/TupleTree.h"
#include "revng/TupleTree/TupleTreeDiff.h"

/* TUPLE-TREE-YAML
name: Binary
doc: |-
  Data structure representing the whole binary.
  This is the entry point of the model.
  It contains the type system (`Types`), the list of functions (`Functions`),
  loading information (`Segments`) and more.
type: struct
fields:
  - name: Functions
    doc: List of the function present in the binary.
    sequence:
      type: SortedVector
      elementType: Function
    optional: true
  - name: ImportedDynamicFunctions
    doc: List of functions imported from dynamic libraries (`.so`, `.dll`).
    sequence:
      type: SortedVector
      elementType: DynamicFunction
    optional: true
  - name: Architecture
    doc: The architecture for this binary.
    type: Architecture
    optional: true
  - name: DefaultABI
    doc: |-
      The default ABI to adopt for analysis purposes.
    type: ABI
    optional: true
  - name: DefaultPrototype
    doc: |-
      The default function prototype to adopt for functions that do not provide
      it explicitly.
    reference:
      pointeeType: Type
      rootType: Binary
    optional: true
  - name: Segments
    doc: |-
      A list of `Segment`.
      Basically, these represent instructions on what part of the raw binary
      needs to be loaded at which address.
    sequence:
      type: SortedVector
      elementType: Segment
    optional: true
  - name: EntryPoint
    doc: The program entry point, if any.
    type: MetaAddress
    optional: true
  - name: Types
    doc: |-
      The type system.
      It contains primitive types, `struct`, `union`, `typedef`, `enum` 
      and function prototypes.
    sequence:
      type: SortedVector
      upcastable: true
      elementType: Type
    optional: true
  - name: ImportedLibraries
    doc: |-
      The list of imported libraries identified by their file name.
      For instance, if the input binary is linked to OpenSSL this list would
      contain `libcrypto.so.1.1`.
    sequence:
      type: SortedVector
      elementType: string
    optional: true
  - name: ExtraCodeAddresses
    doc: |-
      A list of addresses known to contain code.
      rev.ng is usually able to discover all the code by itself by recursively
      visiting the control-flow graph of functions and the call graph.
      However, certain pieces of code cannot be identified through these
      techniques.
      A prime example are the addresses of `catch` blocks of C++ exception
      handlers: no code ever directly jumps there and their address is not
      stored in jump tables. Their address can only be obtained by interpreting
      metadata in the ELF.
    optional: true
    sequence:
      type: SortedVector
      elementType: MetaAddress
TUPLE-TREE-YAML */

#include "revng/Model/Generated/Early/Binary.h"

// TODO: Prevent changing the keys. Currently we need them to be public and
//       non-const for serialization purposes.

namespace model {
using TypePath = TupleTreeReference<model::Type, model::Binary>;
}

class model::Binary : public model::generated::Binary {
public:
  using generated::Binary::Binary;

public:
  model::TypePath getTypePath(const model::Type::Key &Key) {
    return TypePath::fromString(this, "/Types/" + getNameFromYAMLScalar(Key));
  }

  model::TypePath getTypePath(const model::Type::Key &Key) const {
    return TypePath::fromString(this, "/Types/" + getNameFromYAMLScalar(Key));
  }

  model::TypePath getTypePath(const model::Type *T) {
    return getTypePath(T->key());
  }

  model::TypePath getTypePath(const model::Type *T) const {
    return getTypePath(T->key());
  }

  model::TypePath recordNewType(UpcastablePointer<Type> &&T);

  model::TypePath
  getPrimitiveType(PrimitiveTypeKind::Values V, uint8_t ByteSize);

  model::TypePath
  getPrimitiveType(PrimitiveTypeKind::Values V, uint8_t ByteSize) const;

  model::QualifiedType getPointerTo(const model::QualifiedType &Type) const {
    QualifiedType Result = Type;
    Result.Qualifiers.insert(Result.Qualifiers.begin(),
                             model::Qualifier::createPointer(Architecture));
    return Result;
  }

  bool verifyTypes() const debug_function;
  bool verifyTypes(bool Assert) const debug_function;
  bool verifyTypes(VerifyHelper &VH) const;

public:
  bool verify() const debug_function;
  bool verify(bool Assert) const debug_function;
  bool verify(VerifyHelper &VH) const;
  void verify(ErrorList &EL) const;
  void dump() const debug_function;
  void dumpTypeGraph(const char *Path) const debug_function;
  std::string toString() const debug_function;
};

inline model::TypePath
getPrototype(const model::Binary &Binary,
             const model::DynamicFunction &DynamicFunction) {
  if (DynamicFunction.Prototype.isValid())
    return DynamicFunction.Prototype;
  else
    return Binary.DefaultPrototype;
}

#include "revng/Model/Generated/Late/Binary.h"
