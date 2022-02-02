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

/* TUPLE-TREE-YAML
name: Binary
doc: Data structure representing the whole binary
type: struct
fields:
  - name: Functions
    doc: List of the functions within the binary
    sequence:
      type: SortedVector
      elementType: model::Function
  - name: ImportedDynamicFunctions
    doc: List of the functions within the binary
    sequence:
      type: SortedVector
      elementType: model::DynamicFunction
  - name: Architecture
    doc: Binary architecture
    type: model::Architecture::Values
  - name: DefaultABI
    doc: The default ABI of `RawFunctionType`s within the binary
    type: model::ABI::Values
  - name: Segments
    doc: List of segments in the original binary
    sequence:
      type: SortedVector
      elementType: model::Segment
  - name: EntryPoint
    doc: Program entry point
    type: MetaAddress
  - name: Types
    doc: The type system
    sequence:
      type: SortedVector
      upcastable: true
      elementType: model::Type
  - name: ImportedLibraries
    doc: List of imported libraries
    sequence:
      type: SortedVector
      elementType: std::string
    optional: true
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
  model::TypePath getTypePath(const model::Type *T) {
    return TypePath::fromString(this,
                                "/Types/" + getNameFromYAMLScalar(T->key()));
  }

  model::TypePath getTypePath(const model::Type *T) const {
    return TypePath::fromString(this,
                                "/Types/" + getNameFromYAMLScalar(T->key()));
  }

  model::TypePath recordNewType(UpcastablePointer<Type> &&T);

  model::TypePath
  getPrimitiveType(PrimitiveTypeKind::Values V, uint8_t ByteSize);

  model::TypePath
  getPrimitiveType(PrimitiveTypeKind::Values V, uint8_t ByteSize) const;

  bool verifyTypes() const debug_function;
  bool verifyTypes(bool Assert) const debug_function;
  bool verifyTypes(VerifyHelper &VH) const;

public:
  bool verify() const debug_function;
  bool verify(bool Assert) const debug_function;
  bool verify(VerifyHelper &VH) const;
  void dump() const debug_function;
  std::string toString() const debug_function;

public:
  void dumpCFG(const Function &F) const debug_function;
};

inline model::TypePath
getPrototype(const model::Binary &Binary, const model::CallEdge &Edge) {
  if (Edge.Type == model::FunctionEdgeType::FunctionCall) {
    if (not Edge.DynamicFunction.empty()) {
      // Get the dynamic function prototype
      return Binary.ImportedDynamicFunctions.at(Edge.DynamicFunction).Prototype;
    } else if (Edge.Destination.isValid()) {
      // Get the function prototype
      return Binary.Functions.at(Edge.Destination).Prototype;
    } else {
      revng_abort();
    }
  } else {
    return Edge.Prototype;
  }
}

inline bool hasAttribute(const model::Binary &Binary,
                         const model::CallEdge &Edge,
                         model::FunctionAttribute::Values Attribute) {
  using namespace model;

  if (Edge.Attributes.count(Attribute) != 0)
    return true;

  if (Edge.Type == FunctionEdgeType::FunctionCall) {
    const MutableSet<FunctionAttribute::Values> *CalleeAttributes = nullptr;
    if (not Edge.DynamicFunction.empty()) {
      const auto &F = Binary.ImportedDynamicFunctions.at(Edge.DynamicFunction);
      CalleeAttributes = &F.Attributes;
    } else if (Edge.Destination.isValid()) {
      CalleeAttributes = &Binary.Functions.at(Edge.Destination).Attributes;
    } else {
      revng_abort();
    }

    return CalleeAttributes->count(Attribute) != 0;
  }

  return false;
}

#include "revng/Model/Generated/Late/Binary.h"
