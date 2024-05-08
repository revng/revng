#pragma once

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

#include "revng/ADT/SortedVector.h"
#include "revng/Model/Argument.h"
#include "revng/Model/Identifier.h"
#include "revng/Model/QualifiedType.h"
#include "revng/Model/Type.h"

/* TUPLE-TREE-YAML
name: CABIFunctionType
doc: |
  The function type described through a C-like prototype plus an ABI.

  This is an "high level" representation of the prototype of a function. It is
  expressed as list of arguments composed by an index and a type. No
  information about the register is embedded. That information is implicit in
  the ABI this type is associated to.
type: struct
inherits: Type
fields:
  - name: ABI
    type: ABI
  - name: ReturnType
    type: QualifiedType
  - name: ReturnValueComment
    type: string
    optional: true
  - name: Arguments
    sequence:
      type: SortedVector
      elementType: Argument
TUPLE-TREE-YAML */

#include "revng/Model/Generated/Early/CABIFunctionType.h"

class model::CABIFunctionType : public model::generated::CABIFunctionType {
public:
  static constexpr const char *AutomaticNamePrefix = "cabifunction_";

public:
  using generated::CABIFunctionType::CABIFunctionType;
  CABIFunctionType() : generated::CABIFunctionType() {}

public:
  const llvm::SmallVector<model::QualifiedType, 4> edges() const {
    llvm::SmallVector<model::QualifiedType, 4> Result;

    for (const model::Argument &Argument : Arguments())
      Result.push_back(Argument.Type());
    Result.push_back(ReturnType());

    return Result;
  }

public:
  Identifier name() const;
  static bool classof(const Type *T) { return classof(T->key()); }
  static bool classof(const Key &K) { return std::get<1>(K) == AssociatedKind; }
};

#include "revng/Model/Generated/Late/CABIFunctionType.h"
