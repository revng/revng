#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/ADT/RecursiveCoroutine.h"
#include "revng/Model/QualifiedType.h"
#include "revng/Model/VerifyHelper.h"

/* TUPLE-TREE-YAML
name: TypedRegister
type: struct
fields:
  - name: Location
    type: Register
  - name: Type
    type: QualifiedType
  - name: Comment
    type: string
    optional: true
key:
  - Location
TUPLE-TREE-YAML */

#include "revng/Model/Generated/Early/TypedRegister.h"

class model::TypedRegister : public model::generated::TypedRegister {
public:
  using generated::TypedRegister::TypedRegister;

public:
  bool verify() const debug_function;
  bool verify(bool Assert) const debug_function;
  RecursiveCoroutine<bool> verify(VerifyHelper &VH) const;
  void dump() const debug_function;
};

#include "revng/Model/Generated/Late/TypedRegister.h"
