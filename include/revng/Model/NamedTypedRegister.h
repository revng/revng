#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/ADT/RecursiveCoroutine.h"
#include "revng/Model/Identifier.h"
#include "revng/Model/Type.h"
#include "revng/Model/VerifyHelper.h"

/* TUPLE-TREE-YAML
name: NamedTypedRegister
type: struct
doc: |-
  An argument or return values in a `RawFunctionDefinition`.

  It is basically a pair of a register and a `Type`.
fields:
  - name: Location
    type: Register
  - name: Type
    type: Type
    upcastable: true
  - name: CustomName
    type: Identifier
    optional: true
  - name: OriginalName
    type: Identifier
    optional: true
  - name: Comment
    type: string
    optional: true
key:
  - Location
TUPLE-TREE-YAML */

#include "revng/Model/Generated/Early/NamedTypedRegister.h"

class model::NamedTypedRegister : public model::generated::NamedTypedRegister {
public:
  using generated::NamedTypedRegister::NamedTypedRegister;

public:
  bool verify() const debug_function;
  bool verify(bool Assert) const debug_function;
  RecursiveCoroutine<bool> verify(VerifyHelper &VH) const;
};

#include "revng/Model/Generated/Late/NamedTypedRegister.h"
