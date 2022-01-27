#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/Identifier.h"
#include "revng/Model/VerifyHelper.h"
#include "revng/Support/MetaAddress.h"
#include "revng/Support/MetaAddress/YAMLTraits.h"

/* TUPLE-TREE-YAML
name: Segment
type: struct
fields:
  - name: StartAddress
    type: MetaAddress
  - name: EndAddress
    type: MetaAddress
  - name: StartOffset
    type: uint64_t
  - name: EndOffset
    type: uint64_t
  - name: IsReadable
    type: bool
  - name: IsWriteable
    type: bool
  - name: IsExecutable
    type: bool
  - name: CustomName
    type: Identifier
    optional: true
key:
  - StartAddress
  - EndAddress
TUPLE-TREE-YAML */

#include "revng/Model/Generated/Early/Segment.h"

class model::Segment : public model::generated::Segment {
public:
  using generated::Segment::Segment;

public:
  Identifier name() const;

public:
  bool verify() const debug_function;
  bool verify(bool Assert) const debug_function;
  bool verify(VerifyHelper &VH) const;
  void dump() const debug_function;
};

#include "revng/Model/Generated/Late/Segment.h"
