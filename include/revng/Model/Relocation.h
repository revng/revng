#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/RelocationType.h"
#include "revng/Support/MetaAddress.h"
#include "revng/Support/MetaAddress/YAMLTraits.h"

/* TUPLE-TREE-YAML
name: Relocation
type: struct
doc: |-
  A relocation, i.e., a directive to write the address of an object (a
  `DynamicFunction` or a `Segment`) at a specific address.
  Optionally, the address of the object can be modified with a constant offset.
fields:
  - name: Address
    type: MetaAddress
    doc: Where to write the address of the object.
  - name: Type
    type: RelocationType
    doc: |-
      How to write the address of the object (e.g., 32-bit vs 64-bit
      integer).
  - name: Addend
    type: uint64_t
    doc: Fixed offset to add when writing the address of the object.

key:
  - Address
  - Type
TUPLE-TREE-YAML */

#include "revng/Model/Generated/Early/Relocation.h"

namespace model {
class VerifyHelper;
}

class model::Relocation : public model::generated::Relocation {
public:
  using generated::Relocation::Relocation;

public:
  uint64_t size() const { return model::RelocationType::getSize(Type()); }

  /// \return a valid end address.
  MetaAddress endAddress() const { return Address() + size(); }

public:
  bool verify() const debug_function;
  bool verify(bool Assert) const debug_function;
  bool verify(VerifyHelper &VH) const;
};

#include "revng/Model/Generated/Late/Relocation.h"
