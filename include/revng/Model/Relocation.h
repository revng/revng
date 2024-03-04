#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/RelocationType.h"
#include "revng/Model/VerifyHelper.h"
#include "revng/Support/MetaAddress.h"
#include "revng/Support/MetaAddress/YAMLTraits.h"

/* TUPLE-TREE-YAML
name: Relocation
type: struct
fields:
  - name: Address
    type: MetaAddress
  - name: Type
    type: RelocationType
  - name: Addend
    type: uint64_t

key:
  - Address
  - Type
TUPLE-TREE-YAML */

#include "revng/Model/Generated/Early/Relocation.h"

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
