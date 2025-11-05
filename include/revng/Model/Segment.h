#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/ADT/SortedVector.h"
#include "revng/Model/Relocation.h"
#include "revng/Model/TypeDefinition.h"
#include "revng/Support/MetaAddress.h"
#include "revng/Support/MetaAddress/YAMLTraits.h"

#include "revng/Model/Generated/Early/Segment.h"

namespace model {
class VerifyHelper;
}

class model::Segment : public model::generated::Segment {
public:
  using generated::Segment::Segment;

public:
  /// The helper for segment type unwrapping.
  /// Use this when you need to access/modify the existing struct,
  /// and \ref Type() when you need to assign a new one.
  model::StructDefinition *type() {
    if (Type().isEmpty())
      return nullptr;
    else
      return &Type()->toStruct();
  }

  /// The helper for segment type unwrapping.
  /// Use this when you need to access/modify the existing struct,
  /// and \ref Type() when you need to assign a new one.
  const model::StructDefinition *type() const {
    if (Type().isEmpty())
      return nullptr;
    else
      return &Type()->toStruct();
  }

public:
  bool contains(MetaAddress Address) const {
    return (Address.isValid()
            and StartAddress().addressLowerThanOrEqual(Address)
            and Address.addressLowerThan(endAddress()));
  }

  bool contains(MetaAddress Start, uint64_t Size) const {
    return contains(Start) and (Size <= 1 or contains(Start + Size - 1));
  }

  bool hasDataFor(MetaAddress Address) const {
    return (Address.isValid()
            and StartAddress().addressLowerThanOrEqual(Address)
            and Address.addressLowerThan(endDataAddress()));
  }

  bool hasDataFor(MetaAddress Start, uint64_t Size) const {
    return contains(Start) and (Size <= 1 or contains(Start + Size - 1));
  }

  /// \return the end offset (guaranteed to be greater than StartOffset).
  auto endOffset() const { return StartOffset() + FileSize(); }

  /// \return a valid MetaAddress.
  MetaAddress endAddress() const { return StartAddress() + VirtualSize(); }

  /// \return a valid MetaAddress.
  MetaAddress endDataAddress() const { return StartAddress() + FileSize(); }

  std::pair<MetaAddress, MetaAddress> pagesRange() const {
    MetaAddress Start = StartAddress();
    Start = Start - (Start.address() % 4096);

    MetaAddress End = endAddress();
    End = End + (((End.address() + (4096 - 1)) / 4096) * 4096 - End.address());

    return { Start, End };
  }

public:
  bool verify() const debug_function;
  bool verify(bool Assert) const debug_function;
  bool verify(VerifyHelper &VH) const;
};

#include "revng/Model/Generated/Late/Segment.h"
