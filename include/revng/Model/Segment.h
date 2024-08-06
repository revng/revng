#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/ADT/SortedVector.h"
#include "revng/Model/Identifier.h"
#include "revng/Model/Relocation.h"
#include "revng/Model/TypeDefinition.h"
#include "revng/Model/VerifyHelper.h"
#include "revng/Support/MetaAddress.h"
#include "revng/Support/MetaAddress/YAMLTraits.h"

/* TUPLE-TREE-YAML
name: Segment
type: struct
fields:
  - name: StartAddress
    type: MetaAddress
  - name: VirtualSize
    type: uint64_t
  - name: StartOffset
    type: uint64_t
  - name: FileSize
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
  - name: OriginalName
    type: string
    optional: true
  - name: Comment
    type: string
    optional: true
  - name: CanonicalRegisterValues
    optional: true
    sequence:
      type: SortedVector
      elementType: CanonicalRegisterValue
  - name: Relocations
    optional: true
    sequence:
      type: SortedVector
      elementType: Relocation
  - name: Type
    doc: The type of the segment
    type: Type
    upcastable: true
    optional: true

key:
  - StartAddress
  - VirtualSize
TUPLE-TREE-YAML */

#include "revng/Model/Generated/Early/Segment.h"

class model::Segment : public model::generated::Segment {
public:
  using generated::Segment::Segment;

public:
  Identifier name() const;

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
    auto EndAddress = StartAddress() + VirtualSize();
    return (Address.isValid()
            and StartAddress().addressLowerThanOrEqual(Address)
            and Address.addressLowerThan(EndAddress));
  }

  bool contains(MetaAddress Start, uint64_t Size) const {
    return contains(Start) and (Size <= 1 or contains(Start + Size - 1));
  }

  /// \return the end offset (guaranteed to be greater than StartOffset).
  auto endOffset() const { return StartOffset() + FileSize(); }

  /// \return a valid MetaAddress.
  auto endAddress() const { return StartAddress() + VirtualSize(); }
  auto startAddress() const { return StartAddress(); } // TODO(anjo): TMP
  auto size() const { return VirtualSize(); } // TODO(anjo): TMP

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
