#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/Architecture.h"

/* TUPLE-TREE-YAML
name: RelocationType
doc: |-
  A enumeration describing how a `Relocation` should be written at the target
  address.
type: enum
members:
  - name: WriteAbsoluteAddress32
    doc: Write the absolute address of the object as a 32-bit integer.
  - name: WriteAbsoluteAddress64
    doc: Write the absolute address of the object as a 64-bit integer.
  - name: AddAbsoluteAddress32
    doc: |-
      Add to the 32-bit integer present at the target location the absolute
      address of the object.
  - name: AddAbsoluteAddress64
    doc: |-
      Add to the 64-bit integer present at the target location the absolute
      address of the object.
  - name: WriteRelativeAddress32
    doc: |-
      Write the address of the object as a 32-bit integer, expressed as a
      relative from the target of the relocation.
  - name: WriteRelativeAddress64
    doc: |-
      Write the address of the object as a 64-bit integer, expressed as a
      relative from the target of the relocation.
  - name: AddRelativeAddress32
    doc: |-
      Add to the 32-bit integer present at the target location the address of
      the object, expressed as a relative from the target of the relocation.
  - name: AddRelativeAddress64
    doc: |-
      Add to the 32-bit integer present at the target location the address of
      the object, expressed as a relative from the target of the relocation.
TUPLE-TREE-YAML */

#include "revng/Model/Generated/Early/RelocationType.h"

constexpr unsigned char R_MIPS_IMPLICIT_RELATIVE = 255;

namespace model::RelocationType {

inline uint64_t getSize(Values V) {
  switch (V) {
  case WriteAbsoluteAddress32:
  case AddAbsoluteAddress32:
  case WriteRelativeAddress32:
  case AddRelativeAddress32:
    return 4;

  case WriteAbsoluteAddress64:
  case AddAbsoluteAddress64:
  case WriteRelativeAddress64:
  case AddRelativeAddress64:
    return 8;

  default:
    revng_abort();
  }
}

Values fromELFRelocation(model::Architecture::Values Architecture,
                         unsigned char ELFRelocation);

Values formCOFFRelocation(model::Architecture::Values Architecture);

bool isELFRelocationBaseRelative(model::Architecture::Values Architecture,
                                 unsigned char ELFRelocation);

} // namespace model::RelocationType

#include "revng/Model/Generated/Late/RelocationType.h"
