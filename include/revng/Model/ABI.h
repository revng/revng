#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/StringRef.h"

#include "revng/Support/Debug.h"
#include "revng/Support/YAMLTraits.h"
#include "revng/TupleTree/TupleTree.h"

namespace model::abi {

enum Values { Invalid, SystemV_x86_64, Count };

inline llvm::StringRef getName(Values V) {
  switch (V) {
  case Invalid:
    return "Invalid";
  case SystemV_x86_64:
    return "SystemV_x86_64";
  default:
    revng_abort();
  }
  revng_abort();
}

} // namespace model::abi

namespace llvm::yaml {

template<>
struct ScalarEnumerationTraits<model::abi::Values>
  : public NamedEnumScalarTraits<model::abi::Values> {};

} // namespace llvm::yaml
