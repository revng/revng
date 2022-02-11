#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/StringRef.h"

#include "revng/Support/Assert.h"

namespace abi::RegisterState {

enum Values {
  Invalid,
  No,
  NoOrDead,
  Dead,
  Yes,
  YesOrDead,
  Maybe,
  Contradiction,

  Count
};

inline llvm::StringRef getName(Values V) {
  switch (V) {
  case Invalid:
    return "Invalid";
  case No:
    return "No";
  case NoOrDead:
    return "NoOrDead";
  case Dead:
    return "Dead";
  case Yes:
    return "Yes";
  case YesOrDead:
    return "YesOrDead";
  case Maybe:
    return "Maybe";
  case Contradiction:
    return "Contradiction";
  case Count:
    revng_abort();
    break;
  }
}

inline Values fromName(llvm::StringRef Name) {
  if (Name == "Invalid") {
    return Invalid;
  } else if (Name == "No") {
    return No;
  } else if (Name == "NoOrDead") {
    return NoOrDead;
  } else if (Name == "Dead") {
    return Dead;
  } else if (Name == "Yes") {
    return Yes;
  } else if (Name == "YesOrDead") {
    return YesOrDead;
  } else if (Name == "Maybe") {
    return Maybe;
  } else if (Name == "Contradiction") {
    return Contradiction;
  } else {
    revng_abort();
  }
}

inline bool isYesOrDead(abi::RegisterState::Values V) {
  return (V == abi::RegisterState::Yes or V == abi::RegisterState::YesOrDead
          or V == abi::RegisterState::Dead);
}

inline bool shouldEmit(abi::RegisterState::Values V) {
  return isYesOrDead(V);
}

} // namespace abi::RegisterState
