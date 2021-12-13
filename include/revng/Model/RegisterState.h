#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

/* TUPLE-TREE-YAML
name: RegisterState
type: enum
members:
  - name: "No"
  - name: NoOrDead
  - name: Dead
  - name: "Yes"
  - name: YesOrDead
  - name: Maybe
  - name: Contradiction
TUPLE-TREE-YAML */

#include "revng/Model/Generated/Early/RegisterState.h"

namespace model::RegisterState {
inline bool isYesOrDead(model::RegisterState::Values V) {
  return (V == model::RegisterState::Yes or V == model::RegisterState::YesOrDead
          or V == model::RegisterState::Dead);
}

inline bool shouldEmit(model::RegisterState::Values V) {
  return isYesOrDead(V);
}

} // namespace model::RegisterState

#include "revng/Model/Generated/Late/RegisterState.h"
