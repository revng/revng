#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <array>
#include <iterator>
#include <optional>
#include <span>
#include <variant>

#include "revng/ABI/RegisterState.h"
#include "revng/ADT/STLExtras.h"
#include "revng/Model/ABI.h"
#include "revng/Model/Architecture.h"
#include "revng/Model/Register.h"
#include "revng/Support/EnumSwitch.h"

namespace abi {

std::optional<abi::RegisterState::Map>
tryApplyRegisterStateDeductions(const abi::RegisterState::Map &State,
                                model::ABI::Values ABI);

abi::RegisterState::Map
enforceRegisterStateDeductions(const abi::RegisterState::Map &State,
                               model::ABI::Values ABI);

} // namespace abi
