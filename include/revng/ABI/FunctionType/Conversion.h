#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/Binary.h"

namespace abi::FunctionType {

/// Best effort `CABIFunctionDefinition` to `RawFunctionDefinition` conversion.
///
/// If `ABI` is not specified, `Binary.DefaultABI` is used instead.
///
/// \param UseSoftRegisterStateDeductions For specifics see the difference
///        between `abi::Definition::tryDeducingArgumentRegisterState` (`true`)
///        and `abi::Definition::enforceArgumentRegisterState` (`false`).
std::optional<model::TypeDefinitionPath>
tryConvertToCABI(const model::RawFunctionDefinition &Function,
                 TupleTree<model::Binary> &Binary,
                 std::optional<model::ABI::Values> ABI = std::nullopt,
                 bool UseSoftRegisterStateDeductions = true);

} // namespace abi::FunctionType
