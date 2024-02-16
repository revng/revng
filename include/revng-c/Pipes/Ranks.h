#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/Argument.h"
#include "revng/Model/Function.h"
#include "revng/Model/NamedTypedRegister.h"
#include "revng/Pipes/Ranks.h"
#include "revng/Support/YAMLTraits.h"

namespace revng::ranks {

using pipeline::defineRank;

/// Rank for locations associated to QEMU and LLVM helper functions
inline auto HelperFunction = defineRank<"helper-function", std::string>(Binary);

/// Rank for locations associated to struct return types of QEMU and LLVM helper
/// functions
inline auto HelperStructType = defineRank<"helper-struct-type", // formatting
                                          std::string>(Binary);

/// Rank for locations associated to fields of struct return types of QEMU and
/// LLVM helper functions
inline auto HelperStructField = defineRank<"helper-struct-field",
                                           std::string>(HelperStructType);

/// Rank for locations associated to arguments of dynamic functions.
inline auto DynamicFunctionArgument = defineRank<"dynamic-function-argument",
                                                 std::string>(DynamicFunction);

/// Rank for locations associated to function arguments and local variables.
inline auto LocalVariable = defineRank<"local-variable", std::string>(Function);

/// Rank for artificial structs returned by raw functions
inline auto ArtificialStruct = defineRank<"artificial-struct",
                                          model::RawFunctionType::Key>(Binary);

} // namespace revng::ranks
