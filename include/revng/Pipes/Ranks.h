#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/Binary.h"
#include "revng/Pipeline/Rank.h"
#include "revng/Support/BasicBlockID/YAMLTraits.h"
#include "revng/Support/MetaAddress.h"
#include "revng/Support/YAMLTraits.h"

namespace revng::ranks {

namespace detail {

using TDK = model::TypeDefinition::Key;
using SFK = model::StructField::Key;
using UFK = model::UnionField::Key;
using CAK = model::Argument::Key;
using NRK = model::NamedTypedRegister::Key;

} // namespace detail

static_assert(HasScalarOrEnumTraits<MetaAddress>);
static_assert(HasScalarOrEnumTraits<BasicBlockID>);

inline auto Binary = pipeline::defineRootRank<"binary">();

using pipeline::defineRank;
inline auto
  Function = defineRank<"function", model::Function::Key, "Functions">(Binary);
inline auto BasicBlock = defineRank<"basic-block", BasicBlockID>(Function);
inline auto Instruction = defineRank<"instruction", MetaAddress>(BasicBlock);

inline auto TypeDefinition = defineRank<"type-definition",
                                        detail::TDK,
                                        "TypeDefinitions">(Binary);
inline auto StructField = defineRank<"struct-field",
                                     detail::SFK,
                                     "Fields">(TypeDefinition);
inline auto
  UnionField = defineRank<"union-field", detail::UFK, "Fields">(TypeDefinition);
inline auto EnumEntry = defineRank<"enum-entry",
                                   model::EnumEntry::Key,
                                   "Entries">(TypeDefinition);
inline auto CABIArgument = defineRank<"cabi-argument",
                                      detail::CAK,
                                      "Arguments">(TypeDefinition);
inline auto RawArgument = defineRank<"raw-argument",
                                     detail::NRK,
                                     "Arguments">(TypeDefinition);
inline auto ReturnValue = defineRank<"return-value", detail::TDK>(Binary);
inline auto ReturnRegister = defineRank<"return-register",
                                        detail::NRK,
                                        "ReturnValues">(TypeDefinition);

inline auto RawByte = defineRank<"raw-byte", MetaAddress>(Binary);
inline auto RawByteRange = defineRank<"raw-byte-range", MetaAddress>(RawByte);

inline auto
  Segment = defineRank<"segment", model::Segment::Key, "Segments">(Binary);

inline auto DynamicFunction = defineRank<"dynamic-function",
                                         model::DynamicFunction::Key,
                                         "ImportedDynamicFunctions">(Binary);

inline auto PrimitiveType = defineRank<"primitive", std::string>(Binary);

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

/// Rank for locations associated to goto-labels within functions.
inline auto GotoLabel = defineRank<"goto-label", std::string>(Function);

/// Rank for locations associated to comments within function bodies.
inline auto
  StatementComment = defineRank<"statement-comment", uint64_t>(Function);

/// Rank for artificial structs returned by raw functions
inline auto
  ArtificialStruct = defineRank<"artificial-struct",
                                model::RawFunctionDefinition::Key>(Binary);
} // namespace revng::ranks
