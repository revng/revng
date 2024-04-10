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
inline auto Function = defineRank<"function", model::Function::Key>(Binary);
inline auto BasicBlock = defineRank<"basic-block", BasicBlockID>(Function);
inline auto Instruction = defineRank<"instruction", MetaAddress>(BasicBlock);

inline auto TypeDefinition = defineRank<"type-definition", detail::TDK>(Binary);
inline auto
  StructField = defineRank<"struct-field", detail::SFK>(TypeDefinition);
inline auto UnionField = defineRank<"union-field", detail::UFK>(TypeDefinition);
inline auto
  EnumEntry = defineRank<"enum-entry", model::EnumEntry::Key>(TypeDefinition);
inline auto
  CABIArgument = defineRank<"cabi-argument", detail::CAK>(TypeDefinition);
inline auto
  RawArgument = defineRank<"raw-argument", detail::NRK>(TypeDefinition);
inline auto ReturnValue = defineRank<"return-value", detail::TDK>(Binary);
inline auto
  ReturnRegister = defineRank<"return-register", detail::NRK>(TypeDefinition);

inline auto RawByte = defineRank<"raw-byte", MetaAddress>(Binary);
inline auto RawByteRange = defineRank<"raw-byte-range", MetaAddress>(RawByte);

inline auto Segment = defineRank<"segment", model::Segment::Key>(Binary);

inline auto DynamicFunction = defineRank<"dynamic-function",
                                         model::DynamicFunction::Key>(Binary);

} // namespace revng::ranks
