#pragma once

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

#include "revng/Model/DynamicFunction.h"
#include "revng/Model/Function.h"
#include "revng/Model/Segment.h"
#include "revng/Model/Type.h"
#include "revng/Pipeline/Rank.h"
#include "revng/Support/BasicBlockID/YAMLTraits.h"
#include "revng/Support/MetaAddress.h"
#include "revng/Support/YAMLTraits.h"

namespace revng::ranks {

namespace detail {

using SFK = model::StructField::Key;
using UFK = model::UnionField::Key;
using CAK = model::Argument::Key;
using RAK = model::NamedTypedRegister::Key;

} // namespace detail

static_assert(HasScalarOrEnumTraits<MetaAddress>);
static_assert(HasScalarOrEnumTraits<BasicBlockID>);

inline auto Binary = pipeline::defineRootRank<"binary">();

using pipeline::defineRank;
inline auto Function = defineRank<"function", model::Function::Key>(Binary);
inline auto BasicBlock = defineRank<"basic-block", BasicBlockID>(Function);
inline auto Instruction = defineRank<"instruction", MetaAddress>(BasicBlock);

inline auto Type = defineRank<"type", model::Type::Key>(Binary);
inline auto StructField = defineRank<"struct-field", detail::SFK>(Type);
inline auto UnionField = defineRank<"union-field", detail::UFK>(Type);
inline auto EnumEntry = defineRank<"enum-entry", model::EnumEntry::Key>(Type);
inline auto CABIArgument = defineRank<"cabi-argument", detail::CAK>(Type);
inline auto RawArgument = defineRank<"raw-argument", detail::RAK>(Type);
inline auto ReturnValue = defineRank<"return-value", model::Type::Key>(Binary);
inline auto ReturnRegister = defineRank<"return-register",
                                        model::NamedTypedRegister::Key>(Type);

inline auto RawByte = defineRank<"raw-byte", MetaAddress>(Binary);
inline auto RawByteRange = defineRank<"raw-byte-range", MetaAddress>(RawByte);

inline auto Segment = defineRank<"segment", model::Segment::Key>(Binary);

inline auto DynamicFunction = defineRank<"dynamic-function",
                                         model::DynamicFunction::Key>(Binary);

} // namespace revng::ranks
