#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/Function.h"
#include "revng/Model/Type.h"
#include "revng/Pipeline/Rank.h"
#include "revng/Support/MetaAddress.h"
#include "revng/Support/YAMLTraits.h"

namespace revng::ranks {

inline auto Binary = pipeline::defineRootRank<"binary">();

using pipeline::defineRank;
inline auto Function = defineRank<"function", model::Function::Key>(Binary);
inline auto BasicBlock = defineRank<"basic-block", MetaAddress>(Function);
inline auto Instruction = defineRank<"instruction", MetaAddress>(BasicBlock);

inline auto Type = defineRank<"type", model::Type::Key>(Binary);
inline auto TypeField = defineRank<"type-field", std::tuple<uint64_t>>(Type);

inline auto RawByte = defineRank<"raw-byte", MetaAddress>(Binary);
inline auto RawByteRange = defineRank<"raw-byte-range", MetaAddress>(RawByte);

} // namespace revng::ranks
