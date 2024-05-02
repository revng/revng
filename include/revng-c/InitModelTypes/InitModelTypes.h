#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <map>

namespace llvm {
class Value;
class Function;
} // namespace llvm

/// Associate a model type to each `llvm::Instruction`. This is done in 3 ways:
///
/// 1. If the Value has a well defined type in the model (e.g. the stack), use
///    that type
/// 2. If the Value is derived from a Value with a known type, you might
///    propagate it (e.g. loads or `AddressOf` calls)
/// 3. In all other cases, derive the type from the LLVM Type
///
/// \note If the `PointersOnly` flag is set, only pointer types will be added to
/// the map
extern std::map<const llvm::Value *, const model::UpcastableType>
initModelTypes(const llvm::Function &F,
               const model::Function *ModelF,
               const model::Binary &Model,
               bool PointersOnly);
