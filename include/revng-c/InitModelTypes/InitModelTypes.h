#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <map>

#include "revng/EarlyFunctionAnalysis/FunctionMetadataCache.h"

namespace llvm {
class Value;
class Function;
} // namespace llvm

namespace model {
class Function;
class QualifiedType;
class Binary;
} // namespace model

/// Associate a QualifiedType to each llvm::Instruction. This is done
/// in 3 ways:
/// 1. If the Value has a well defined type in the model (e.g. the stack), use
///    that type
/// 2. If the Value is derived from a Value with a known type, you might
///    propagate it (e.g. loads or `AddressOf` calls)
/// 3. In all other cases, derive the QualifiedType from the LLVM Type
/// \note If the `PointersOnly` flag is set, only pointer types will be added to
/// the map
extern std::map<const llvm::Value *, const model::QualifiedType>
initModelTypes(FunctionMetadataCache &Cache,
               const llvm::Function &F,
               const model::Function *ModelF,
               const model::Binary &Model,
               bool PointersOnly);
