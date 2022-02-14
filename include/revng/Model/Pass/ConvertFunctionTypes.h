#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/Binary.h"

namespace model {

/// Tries to convert all the `model::RawFunctionType`s within the input `Model`
/// to `model::CABIFunctionType`.
///
/// Internally uses `model::convertToCABIFunctionType`.
void convertAllFunctionsToCABI(TupleTree<model::Binary> &Model,
                               model::ABI::Values TargetABI);

/// Tries to convert all the `model::RawFunctionType`s within the input `Model`
/// to `model::CABIFunctionType`.
///
/// Internally uses `model::convertToCABIFunctionType`.
inline void convertAllFunctionsToCABI(TupleTree<model::Binary> &Model) {
  convertAllFunctionsToCABI(Model, Model->DefaultABI);
}

/// Tries to convert all the `model::CABIFunctionType`s within the input `Model`
/// to `model::RawFunctionType`.
///
/// Internally uses `model::convertToRawFunctionType`.
void convertAllFunctionsToRaw(TupleTree<model::Binary> &Model);

} // namespace model
