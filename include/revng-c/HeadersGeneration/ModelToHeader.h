#pragma once

//
// Copyright (c) rev.ng Labs Srl. See LICENSE.md for details.
//

#include <set>

#include "llvm/Support/raw_ostream.h"

#include "revng/Model/ForwardDecls.h"

/// Generate a C header containing a serialization of the type system,
/// i.e. function prototypes, structs, unions, typedefs, and anything that
/// resides in the model.
/// All the types from the Model we want to ignore when printing the C header we
/// should pass through \p IgnoreTypes. If there is a function that should be
/// avoided, its address should be provided via \p FunctionToIgnore. The \p
/// GeneratePlainC controls printing of either PTML or plain C code.
bool dumpModelToHeader(const model::Binary &Model,
                       llvm::raw_ostream &Out,
                       const std::set<const model::Type *> &IgnoreTypes = {},
                       MetaAddress FunctionToIgnore = MetaAddress::invalid(),
                       bool GeneratePlainC = false);
