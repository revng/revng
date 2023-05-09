#pragma once

//
// Copyright (c) rev.ng Labs Srl. See LICENSE.md for details.
//

#include "llvm/Support/raw_ostream.h"

#include "revng/Model/ForwardDecls.h"

/// Generate a C header containing a serialization of the type system,
/// i.e. function prototypes, structs, unions, typedefs, and anything that
/// resides in the model.
bool dumpModelToHeader(const model::Binary &Model,
                       llvm::raw_ostream &Out,
                       bool GeneratePlainC = false);
