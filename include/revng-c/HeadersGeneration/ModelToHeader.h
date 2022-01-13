#pragma once

//
// Copyright (c) rev.ng Srls. See LICENSE.md for details.
//

namespace llvm {

class raw_ostream;

} // end namespace llvm

namespace model {

class Binary;

} // end namespace model

/// \brief Generate a C header containing a serialization of the type system,
/// i.e. function prototypes, structs, unions, typedefs, and anything that
/// resides in the model.
bool dumpModelToHeader(const model::Binary &Model, llvm::raw_ostream &Header);
