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

bool dumpModelToHeader(const model::Binary &Model, llvm::raw_ostream &Header);
