#pragma once

//
// Copyright (c) rev.ng Labs Srl. See LICENSE.md for details.
//

#include <set>

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/Model/ForwardDecls.h"
#include "revng/Support/MetaAddress.h"

struct ModelToHeaderOptions {
  /// If true, print plain C (i.e., no PTML markup)
  bool GeneratePlainC = false;
  llvm::SmallPtrSet<const model::Type *, 2> TypesToOmit;
  std::set<MetaAddress> FunctionsToOmit;
  /// Piece of code to include after the main includes have been emitted
  std::string PostIncludes;
};

/// Generate a C header containing a serialization of the type system,
/// i.e. function prototypes, structs, unions, typedefs, and anything that
/// resides in the model.
bool dumpModelToHeader(const model::Binary &Model,
                       llvm::raw_ostream &Out,
                       const ModelToHeaderOptions &Options);
