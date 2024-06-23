#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
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
  /// Minimize the emitted code to avoid pitfalls for the user editing the type
  ///
  /// For example, this option disables emission of the _enum_max_value_* entry
  /// in enums.
  bool ForEditing = false;
  /// If set to true, disables all type inlining mechanisms. These include:
  /// 1. Not printing a type if its only use is by a single function as a stack
  ///    type
  /// 2. Not printing any type which its only use (either directly or
  ///    transitively) derives from (1)
  /// TODO: this now conflates a) printing stack types inline in function=
  /// bodies; b) printing inline types in types-and-globals.h
  /// In principle the two options could be split and activated separately.
  /// TODO: type inlining is currently broken in rare cases involving recursive
  /// types, due to its logic using a different graph than ModelToHeader proper.
  /// Revert back to false when the bugs are fixed.
  bool DisableTypeInlining = true;
};

/// Generate a C header containing a serialization of the type system,
/// i.e. function prototypes, structs, unions, typedefs, and anything that
/// resides in the model.
bool dumpModelToHeader(const model::Binary &Model,
                       llvm::raw_ostream &Out,
                       const ModelToHeaderOptions &Options);
