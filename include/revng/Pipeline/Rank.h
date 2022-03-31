#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Support/DynamicHierarchy.h"

namespace pipeline {

/// The rank tree is a tree used by targets to find out how many names
/// are required to name a target
class Rank : public DynamicHierarchy<Rank> {
public:
  Rank(llvm::StringRef Name) : DynamicHierarchy(Name) {}
  Rank(llvm::StringRef Name, Rank &Parent) : DynamicHierarchy(Name, Parent) {}
};

} // namespace pipeline
