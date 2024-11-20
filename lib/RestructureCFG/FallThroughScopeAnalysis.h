#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/IRHelpers.h"

// Forward declarations
class ASTNode;
class ASTTree;

// This `enum class` is used to represent the fallthrough type
enum class FallThroughScopeType {
  FallThrough,

  // This is a placeholder state to represent the combination of two no
  // fallthrough states
  MixedNoFallThrough,
  CallNoReturn,
  Return,
  Continue,
  LoopBreak,
};

using FallThroughScopeTypeMap = std::map<const ASTNode *, FallThroughScopeType>;

bool fallsThrough(FallThroughScopeType Element);

extern FallThroughScopeTypeMap
computeFallThroughScope(const model::Binary &Model, ASTNode *RootNode);
