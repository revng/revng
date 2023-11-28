#pragma once

//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

#include "FallThroughScopeAnalysis.h"

// Forward declarations
class ASTNode;
class ASTTree;

extern ASTNode *inlineDispatcherSwitch(ASTTree &AST, ASTNode *RootNode);
