#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "FallThroughScopeAnalysis.h"

// Forward declarations
class ASTNode;
class ASTTree;

extern ASTNode *inlineDispatcherSwitch(ASTTree &AST, ASTNode *RootNode);

extern ASTNode *simplifySwitchBreak(ASTTree &AST, ASTNode *RootNode);
