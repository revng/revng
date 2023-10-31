#pragma once

//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

// Forward declarations
#include "FallThroughScopeAnalysis.h"

class ASTNode;
class ASTTree;

extern ASTNode *promoteCallNoReturn(const model::Binary &Model,
                                    ASTTree &AST,
                                    ASTNode *RootNode);
