#pragma once

//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

// Forward declarations
class ASTNode;
class ASTTree;

extern ASTNode *simplifyHybridNot(ASTTree &AST, ASTNode *RootNode);
