#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Forward declarations
class ASTNode;
class ASTTree;

extern void simplifyImplicitReturn(ASTTree &AST, ASTNode *RootNode);

extern void simplifyImplicitContinue(ASTTree &AST);
