#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Forward declarations
class ASTNode;
class ASTTree;

extern RecursiveCoroutine<ASTNode *> simplifyDualSwitch(ASTTree &AST,
                                                        ASTNode *RootNode);
