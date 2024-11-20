#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

class ASTNode;
class ASTTree;

extern bool needsLoopVar(const ASTNode *N);

extern void flipEmptyThen(ASTTree &AST, ASTNode *RootNode);

extern ASTNode *collapseSequences(ASTTree &AST, ASTNode *RootNode);

extern ASTNode *simplifyAtomicSequence(ASTTree &AST, ASTNode *RootNode);

extern ASTNode *canonicalize(ASTTree &AST, ASTNode *RootNode);
