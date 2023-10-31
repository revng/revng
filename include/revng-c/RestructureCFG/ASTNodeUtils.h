#pragma once

//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

class ASTNode;
class ASTTree;

extern bool needsLoopVar(const ASTNode *N);

extern void flipEmptyThen(ASTTree &AST, ASTNode *RootNode);

extern ASTNode *collapseSequences(ASTTree &AST, ASTNode *RootNode);

extern ASTNode *simplifyAtomicSequence(ASTTree &AST, ASTNode *RootNode);

extern ASTNode *canonicalize(ASTTree &AST, ASTNode *RootNode);
