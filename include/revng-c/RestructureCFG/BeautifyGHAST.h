#pragma once

//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

namespace llvm {

class Function;

} // end namespace llvm

class ASTTree;

extern void
beautifyAST(const model::Binary &Model, llvm::Function &F, ASTTree &CombedAST);
