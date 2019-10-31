#ifndef REVNGC_CDECOMPILERACTION_H
#define REVNGC_CDECOMPILERACTION_H

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// std includes
#include <cstdio>

// clang includes
#include <clang/Frontend/ASTConsumers.h>
#include <clang/Frontend/FrontendAction.h>

// Local directory includes
#include "CDecompilerBeautify.h"

class ASTTree;


namespace clang {

class CompilerInstance;

namespace tooling {

class CDecompilerAction : public ASTFrontendAction {

private:
  using PHIIncomingMap = SmallMap<llvm::PHINode *, unsigned, 4>;
  using BBPHIMap = SmallMap<llvm::BasicBlock *, PHIIncomingMap, 4>;
  using DuplicationMap = std::map<llvm::BasicBlock *, size_t>;

public:
  CDecompilerAction(llvm::Function &F,
                    RegionCFG<llvm::BasicBlock *> &RCFG,
                    ASTTree &CombedAST,
                    BBPHIMap &BlockToPHIIncoming,
                    std::unique_ptr<llvm::raw_ostream> O,
                    DuplicationMap &NDuplicates) :
    F(F),
    RCFG(RCFG),
    CombedAST(CombedAST),
    BlockToPHIIncoming(BlockToPHIIncoming),
    O(std::move(O)),
    NDuplicates(NDuplicates) {}

public:
  std::unique_ptr<ASTConsumer> newASTConsumer();

  virtual std::unique_ptr<ASTConsumer>
  CreateASTConsumer(CompilerInstance &, llvm::StringRef) override;

private:
  llvm::Function &F;
  RegionCFG<llvm::BasicBlock *> &RCFG;
  ASTTree &CombedAST;
  BBPHIMap &BlockToPHIIncoming;
  std::unique_ptr<llvm::raw_ostream> O;
  DuplicationMap &NDuplicates;
};

} // end namespace tooling

} // namespace clang

#endif // REVNGC_CDECOMPILERACTION_H
