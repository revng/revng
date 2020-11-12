#pragma once

//
// Copyright rev.ng Srls. See LICENSE.md for details.
//

#include <cstdio>

#include "clang/Frontend/ASTConsumers.h"
#include "clang/Frontend/FrontendAction.h"

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
                    ASTTree &CombedAST,
                    BBPHIMap &BlockToPHIIncoming,
                    std::unique_ptr<llvm::raw_ostream> O,
                    DuplicationMap &NDuplicates) :
    F(F),
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
  ASTTree &CombedAST;
  BBPHIMap &BlockToPHIIncoming;
  std::unique_ptr<llvm::raw_ostream> O;
  DuplicationMap &NDuplicates;
};

} // end namespace tooling

} // namespace clang
