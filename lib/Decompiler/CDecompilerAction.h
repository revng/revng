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

class ASTTree;
class RegionCFG;

namespace clang {

class CompilerInstance;

namespace tooling {

class CDecompilerAction : public ASTFrontendAction {

public:
  CDecompilerAction(llvm::Function &F,
                    RegionCFG &RCFG,
                    ASTTree &CombedAST,
                    std::unique_ptr<llvm::raw_ostream> O) :

    F(F),
    RCFG(RCFG),
    CombedAST(CombedAST),
    O(std::move(O)) {}

public:
  std::unique_ptr<ASTConsumer> newASTConsumer();

  virtual std::unique_ptr<ASTConsumer>
  CreateASTConsumer(CompilerInstance &, llvm::StringRef) override {
    return newASTConsumer();
  }

private:
  llvm::Function &F;
  RegionCFG &RCFG;
  ASTTree &CombedAST;
  std::unique_ptr<llvm::raw_ostream> O;
};

} // end namespace tooling

} // namespace clang

#endif // REVNGC_CDECOMPILERACTION_H
