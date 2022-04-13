#pragma once

//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

#include <cstdio>

#include "llvm/Analysis/ScalarEvolution.h"

#include "clang/Frontend/ASTConsumers.h"
#include "clang/Frontend/FrontendAction.h"

#include "revng-c/DataLayoutAnalysis/DLALayouts.h"
#include "revng-c/MarkForSerialization/MarkForSerializationFlags.h"

namespace llvm {

class ScalarEvolution;

} // end namespace llvm

class ASTTree;

namespace clang {

class CompilerInstance;

namespace tooling {

class CDecompilerAction : public ASTFrontendAction {

private:
  using PHIIncomingMap = SmallMap<llvm::PHINode *, unsigned, 4>;
  using BBPHIMap = SmallMap<llvm::BasicBlock *, PHIIncomingMap, 4>;
  using DuplicationMap = std::map<const llvm::BasicBlock *, size_t>;

public:
  CDecompilerAction(llvm::Function &F,
                    ASTTree &CombedAST,
                    BBPHIMap &BlockToPHIIncoming,
                    const dla::ValueLayoutMap *LM,
                    llvm::ScalarEvolution *SCEV,
                    const SerializationMap &M,
                    std::unique_ptr<llvm::raw_ostream> O) :
    F(F),
    CombedAST(CombedAST),
    BlockToPHIIncoming(BlockToPHIIncoming),
    LayoutMap(LM),
    SE(SCEV),
    Mark(M),
    O(std::move(O)) {}

public:
  std::unique_ptr<clang::ASTConsumer> newASTConsumer();

  virtual std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(CompilerInstance &, llvm::StringRef) override;

private:
  llvm::Function &F;
  ASTTree &CombedAST;
  BBPHIMap &BlockToPHIIncoming;
  const dla::ValueLayoutMap *LayoutMap;
  llvm::ScalarEvolution *SE;
  const SerializationMap &Mark;
  std::unique_ptr<llvm::raw_ostream> O;
};

} // end namespace tooling

} // namespace clang
