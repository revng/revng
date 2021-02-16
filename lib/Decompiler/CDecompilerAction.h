#pragma once

//
// Copyright rev.ng Srls. See LICENSE.md for details.
//

#include <cstdio>

#include "llvm/Analysis/ScalarEvolution.h"

#include "clang/Frontend/ASTConsumers.h"
#include "clang/Frontend/FrontendAction.h"

#include "revng-c/Decompiler/DLALayouts.h"
#include "revng-c/Decompiler/MarkForSerialization.h"

#include "CDecompilerBeautify.h"

namespace model {

class Binary;

} // end namespace model

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
  CDecompilerAction(const model::Binary &Model,
                    llvm::Function &F,
                    ASTTree &CombedAST,
                    BBPHIMap &BlockToPHIIncoming,
                    const dla::ValueLayoutMap *LM,
                    llvm::ScalarEvolution *SCEV,
                    const SerializationMap &M,
                    std::unique_ptr<llvm::raw_ostream> O) :
    Model(Model),
    F(F),
    CombedAST(CombedAST),
    BlockToPHIIncoming(BlockToPHIIncoming),
    LayoutMap(LM),
    SE(SCEV),
    Mark(M),
    O(std::move(O)) {}

public:
  std::unique_ptr<ASTConsumer> newASTConsumer();

  virtual std::unique_ptr<ASTConsumer>
  CreateASTConsumer(CompilerInstance &, llvm::StringRef) override;

private:
  const model::Binary &Model;
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
