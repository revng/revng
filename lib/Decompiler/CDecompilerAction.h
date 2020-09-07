#pragma once

//
// Copyright rev.ng Srls. See LICENSE.md for details.
//

#include <cstdio>

#include "clang/Frontend/ASTConsumers.h"
#include "clang/Frontend/FrontendAction.h"

#include "revng-c/Decompiler/DLALayouts.h"
#include "revng-c/Decompiler/MarkForSerialization.h"

#include "CDecompilerBeautify.h"

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
                    const SerializationMap &M,
                    std::unique_ptr<llvm::raw_ostream> O) :
    F(F),
    CombedAST(CombedAST),
    BlockToPHIIncoming(BlockToPHIIncoming),
    LayoutMap(LM),
    Mark(M),
    O(std::move(O)) {}

public:
  std::unique_ptr<ASTConsumer> newASTConsumer();

  virtual std::unique_ptr<ASTConsumer>
  CreateASTConsumer(CompilerInstance &, llvm::StringRef) override;

private:
  llvm::Function &F;
  ASTTree &CombedAST;
  BBPHIMap &BlockToPHIIncoming;
  const dla::ValueLayoutMap *LayoutMap;
  const SerializationMap &Mark;
  std::unique_ptr<llvm::raw_ostream> O;
};

} // end namespace tooling

} // namespace clang
