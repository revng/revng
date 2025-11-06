#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/STLExtras.h"

#include "revng/ADT/Concepts.h"
#include "revng/Model/FunctionTags.h"

// We use a template here in order to instantiate `FunctionType` both as
// `Function` and `const Function`
template<ConstOrNot<llvm::Module> ModuleType>
inline typename std::conditional_t<std::is_const_v<ModuleType>,
                                   const llvm::Function,
                                   llvm::Function> *
getUniqueFunctionWithTag(FunctionTags::Tag &MarkerFunctionTag, ModuleType *M) {
  using FunctionType = typename std::conditional_t<std::is_const_v<ModuleType>,
                                                   const llvm::Function,
                                                   llvm::Function>;
  FunctionType *MarkerCallFunction = nullptr;

  // We could early break from this loop but we would loose the ability of
  // asserting that a single marker function is present
  for (FunctionType &F : MarkerFunctionTag.functions(M)) {
    revng_assert(not MarkerCallFunction);
    MarkerCallFunction = &F;
  }

  return MarkerCallFunction;
}

/// A class that wraps all the logic for injecting goto edges and scope closer
/// edges on LLVM IR. Such edges are then necessary for the ScopeGraph view on
/// LLVM IR
class ScopeGraphBuilder {
private:
  llvm::Function *ScopeCloserFunction = nullptr;
  llvm::Function *GotoBlockFunction = nullptr;

public:
  ScopeGraphBuilder(llvm::Function *F);

public:
  void makeGoto(llvm::BasicBlock *GotoBlock) const;
  void eraseGoto(llvm::BasicBlock *GotoBlock) const;
  void addScopeCloser(llvm::BasicBlock *Source, llvm::BasicBlock *Target) const;

  /// Helper method which erase a `scope_closer`, and returns the block which
  /// was target of the `scope_closer`
  llvm::BasicBlock *eraseScopeCloser(llvm::BasicBlock *Source) const;

  /// With the usage of this helper,  all the successor in the `Terminator` of
  /// the `Source` block pointing to `Target` will be redirected to the newly
  /// inserted `goto` block
  llvm::BasicBlock *makeGotoEdge(llvm::BasicBlock *Source,
                                 llvm::BasicBlock *Target) const;
};

template<ConstOrNot<llvm::BasicBlock> BasicBlockType>
llvm::SmallVector<std::conditional_t<std::is_const_v<BasicBlockType>,
                                     const llvm::Instruction,
                                     llvm::Instruction> *,
                  2>
getLast2InstructionsBeforeTerminator(BasicBlockType *BB);

/// Helper function to retrieve the `BasicBlock` target of the marker
llvm::BasicBlock *getScopeCloserTarget(const llvm::BasicBlock *BB);

/// Helper function to determine if `BB` contains a `scope_closer` marker
bool isScopeCloserBlock(const llvm::BasicBlock *BB);

/// Helper function to determine if `BB` contains a `goto_block` marker
bool isGotoBlock(const llvm::BasicBlock *BB);

void verifyScopeGraphAnnotationsImpl(FunctionTags::Tag &Tag,
                                     const llvm::BasicBlock *BB);

void verifyScopeGraphAnnotations(const llvm::BasicBlock *BB);

/// Helper function which substitutes some successors in the `Terminator` with
/// `NewTarget`
template<RangeOf<llvm::BasicBlock *> Container>
void replaceSuccessors(llvm::Instruction *Terminator,
                       Container &SuccessorsToRemove,
                       llvm::BasicBlock *NewTarget) {
  for (llvm::BasicBlock *Successor : SuccessorsToRemove) {
    Terminator->replaceSuccessorWith(Successor, NewTarget);
  }
}
