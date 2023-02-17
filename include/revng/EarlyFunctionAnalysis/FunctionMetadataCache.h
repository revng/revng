#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/Instructions.h"
#include "llvm/IR/PassManager.h"

#include "revng/EarlyFunctionAnalysis/FunctionMetadata.h"
#include "revng/Model/Binary.h"
#include "revng/Model/IRHelpers.h"
#include "revng/Support/Assert.h"
#include "revng/Support/IRHelpers.h"
#include "revng/Support/MetaAddress.h"
#include "revng/TupleTree/TupleTree.h"

namespace detail {

inline TupleTree<efa::FunctionMetadata>
extractFunctionMetadata(llvm::MDNode *MD) {
  using namespace llvm;

  efa::FunctionMetadata FM;
  revng_assert(MD != nullptr);
  const MDOperand &Op = MD->getOperand(0);
  revng_assert(isa<MDString>(Op));

  StringRef YAMLString = cast<MDString>(Op)->getString();
  auto MaybeParsed = TupleTree<efa::FunctionMetadata>::deserialize(YAMLString);
  revng_assert(MaybeParsed);
  MaybeParsed->verify();
  return std::move(MaybeParsed.get());
}

inline TupleTree<efa::FunctionMetadata>
extractFunctionMetadata(const llvm::Function *F) {
  auto *MDNode = F->getMetadata(FunctionMetadataMDName);
  return detail::extractFunctionMetadata(MDNode);
}

inline TupleTree<efa::FunctionMetadata>
extractFunctionMetadata(const llvm::BasicBlock *BB) {
  auto *MDNode = BB->getTerminator()->getMetadata(FunctionMetadataMDName);
  return detail::extractFunctionMetadata(MDNode);
}

} // namespace detail

class FunctionMetadataCache {
private:
  std::map<const llvm::Value *, efa::FunctionMetadata> FunctionCache;

public:
  const efa::FunctionMetadata &
  getFunctionMetadata(const llvm::Function *Function) {
    if (auto Iter = FunctionCache.find(Function); Iter == FunctionCache.end()) {
      efa::FunctionMetadata FM = *detail::extractFunctionMetadata(Function)
                                    .get();
      FunctionCache.try_emplace(Function, FM);
    }

    return FunctionCache.find(Function)->second;
  }

  const efa::FunctionMetadata &getFunctionMetadata(const llvm::BasicBlock *BB) {
    if (auto Iter = FunctionCache.find(BB); Iter == FunctionCache.end()) {
      efa::FunctionMetadata FM = *detail::extractFunctionMetadata(BB).get();
      FunctionCache.try_emplace(BB, FM);
    }

    return FunctionCache.find(BB)->second;
  }

  /// Given a Call instruction and the model type of its parent function, return
  /// the edge on the model that represents that call (std::nullopt if this
  /// doesn't exist) and the BasicBlockID associated to the call-site.
  inline std::pair<std::optional<efa::CallEdge>, BasicBlockID>
  getCallEdge(const model::Binary &Binary, const llvm::CallInst *Call) {
    using namespace llvm;

    auto
      BlockAddress = fromStringMetadata<BasicBlockID>(Call,
                                                      CallerBlockStartMDName);
    if (not BlockAddress.isValid())
      return { std::nullopt, BlockAddress };

    auto *ParentFunction = Call->getParent()->getParent();
    const efa::FunctionMetadata &FM = getFunctionMetadata(ParentFunction);
    const efa::BasicBlock &Block = FM.ControlFlowGraph().at(BlockAddress);

    // Find the call edge
    efa::CallEdge *ModelCall = nullptr;
    for (auto &Edge : Block.Successors()) {
      if (auto *CE = dyn_cast<efa::CallEdge>(Edge.get())) {
        revng_assert(ModelCall == nullptr);
        ModelCall = CE;
      }
    }
    revng_assert(ModelCall != nullptr);

    return { *ModelCall, Block.ID() };
  }

  /// \return the prototype associated to a CallInst.
  ///
  /// \note If the model type of the parent function is not provided, this will
  /// be
  ///       deduced using the Call instruction's parent function.
  ///
  /// \note If the callsite has no associated prototype, e.g. the called
  /// functions
  ///       is not an isolated function, a null pointer is returned.
  inline model::TypePath
  getCallSitePrototype(const model::Binary &Binary,
                       const llvm::CallInst *Call,
                       const model::Function *ParentFunction = nullptr) {
    if (not ParentFunction)
      ParentFunction = llvmToModelFunction(Binary, *Call->getFunction());

    if (not ParentFunction)
      return {};

    const auto &[Edge, BlockAddress] = getCallEdge(Binary, Call);
    if (not Edge)
      return {};

    return getPrototype(Binary, ParentFunction->Entry(), BlockAddress, *Edge);
  }
};

class FunctionMetadataCachePass : public llvm::ImmutablePass {
public:
  static char ID;

private:
  FunctionMetadataCache Cache;

public:
  FunctionMetadataCachePass() : llvm::ImmutablePass(ID) {}
  FunctionMetadataCache &get() { return Cache; }
};

class FunctionMetadataCacheAnalysis
  : public llvm::AnalysisInfoMixin<FunctionMetadataCacheAnalysis> {
  friend llvm::AnalysisInfoMixin<FunctionMetadataCacheAnalysis>;

private:
  FunctionMetadataCache Cache;
  static llvm::AnalysisKey Key;

public:
  using Result = FunctionMetadataCache;

public:
  FunctionMetadataCache *runOnModule(llvm::Module &M) { return &Cache; }
};
