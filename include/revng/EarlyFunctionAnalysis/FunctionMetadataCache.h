#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/Instructions.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"

#include "revng/EarlyFunctionAnalysis/FunctionMetadata.h"
#include "revng/Model/Binary.h"
#include "revng/Model/IRHelpers.h"
#include "revng/Pipes/IRHelpers.h"
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
  revng_assert(MaybeParsed and MaybeParsed->verify());
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

template<typename T>
concept FunctionMetadataCacheTraits = requires {
  typename T::BasicBlock;
  typename T::Function;
  typename T::CallInst;
  typename T::KeyType;

  {
    T::getKey(std::declval<typename T::Function>())
  } -> std::same_as<typename T::KeyType>;

  {
    T::getKey(std::declval<typename T::BasicBlock>())
  } -> std::same_as<typename T::KeyType>;

  {
    T::getLocation(std::declval<typename T::CallInst>())
  } -> std::same_as<
    std::optional<pipeline::Location<decltype(revng::ranks::Instruction)>>>;

  {
    T::getFunction(std::declval<typename T::CallInst>())
  } -> std::same_as<typename T::Function>;

  {
    T::getModelFunction(std::declval<const model::Binary &>(),
                        std::declval<typename T::Function>())
  } -> std::same_as<const model::Function *>;

  {
    T::extractFunctionMetadata(std::declval<typename T::Function>())
  } -> std::same_as<TupleTree<efa::FunctionMetadata>>;
};

/// The BasicFunctionMetadataCache is implemented as a class template customised
/// via a traits class in order to enable reuse for both LLVM IR and MLIR.
template<FunctionMetadataCacheTraits Traits>
class BasicFunctionMetadataCache {
  using BasicBlock = typename Traits::BasicBlock;
  using Function = typename Traits::Function;
  using CallInst = typename Traits::CallInst;

  std::map<typename Traits::KeyType, efa::FunctionMetadata> FunctionCache;

public:
  const efa::FunctionMetadata &getFunctionMetadata(Function Function) {
    typename Traits::KeyType Key = Traits::getKey(Function);

    auto Iterator = FunctionCache.find(Key);
    if (Iterator != FunctionCache.end())
      return Iterator->second;

    efa::FunctionMetadata FM = *Traits::extractFunctionMetadata(Function).get();
    return FunctionCache.try_emplace(Key, std::move(FM)).first->second;
  }

  const efa::FunctionMetadata &getFunctionMetadata(BasicBlock BB) {
    typename Traits::KeyType Key = Traits::getKey(BB);

    auto Iterator = FunctionCache.find(Key);
    if (Iterator != FunctionCache.end())
      return Iterator->second;

    efa::FunctionMetadata FM = *Traits::extractFunctionMetadata(BB).get();
    return FunctionCache.try_emplace(Key, std::move(FM)).first->second;
  }

  /// Given a Call instruction and the model type of its parent function, return
  /// the edge on the model that represents that call (std::nullopt if this
  /// doesn't exist) and the BasicBlockID associated to the call-site.
  inline std::pair<std::optional<efa::CallEdge>, BasicBlockID>
  getCallEdge(const model::Binary &Binary, CallInst Call) {
    using namespace llvm;

    auto MaybeLocation = Traits::getLocation(Call);

    if (not MaybeLocation)
      return { std::nullopt, BasicBlockID::invalid() };

    auto BlockAddress = MaybeLocation->parent().back();

    Function ParentFunction = Traits::getFunction(Call);
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
  model::TypePath
  getCallSitePrototype(const model::Binary &Binary,
                       CallInst Call,
                       const model::Function *ParentFunction = nullptr) {
    if (not ParentFunction)
      ParentFunction = Traits::getModelFunction(Binary,
                                                Traits::getFunction(Call));

    if (not ParentFunction)
      return {};

    const auto &[Edge, BlockAddress] = getCallEdge(Binary, Call);

    if (not Edge)
      return {};

    return getPrototype(Binary, ParentFunction->Entry(), BlockAddress, *Edge);
  }
};

class LLVMIRMetadataTraits {
  using Location = pipeline::Location<decltype(revng::ranks::Instruction)>;

public:
  using BasicBlock = const llvm::BasicBlock *;
  using Function = const llvm::Function *;
  using CallInst = const llvm::CallInst *;
  using KeyType = const llvm::Value *;

  static KeyType getKey(Function F) { return F; }
  static KeyType getKey(BasicBlock BB) { return BB; }

  static const llvm::Function *getFunction(const llvm::Instruction *const I) {
    return I->getFunction();
  }

  static std::optional<Location> getLocation(const llvm::Instruction *const I) {
    return ::getLocation(I);
  }

  static TupleTree<efa::FunctionMetadata>
  extractFunctionMetadata(const llvm::Function *F) {
    return detail::extractFunctionMetadata(F);
  }

  static TupleTree<efa::FunctionMetadata>
  extractFunctionMetadata(const llvm::BasicBlock *BB) {
    return detail::extractFunctionMetadata(BB);
  }

  static const model::Function *getModelFunction(const model::Binary &Binary,
                                                 const llvm::Function *F) {
    return llvmToModelFunction(Binary, *F);
  }
};
using FunctionMetadataCache = BasicFunctionMetadataCache<LLVMIRMetadataTraits>;

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
