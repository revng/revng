#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/Instructions.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"

#include "revng/EarlyFunctionAnalysis/CFGStringMap.h"
#include "revng/EarlyFunctionAnalysis/ControlFlowGraph.h"
#include "revng/Model/Binary.h"
#include "revng/Model/IRHelpers.h"
#include "revng/Pipes/IRHelpers.h"
#include "revng/Pipes/Kinds.h"
#include "revng/Support/Assert.h"
#include "revng/Support/IRHelpers.h"
#include "revng/Support/MetaAddress.h"
#include "revng/Support/YAMLTraits.h"
#include "revng/TupleTree/TupleTree.h"

template<typename T>
concept ControlFlowGraphCacheTraits = requires {
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
    T::getFunctionAddress(std::declval<typename T::Function>())
  } -> std::same_as<MetaAddress>;
};

/// The BasicControlFlowGraphCache is implemented as a class template customised
/// via a traits class in order to enable reuse for both LLVM IR and MLIR.
template<ControlFlowGraphCacheTraits Traits>
class BasicControlFlowGraphCache {
  using BasicBlock = typename Traits::BasicBlock;
  using Function = typename Traits::Function;
  using CallInst = typename Traits::CallInst;

  const revng::pipes::CFGMap &CFGs;
  std::map<MetaAddress, TupleTree<efa::ControlFlowGraph>> Deserialized;

public:
  BasicControlFlowGraphCache(const revng::pipes::CFGMap &CFGs) : CFGs(CFGs) {}

public:
  void set(TupleTree<efa::ControlFlowGraph> &&New) {
    Deserialized[New->Entry()] = std::move(New);
  }

public:
  const efa::ControlFlowGraph &getControlFlowGraph(const MetaAddress &Address) {
    auto It = Deserialized.find(Address);
    if (It != Deserialized.end())
      return *It->second.get();

    TupleTree<efa::ControlFlowGraph> &Result = Deserialized[Address];
    Result = TupleTree<efa::ControlFlowGraph>::deserialize(CFGs.at(Address))
               .get();
    return *Result.get();
  }

  const efa::ControlFlowGraph &getControlFlowGraph(const Function Function) {
    return getControlFlowGraph(Traits::getFunctionAddress(Function));
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
    const efa::ControlFlowGraph &FM = getControlFlowGraph(ParentFunction);
    const efa::BasicBlock &Block = FM.Blocks().at(BlockAddress);

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
  inline model::TypeDefinitionPath
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

  static const model::Function *getModelFunction(const model::Binary &Binary,
                                                 const llvm::Function *F) {
    return llvmToModelFunction(Binary, *F);
  }

  static MetaAddress getFunctionAddress(const llvm::Function *F) {
    return getMetaAddressOfIsolatedFunction(*F);
  }
};

using ControlFlowGraphCache = BasicControlFlowGraphCache<LLVMIRMetadataTraits>;

class ControlFlowGraphCachePass : public llvm::ImmutablePass {
public:
  static char ID;

private:
  ControlFlowGraphCache Cache;

public:
  ControlFlowGraphCachePass(const revng::pipes::CFGMap &CFGs) :
    llvm::ImmutablePass(ID), Cache(CFGs) {}
  ControlFlowGraphCache &get() { return Cache; }
};

class ControlFlowGraphCacheAnalysis
  : public llvm::AnalysisInfoMixin<ControlFlowGraphCacheAnalysis> {
  friend llvm::AnalysisInfoMixin<ControlFlowGraphCacheAnalysis>;

private:
  ControlFlowGraphCache Cache;
  static llvm::AnalysisKey Key;

public:
  using Result = ControlFlowGraphCache;

public:
  ControlFlowGraphCache *runOnModule(llvm::Module &M) { return &Cache; }
};
