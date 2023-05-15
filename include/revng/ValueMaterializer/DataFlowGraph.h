#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <optional>

#include "llvm/IR/Constants.h"
#include "llvm/Support/DOTGraphTraits.h"

#include "revng/ADT/ConstantRangeSet.h"
#include "revng/ADT/GenericGraph.h"
#include "revng/ADT/RecursiveCoroutine.h"
#include "revng/BasicAnalyses/MaterializedValue.h"
#include "revng/Support/IRHelpers.h"
#include "revng/ValueMaterializer/Helpers.h"
#include "revng/ValueMaterializer/MemoryOracle.h"

class DataFlowNode {
public:
  static constexpr uint64_t
    MaxSizeLowerBound = std::numeric_limits<uint64_t>::max();

public:
  llvm::Value *Value = nullptr;
  std::optional<ConstantRangeSet> OracleRange;
  uint64_t SizeLowerBound = 1;
  bool UseOracle = false;

public:
  DataFlowNode(llvm::Value *V) : Value(V) {}

public:
  void dump() debug_function { dump(dbg); }

  template<typename T>
  void dump(T &Stream) const {
    using namespace llvm;

    Stream << "Value: ";
    if (auto *CI = dyn_cast<ConstantInt>(Value)) {
      Stream << aviFormatter(CI->getValue());
    } else if (auto *CE = dyn_cast<ConstantExpr>(Value)) {
      Stream << CE->getOpcodeName();
    } else if (auto *I = dyn_cast<Instruction>(Value)) {
      Stream << getName(Value);
      Stream << " (" << I->getOpcodeName() << ")";
    } else if (Value->hasName()) {
      Stream << Value->getName().str();
    }

    Stream << "\n";

    if (not isa<ConstantInt>(Value) and OracleRange.has_value()) {
      Stream << "OracleRange: ";
      if (OracleRange->isFullSet())
        Stream << "full set";
      else
        OracleRange->dump(Stream, aviFormatter);
      Stream << "\n";
    }

    Stream << "SizeLowerBound: ";
    if (SizeLowerBound == MaxSizeLowerBound)
      Stream << "max";
    else
      Stream << SizeLowerBound;

    Stream << "\n";
  }

  std::string valueToString() const;
};

class DataFlowGraph : public GenericGraph<BidirectionalNode<DataFlowNode>> {
public:
  using Node = BidirectionalNode<DataFlowNode>;
  using Base = GenericGraph<Node>;

  class Limits {
  public:
    static constexpr const auto Max = std::numeric_limits<unsigned>::max();

  private:
    unsigned MaxPhiLike = Max;
    unsigned MaxLoad = Max;

  public:
    Limits() = default;
    Limits(unsigned MaxPhiLike, unsigned MaxLoad) :
      MaxPhiLike(MaxPhiLike), MaxLoad(MaxLoad) {}

  public:
    bool consumePhiLike() {
      if (MaxPhiLike == 0)
        return false;

      --MaxPhiLike;
      return true;
    }

    bool consumeLoad() {
      if (MaxLoad == 0)
        return false;

      --MaxLoad;
      return true;
    }
  };

private:
  using NodeValuesMap = std::map<Node *, std::optional<MaterializedValues>>;

private:
  llvm::DenseMap<llvm::Value *, Node *> NodeMap;

public:
  static DataFlowGraph fromValue(llvm::Value *Root, Limits Limits) {
    DataFlowGraph Result;
    Result.setEntryNode(Result.processValue(Root, Limits));
    return Result;
  }

public:
  // \return if materialization is successful, an non-empty optional composed by
  // a list of values and a list of read memory areas
  std::optional<MaterializedValues> materialize(Node *N,
                                                MemoryOracle &MO) const {
    using Map = std::map<Node *, std::optional<MaterializedValues>>;
    Map Results;
    return materializeImpl(N, MO, Results);
  }

public:
  void removeCycles();
  void purgeUnreachable();
  void enforceLimits(Limits TheLimits);

private:
  RecursiveCoroutine<std::optional<MaterializedValues>>
  materializeImpl(Node *N, MemoryOracle &MO, NodeValuesMap &Results) const;

  /// \param Limits best effort limits for the creation of the data-flow graph.
  ///        In order to reliably enforce these limits, invoke enforceLimits at
  ///        the end.
  RecursiveCoroutine<Node *> processValue(llvm::Value *V, Limits Limits);

public:
  void dump() const;
};

template<>
struct llvm::DOTGraphTraits<const DataFlowGraph::Base *>
  : public llvm::DefaultDOTGraphTraits {

  DOTGraphTraits(bool IsSimple = false) : DefaultDOTGraphTraits(IsSimple) {}

  static std::string getGraphProperties(const DataFlowGraph::Base *);

  static std::string getNodeLabel(const DataFlowGraph::Node *Node,
                                  const DataFlowGraph::Base *Graph);

  static std::string getNodeAttributes(const DataFlowGraph::Node *Node,
                                       const DataFlowGraph::Base *Graph);
};

template<>
struct llvm::DOTGraphTraits<DataFlowGraph::Base *>
  : public llvm::DOTGraphTraits<const DataFlowGraph::Base *> {};
