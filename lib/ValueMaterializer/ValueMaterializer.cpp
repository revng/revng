/// \file ValueMaterializer.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Support/GenericDomTreeConstruction.h"

#include "revng/Support/Debug.h"
#include "revng/ValueMaterializer/ValueMaterializer.h"

void ValueMaterializer::run() {
  revng_log(ValueMaterializerLogger,
            "Evaluating " << getName(V) << " using " << getName(Context)
                          << " as context");

  DataFlowGraph = DataFlowGraph::fromValue(V, TheLimits);

  revng_log(ValueMaterializerLogger,
            "The data-flow graph has " << DataFlowGraph.size() << " nodes");
  DFGSizeStatitistics.push(DataFlowGraph.size());

  //
  // Prepare the data-flow graph
  //

  // Drop all the SCCs: we want to have a DAG.
  // Note that dropping part of the data flow graph does not affect
  // correctness.
  DataFlowGraph.removeCycles();

  DataFlowGraph.enforceLimits(TheLimits);

  DataFlowGraph.purgeUnreachable();

  computeOracleConstraints();

  applyOracleResultsToDataFlowGraph();

  computeSizeLowerBound();

  electMaterializationStartingPoints();

  Values = DataFlowGraph.materialize(DataFlowGraph.getEntryNode(), MO);
}

void ValueMaterializer::computeOracleConstraints() {
  using namespace llvm;

  switch (Oracle) {
  case Oracle::None:
    break;

  case Oracle::LazyValueInfo:
    for (DataFlowGraph::Node *Node : DataFlowGraph.nodes())
      if (auto *I = dyn_cast<Instruction>(V))
        if (I->getType()->isIntegerTy())
          OracleConstraints[I] = LVI.getConstantRange(I, Context);
    break;

  case Oracle::AdvancedValueInfo:
    std::tie(OracleConstraints,
             CFEG,
             MFIResults) = runAVI(DataFlowGraph, Context, DT, LVI);
    break;

  default:
    revng_abort();
  }
}

void ValueMaterializer::applyOracleResultsToDataFlowGraph() {
  using namespace llvm;
  const DataLayout &DL = getModule(Context)->getDataLayout();

  for (DataFlowGraph::Node *Node : DataFlowGraph.nodes()) {
    auto *V = Node->Value;
    auto *CE = dyn_cast<ConstantExpr>(V);
    if (auto *C = dyn_cast<ConstantInt>(V)) {
      auto Lower = C->getValue();
      auto Upper = Lower + APInt(V->getType()->getIntegerBitWidth(), 1, false);
      Node->OracleRange = ConstantRange(Lower, Upper);
      revng_assert(Lower != Upper);
      revng_assert(Node->OracleRange->size() != 0);
    } else if (isa<ConstantPointerNull>(V) or isa<UndefValue>(V)) {
      // TODO: should we set a range of 0 for ConstantPointerNull
      unsigned BitWidth = 0;
      auto *ValueType = V->getType();
      if (isa<PointerType>(ValueType)) {
        BitWidth = DL.getPointerTypeSizeInBits(ValueType);
      } else if (isa<IntegerType>(ValueType)) {
        BitWidth = V->getType()->getIntegerBitWidth();
      } else {
        revng_abort();
      }
      Node->OracleRange = ConstantRange(BitWidth, false);
    } else if (auto *I = dyn_cast<Instruction>(V)) {
      // Query the oracle
      auto It = OracleConstraints.find(I);
      if (It != OracleConstraints.end())
        Node->OracleRange = It->second;
    } else if (CE != nullptr
               and (CE->getOpcode() == Instruction::IntToPtr
                    or CE->getOpcode() == Instruction::PtrToInt)) {
      // Ignore
    } else if (isa<GlobalVariable>(V)) {
      // Ignore
    } else {
      revng_abort();
    }
  }
}

void ValueMaterializer::computeSizeLowerBound() {
  using namespace llvm;

  // SizeLowerBound is the size of the range we expect *in absence of
  // collisions*. An example of collisions is when you have [0, 10) * 0. You'd
  // expect 10 elements, but you'll get only 1.

  for (DataFlowGraph::Node *Node : post_order(&DataFlowGraph)) {
    auto *V = Node->Value;

    unsigned Opcode = 0;
    if (auto *I = dyn_cast<Instruction>(V))
      Opcode = I->getOpcode();

    auto SuccessorsCount = Node->successorCount();

    Node->SizeLowerBound = DataFlowGraph::Node::MaxSizeLowerBound;

    if (SuccessorsCount) {
      // Is this an instruction whose SizeLowerBound should be the max of the
      // successors?
      // If not, we'll just set SizeLowerBound to 1.
      bool UseMaxOfSuccessors = (isPhiLike(V) or Opcode == Instruction::Add
                                 or Opcode == Instruction::Sub
                                 or Opcode == Instruction::Shl);
      if (UseMaxOfSuccessors) {
        // Use max of successors
        revng_assert(SuccessorsCount > 0);
        Node->SizeLowerBound = (*Node->successors().begin())->SizeLowerBound;
        for (auto *Successor : skip_front(Node->successors())) {
          if (Successor->SizeLowerBound > Node->SizeLowerBound)
            Node->SizeLowerBound = Successor->SizeLowerBound;
        }
      } else {
        // Compute the proudct of successors
        if (SuccessorsCount > 0) {
          OverflowSafeInt<uint64_t> Product = 1;
          for (auto *Successor : Node->successors())
            Product *= Successor->SizeLowerBound;

          if (Product)
            Node->SizeLowerBound = *Product;
        }
      }
    }

    // SizeLowerBound cannot exceed the size of the OracleRange
    if (Node->OracleRange.has_value()) {
      auto OracleRangeSize = Node->OracleRange->size();
      if (OracleRangeSize.ult(Node->SizeLowerBound))
        Node->SizeLowerBound = OracleRangeSize.getLimitedValue();
    }
  }
}

void ValueMaterializer::electMaterializationStartingPoints() {
  using namespace llvm;

  //
  // Choose initial materialization candidates
  //
  using Node = DataFlowGraph::Node;
  for (Node *N : DataFlowGraph.nodes()) {
    if (N->OracleRange.has_value()) {
      auto OracleRangeSize = N->OracleRange->size();

      revng_assert(OracleRangeSize.uge(N->SizeLowerBound));

      if (N->SizeLowerBound == OracleRangeSize) {
        revng_log(ValueMaterializerLogger,
                  "Setting UseOracle for " << N->valueToString()
                                           << " since SizeLowerBound == "
                                              "OracleRangeSize ("
                                           << N->SizeLowerBound << ")");
        N->UseOracle = true;
      }
    }
  }

  //
  // Finalize materialization entry points
  //
  {
    //
    // Exclude candidates that do not dominate all the leaves they reach
    // and compute final set of candidates
    //

    // Prepare dominator tree
    DominatorTreeBase<Node, false> DFGDT;
    DFGDT.recalculate(DataFlowGraph);

    using NodeSet = SmallPtrSet<const Node *, 4>;
    struct NodeData {
      NodeSet ReachableLeaves;
      NodeSet ToMaterialize;
    };
    std::map<const Node *, NodeData> NodesData;

    // Initialize data structure for leaves
    for (Node *N : DataFlowGraph.nodes()) {
      if (not N->hasSuccessors()) {
        NodesData[N].ReachableLeaves = { N };
        NodesData[N].ToMaterialize = { N };
      }
    }

    auto MergeFromSuccessors = [&NodesData](const Node *N, auto &&Getter) {
      auto &CurrentNodeData = NodesData[N];
      for (const Node *Successor : N->successors()) {
        const auto &SuccessorNodeData = NodesData.at(Successor);
        Getter(CurrentNodeData)
          ->insert(Getter(SuccessorNodeData)->begin(),
                   Getter(SuccessorNodeData)->end());
      }
    };

    for (Node *N : post_order(&DataFlowGraph)) {
      auto &CurrentNodeData = NodesData[N];

      // Compute the set of reachable leaves by merging the set of leaves of
      // the successors
      MergeFromSuccessors(N, [](auto &Node) { return &Node.ReachableLeaves; });

      if (N->UseOracle) {
        // Check if it dominates all the reachable leaves, if not, do not use
        // the oracle
        for (const Node *ReachableLeaf : CurrentNodeData.ReachableLeaves) {
          if (not DFGDT.dominates(N, ReachableLeaf)) {
            N->UseOracle = false;
            break;
          }
        }
      }

      // Update the results: either replace results with self or merge results
      // from inputs
      if (N->UseOracle) {
        CurrentNodeData.ToMaterialize = { N };
      } else {
        // Merge results of successors
        MergeFromSuccessors(N, [](auto &Node) { return &Node.ToMaterialize; });
      }
    }

    // Update UseOracle using ToMaterialize of the analysis associated to the
    // root node
    const auto &RootData = NodesData.at(DataFlowGraph.getEntryNode());
    for (Node *N : DataFlowGraph.nodes())
      N->UseOracle = RootData.ToMaterialize.contains(N);
  }
}
