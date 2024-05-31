//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

// This file contains the implementation of the Cycle Equivalence Analysis,
// taken from "The Program Structure Tree - Richard Johnson, David Pearson,
// Keshav Pingali - 1994" https://dl.acm.org/doi/pdf/10.1145/178243.178258.

#include <iterator>
#include <limits>
#include <map>

#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/PostOrderIterator.h"

#include "revng/ADT/GenericGraph.h"
#include "revng/Support/Debug.h"
#include "revng/Support/IRHelpers.h"

#include "revng-c/RestructureCFG/CycleEquivalenceAnalysis.h"

using namespace llvm;
using namespace llvm::cl;

// Debug logger.
Logger<> CycleEquivalenceAnalysisLogger("cycle-equivalence");

template<class GraphT, class GT>
void CycleEquivalenceAnalysis<GraphT, GT>::insertEdge(BlockEdgeDescriptor E,
                                                      EdgeLabel *Label,
                                                      bool IsInverted,
                                                      BracketDescriptor BD) {

  // We do not want to insert `Fake` edges in the equivalence classes. Such
  // edges can be both the capping backedges and the additional `exit` ->
  // `entry` bracket
  revng_assert(Label->kind() != OriginalEdgeKind::Invalid);
  if (Label->kind() == OriginalEdgeKind::Fake)
    return;

  // We marshal the `EdgeDescriptor` object to unwrap the underlying edge, not
  // wrapped in any `GenericGraph` node, and we invert the direction of the edge
  // if are traversing it in the inverse direction (wrt the direction of the
  // original directed edge) during the current exploration
  using EdgeDescriptor = CycleEquivalenceClass<NodeT>::EdgeDescriptor;
  EdgeDescriptor NakedEdge;
  if (not IsInverted) {
    NakedEdge = EdgeDescriptor({ std::get<0>(E)->getBlock(),
                                 std::get<1>(E)->getBlock(),
                                 Label->succNum() });
  } else {
    NakedEdge = EdgeDescriptor({ std::get<1>(E)->getBlock(),
                                 std::get<0>(E)->getBlock(),
                                 Label->succNum() });
  }

  size_t EquivalenceClassIndex = SizeTMaxValue;
  auto MapIt = BracketToClass.find(BD);
  if (MapIt != BracketToClass.end()) {

    // The cycle equivalence class already exists
    EquivalenceClassIndex = MapIt->second;
    revng_assert(CycleEquivalenceClasses.size() > EquivalenceClassIndex);
  } else {

    // We need to create the cycle equivalence class
    EquivalenceClassIndex = CycleEquivalenceClasses.size();
    BracketToClass.emplace(BD, EquivalenceClassIndex);
    ClassToBracketDescriptor.emplace(EquivalenceClassIndex, BD);
    CycleEquivalenceClasses.emplace_back(EquivalenceClassIndex);
  }

  auto &EClass = CycleEquivalenceClasses[EquivalenceClassIndex];
  EClass.insert(NakedEdge);

  EdgeToCycleEquivalenceClassIDMap.insert(NakedEdge, EquivalenceClassIndex);
}

template<class GraphT, class GT>
std::string CycleEquivalenceAnalysis<GraphT, GT>::print() {

  std::string Output;

  // Print the Brackets Analysis results
  Output += "\nOrdered Bracket Analysis Results:\n";
  for (auto &EClass : CycleEquivalenceClasses) {
    Output += EClass.print();
  }

  // Print the correspondence between the Equivalence Class and the Bracket
  // Descriptor which gave origin to it
  Output += "\nClass Bracket Correspondence:\n";
  for (const auto &[Index, BD] : ClassToBracketDescriptor) {
    auto &Edge = BD.first;
    auto &Size = BD.second;
    Output += std::to_string(Index) + " => " + "("
              + std::get<0>(Edge)->getName().str() + " <-> "
              + std::get<1>(Edge)->getName().str() + ","
              + std::to_string(std::get<2>(BD.first)) + "), "
              + std::to_string(Size) + "\n";
  }

  return Output;
}

template<class GraphT, class GT>
bool CycleEquivalenceAnalysis<GraphT, GT>::isBackedge(BlockEdgeDescriptor &ED) {

  size_t SourceDFSNum = std::get<0>(ED)->getDFSNum();
  size_t TargetDFSNum = std::get<1>(ED)->getDFSNum();

  // The equal is very important, or we exclude the self loop backedges. When we
  // have a self loop indeed, the source node is the same as the target node,
  // and therefore `TargetDFSNum` will be equal to `SourceDFSNum`. If the
  // comparison does not include the equal, a self loop edge will erroneously
  // not be marked as a backedge.
  return TargetDFSNum <= SourceDFSNum;
}

template<class GraphT, class GT>
void CycleEquivalenceAnalysis<GraphT, GT>::computeDFSAndSpanningTree(BlockGraph
                                                                       &Graph) {
  using ExtType = llvm::df_iterator_default_set<BlockNode *>;
  using UGT = llvm::GraphTraits<llvm::Undirected<BlockNode *>>;
  using udf_iterator = llvm::df_iterator<BlockNode *, ExtType, false, UGT>;

  BlockNode *EntryNode = Graph.getEntryNode();
  auto It = udf_iterator::begin(EntryNode);
  auto End = udf_iterator::end(EntryNode);

  size_t DFSNum = 0;
  for (; It != End; It++) {
    BlockNode *CurrentNode = *It;

    // We assign the DFSNum for `CurrentNode`
    CurrentNode->setDFSNum(DFSNum);
    DFSNum++;

    // Each time we reach a new node during the `DFS` visit, we search for the
    // edge that brougth us here, and that we therefore need to insert into the
    // spanning tree.
    // Retrieve the last-but-one element on the `VisitStack`, which is the
    // candidate as the source node of the edge we are interested into adding to
    // the spanning tree.
    auto VisitStackSize = It.getPathLength();

    // When we have a single node on the `VisitStack`, we cannot assign an edge
    // to the spanning tree
    if (VisitStackSize < 2)
      continue;

    BlockNode *SourceNode = It.getPath(VisitStackSize - 2);

    // Search for the edge which connects `Source` and `CurrentNode`, which is
    // the edge composing the tree edge of the current spanning tree
    // exploration.
    // Please be aware of the following implementation detail: in the code
    // below, we do not take into account the fact that there may be multiple
    // edges connecting the same pair of nodes, but in the body of the
    // `FindCurrentNode` lambda, we just check the destination node.
    // However, this is in the end operates correctly, for the following
    // reasons related to the implementation details:
    // 1) In the `Undirected` graph, the edges that were originally forward,
    //    are always traversed first, therefore when we do the `find_if` over
    //    the `ChildrenEdgesRange` (which is a vector under the hood), we are
    //    sure to encounter the first occurrence, and this is what brought use
    //    here the first time in the DFS.
    // 2) Even in the situation where in the original graph we had multiple
    //    forward edges from the same pair of nodes, say A and B, the DFS only
    //    ever explores the first edge, because for all the following cases B
    //    has already been visited and the DFS doesn't continue on B, and
    //    therefore will not encounter the subsequent edge connecting A to B. So
    //    it never happens that the search of edges we are doing really wants to
    //    see any other edge beyond the first in the aforementioned order.
    using UndirectedGraphT = llvm::Undirected<BlockNode *>;
    auto
      ChildrenEdgesRange = llvm::children_edges<UndirectedGraphT>(SourceNode);
    const auto FindCurrentNode = [&CurrentNode](const auto &Pair) {
      return Pair.Neighbor == CurrentNode;
    };
    auto EdgeIt = llvm::find_if(ChildrenEdgesRange, FindCurrentNode);
    revng_assert(EdgeIt != ChildrenEdgesRange.end());
    EdgeIt->Label->setType(SpanningTreeEdgeKind::TreeEdge);
  }

  // All the remaining edges that have not been marked as `TreeEdge`s, are the
  // `BackEdge`s. Mind that we cannot perform this assignment during the
  // previous visit itself, because we first need to visit all the edges to mark
  // the `TreeEdge`s
  for (BlockNode *CurrentNode :
       llvm::depth_first(llvm::Undirected(Graph.getEntryNode()))) {
    for (auto [Neighbor, Label] :
         llvm::children_edges<llvm::Undirected<BlockNode *>>(CurrentNode)) {
      if (Label->type() == SpanningTreeEdgeKind::Invalid) {
        Label->setType(SpanningTreeEdgeKind::BackEdge);
      }
    }
  }
}

/// Helper function to materialize the capping backedges
template<class GraphT, class GT>
void CycleEquivalenceAnalysis<GraphT, GT>::insertCappingBackedges(BlockGraph
                                                                    &Graph) {
  // Highest, i.e., closer to the spanning tree root, reached DFSNum for each
  // `BB` (and the corresponding `BlockNode *`). Please note that for higher
  // nodes, the DFSNum is smaller. This use of the highest term is made to be
  // coherent with the terminology used in the paper describing the cycle
  // equivalence algorithm. It is highly suggested to read it in order to get
  // confident with the terminology.
  std::map<BlockNode *, std::pair<size_t, BlockNode *>> HighestReachedDFSNum;

  llvm::SmallVector<BlockNode *> PostOrder;
  for (BlockNode *BB :
       llvm::post_order(llvm::Undirected<BlockNode *>(Graph.getEntryNode()))) {
    PostOrder.push_back(BB);
  }

  for (BlockNode *BB : PostOrder) {

    // 1) Compute the highest reached node (`HI0Node`) from a backedge starting
    //    from the current `Block`, which has `HI0` `DFSNum`
    size_t HI0 = SizeTMaxValue;
    BlockNode *HI0Node = nullptr;

    // We are only interested in looking at backedges
    for (const auto &[BackwardEdge, _, __] : getBackwardEdges(BB)) {
      BlockNode *Target = std::get<1>(BackwardEdge);
      size_t NewMin = std::min(HI0, Target->getDFSNum());
      if (NewMin < HI0) {
        HI0Node = Target;
      }
      HI0 = NewMin;
    }

    // 2) Compute the highest reached node (`HI1Node`) for any child of the
    //    current `Block` (represented by the `HI1` index), and the `HI1Child`,
    //    which is the child from which the subtree that reaches the `HI1` node
    //    originates
    size_t HI1 = SizeTMaxValue;
    BlockNode *HI1Node = nullptr;
    BlockNode *HI1Child = nullptr;
    for (const auto &[ForwardEdge, _, __] : getForwardEdges(BB)) {
      BlockNode *Target = std::get<1>(ForwardEdge);
      size_t NewMin = std::min(HI1, HighestReachedDFSNum.at(Target).first);
      if (NewMin < HI1) {
        HI1Node = HighestReachedDFSNum.at(Target).second;
        HI1Child = Target;
      }
      HI1 = NewMin;
    }

    // 3) Assign the highest reached node for the current `Block`
    if (HI0 < HI1) {
      HighestReachedDFSNum[BB] = std::make_pair(HI0, HI0Node);
    } else {
      HighestReachedDFSNum[BB] = std::make_pair(HI1, HI1Node);
    }

    // 4) This is the highest reached node (`HI2Node`) for any child of the
    //    current `Block`, excluding the one which reaches the highest target
    //    (that we know is reachable from `HI1Child`)
    size_t HI2 = SizeTMaxValue;
    BlockNode *HI2Node = nullptr;
    for (const auto &[ForwardEdge, _, __] : getForwardEdges(BB)) {
      BlockNode *Target = std::get<1>(ForwardEdge);
      if (Target != HI1Child) {
        size_t NewMin = std::min(HI2, HighestReachedDFSNum.at(Target).first);
        if (NewMin < HI2) {
          HI2Node = HighestReachedDFSNum.at(Target).second;
        }
        HI2 = NewMin;
      }
    }

    // Insert the capping backedge, if the second highest reached node for any
    // child of the current `BB`, is higher than any backedges which starts from
    // `BB`. The capping backedge goes from `BB` to such second highest reached
    // node.
    if (HI2 < HI0) {

      // In some particular cases, the capping backedge would be inserted
      // between `BB` and itself. However, since no tree edge can exists between
      // a node and himself, it is useless to add such capping backedge
      if (BB != HI2Node) {

        // We need to compute the number of current successors of `BB`, so we
        // can correctly assign the `SuccNum` for the fake edge we are
        // introducing.
        size_t CurrentSuccs = llvm::size(llvm::children_edges<BlockNode *>(BB));
        BB->addSuccessor(HI2Node,
                         { OriginalEdgeKind::Fake,
                           SpanningTreeEdgeKind::BackEdge,
                           getIncrementalIndex(),
                           CurrentSuccs });
      }
    }
  }
}

/// Helper function that returns only the `forward` edges, according to
/// the spanning tree. Forward edges are defined as edges connecting the current
/// `Source` node to nodes that are successor of it on the spanning tree.
template<class GraphT, class GT>
CycleEquivalenceAnalysis<GraphT, GT>::EdgesVectorType
CycleEquivalenceAnalysis<GraphT, GT>::getForwardEdges(BlockNode *Source) {
  EdgesVectorType ForwardEdges;

  // We save the size of the successors when the graph is not treated as
  // undirected (we use this info to understand when the concatenated iterator
  // has surpassed the end of the non-concat one, so we can deduce if we are
  // seeing the inverted version of the edge wrt. the directed version of the
  // graph)
  size_t EndOfSuccs = llvm::size(llvm::children_edges<BlockNode *>(Source));
  size_t Index = 0;

  for (auto [Neighbor, Label] :
       llvm::children_edges<llvm::Undirected<BlockNode *>>(Source)) {
    auto Edge = BlockEdgeDescriptor({ Source, Neighbor, Label->id() });
    if (Label->type() == SpanningTreeEdgeKind::TreeEdge
        and not isBackedge(Edge)) {

      // Compute the `IsInverted` information checking if our iterator superated
      // the end of the successors (and ended up in the predecessor half
      // computed by the concatenation)
      bool IsInverted = Index >= EndOfSuccs;
      ForwardEdges.push_back({ Edge, Label, IsInverted });
    }

    Index++;
  }
  return ForwardEdges;
}

/// Helper function that returns only the `backward` edges, according to the
/// spanning tree. Backward edges are defined as edges connecting the current
/// `Source` node to nodes that are not successors of it on the spanning tree,
/// and therefore are backedges.
template<class GraphT, class GT>
CycleEquivalenceAnalysis<GraphT, GT>::EdgesVectorType
CycleEquivalenceAnalysis<GraphT, GT>::getBackwardEdges(BlockNode *Source) {
  EdgesVectorType BackwardEdges;

  // We save the size of the successors when the graph is not treated as
  // undirected (we use this info to understand when the concatenated iterator
  // has surpassed the end of the non-concat one, so we can deduce if we are
  // seeing the inverted version of the edge wrt. the directed version of the
  // graph)
  size_t EndOfSuccs = llvm::size(llvm::children_edges<BlockNode *>(Source));
  size_t Index = 0;

  for (auto [Neighbor, Label] :
       llvm::children_edges<llvm::Undirected<BlockNode *>>(Source)) {
    auto Edge = BlockEdgeDescriptor({ Source, Neighbor, Label->id() });
    if (Label->type() == SpanningTreeEdgeKind::BackEdge and isBackedge(Edge)) {

      // Compute the `IsInverted` information checking if our iterator surpassed
      // the end of the successors (and ended up in the predecessor half
      // computed by the concatenation)
      bool IsInverted = Index >= EndOfSuccs;

      // Avoid double insertion of self-loops as a backedge. This happens when
      // the target of the edge is equal to the source, and we are in the second
      // half of the concat iterator. If a self-loop is present, we will end up
      // enqueuing it two times, since the `isBackedge` helper function only
      // uses the `DFSNum` as a criterion, and therefore cannot distinguish,
      // when we have `A <-> A`, when we are traversing it as `A -> A` and later
      // on as `A <- A`.
      if (std::get<1>(Edge) != Source or not IsInverted) {
        BackwardEdges.push_back({ Edge, Label, IsInverted });
      }
    }

    Index++;
  }
  return BackwardEdges;
}

/// Helper function that returns only the `tree` edges, according to the
/// spanning tree. Tree edges are defined as edges connecting the current
/// `Source` node to nodes that are predecessor of it on the spanning tree.
template<class GraphT, class GT>
CycleEquivalenceAnalysis<GraphT, GT>::EdgesVectorType
CycleEquivalenceAnalysis<GraphT, GT>::getTreeEdges(BlockNode *Source) {
  EdgesVectorType TreeEdges;

  // We save the size of the successors when the graph is not treated as
  // undirected (we use this info to understand when the concatenated iterator
  // has surpassed the end of the non-concat one, so we can deduce if we are
  // seeing the inverted version of the edge wrt. the directed version of the
  // graph)
  size_t EndOfSuccs = llvm::size(llvm::children_edges<BlockNode *>(Source));
  size_t Index = 0;

  for (auto [Neighbor, Label] :
       llvm::children_edges<llvm::Undirected<BlockNode *>>(Source)) {
    auto Edge = BlockEdgeDescriptor({ Neighbor, Source, Label->id() });
    if (Label->type() == SpanningTreeEdgeKind::TreeEdge
        and not isBackedge(Edge)) {

      // Mind that the tree edges identified in this routine, start from
      // `Neighbor` and target `Source`, therefore we need to flip the
      // `IsInverted` criterion
      bool IsInverted = Index < EndOfSuccs;
      TreeEdges.push_back({ Edge, Label, IsInverted });
    }

    Index++;
  }
  return TreeEdges;
}

// Helper function that creates the `GenericGraph` representing the undirected
// version of the original CFG
template<class GraphT, class GT>
CycleEquivalenceAnalysis<GraphT, GT>::BlockGraph
CycleEquivalenceAnalysis<GraphT, GT>::initializeGenericGraph(const GraphT F) {
  BlockGraph Graph;
  std::map<NodeT, BlockNode *> BBToNode;

  // For every BasicBlock in F, we create a corresponding node in the
  // `GenericGraph`, and we keep a correspondence between the `BasicBlock` and
  // the node in the `GenericGraph`
  auto NodesRange = llvm::nodes(F);
  for (NodeT BB : NodesRange) {
    BBToNode[BB] = Graph.addNode(BB);
  }

  // Set the entry node
  BlockNode *EntryNode = BBToNode.at(GT::getEntryNode(F));
  Graph.setEntryNode(EntryNode);

  // We keep in a set all the nodes that do not have a successor. By
  // construction, we require such node to be unique (in order to perform the
  // insertion of the artificial `exit` -> `entry` backedge).
  llvm::SmallVector<BlockNode *, 1> Exits;

  // For every BasicBlock in F, we connect its successors in the `GenericGraph`
  for (NodeT BB : NodesRange) {
    BlockNode *From = BBToNode.at(BB);

    // Iterate over the successors
    auto SuccessorsRange = llvm::make_range(GT::child_begin(BB),
                                            GT::child_end(BB));
    for (auto &Group : llvm::enumerate(SuccessorsRange)) {
      NodeT SuccessorBB = Group.value();

      // Insert the edge from `BB` to `SuccessorBB` and at each edge insertion,
      // we add an incremental distinct label
      BlockNode *SuccessorNode = BBToNode.at(SuccessorBB);
      From->addSuccessor(SuccessorNode,
                         { OriginalEdgeKind::Real,
                           SpanningTreeEdgeKind::Invalid,
                           getIncrementalIndex(),
                           Group.index() });
    }

    // If the BasicBlock has no successors, we save the current node in the
    // `Exits` set for later insertion of the `exit` -> `entry` artificial
    // backedge
    if (SuccessorsRange.empty()) {
      Exits.push_back(From);
    }
  }

  BlockNode *Sink = nullptr;
  switch (Exits.size()) {
  case 0: {

    // If we do not have any exit candidate, we elect as the sink, the node
    // with the highest DFS number.
    // We cannot anticipate the DFSNum computation performed in
    // `computeDFSAndSpanningTree` here, because we use such computation also to
    // compute the `TreeEdge`s and the `BackEdge`s in the graph. The reasoning
    // is: if we anticipate such computation, it means that we need to assign
    // all the edges that we introduce after this stage as either `TreeEdge`s or
    // `BackEdge`s. We can get away with assigning the exit -> entry additional
    // edge as a `BackEdge` (even though it may exists a DFS visit treating it
    // as a `TreeEdge`), but we cannot assign the additional edges connecting
    // the exit candidates to the Sink node, because one of such edges may be
    // assigned as a `TreeEdge` on the undirected graph.
    BlockNode *HighestDFSNumNode = nullptr;
    for (auto *DFS : llvm::depth_first(Graph.getEntryNode())) {
      HighestDFSNumNode = DFS;
    }
    revng_assert(HighestDFSNumNode);
    Sink = HighestDFSNumNode;
  } break;
  case 1: {

    // No preprocessing is required, the `Exit` is the only exit candidate
    Sink = *Exits.begin();
  } break;
  default: {

    // We need to insert a new sink node which connects all the exit nodes. This
    // node is a bit of an exception wrt. to all the others, since it does not
    // have a correspondent one in the original graph. Therefore, the argument
    // passed to its construct must be `nullptr`.
    Sink = Graph.addNode(nullptr);

    // Connect all the exit candidates to the sink
    for (auto *ExitCandidate : Exits) {

      // By definition, each exit candidate does not have any successor, so the
      // `SuccNum` will be 0
      ExitCandidate->addSuccessor(Sink,
                                  { OriginalEdgeKind::Fake,
                                    SpanningTreeEdgeKind::Invalid,
                                    getIncrementalIndex(),
                                    0 });
    }
  } break;
  }

  // We insert the exit -> start edge. By design, the exit edge does not have
  // any outgoing edges, therefore we can use 0 as the index for the successor
  // number
  revng_assert(Sink != nullptr);

  // We need to compute the number of current successors of `BB`, so we
  // can correctly assign the `SuccNum` for the fake edge we are
  // introducing (in particular, when we elect the node in a original graph with
  // no exit candidates, where for sure the `Sink` node will already have at
  // least one successor).
  size_t CurrentSuccs = llvm::size(llvm::children_edges<BlockNode *>(Sink));
  Sink->addSuccessor(EntryNode,
                     { OriginalEdgeKind::Fake,
                       SpanningTreeEdgeKind::BackEdge,
                       getIncrementalIndex(),
                       CurrentSuccs });

  return Graph;
}

template<class GraphT, class GT>
CycleEquivalenceAnalysis<GraphT, GT>::CycleEquivalenceResult
CycleEquivalenceAnalysis<GraphT, GT>::getEdgeToCycleEquivalenceClassIDMap() {
  return EdgeToCycleEquivalenceClassIDMap;
}

/// `run` method of the `CycleEquivalenceAnalysis` class
template<class GraphT, class GT>
void CycleEquivalenceAnalysis<GraphT, GT>::run(GraphT F) {

  // Here we perform the initialization of the `GenericGraph`, and then
  // call the analysis implementation
  BlockGraph Graph = initializeGenericGraph(F);

  computeCycleEquivalence(F, Graph);

  // Print the resulting cycle equivalence classes when the analysis logger is
  // enabled
  revng_log(CycleEquivalenceAnalysisLogger, print());
}

/// Helper function which executes the Brackets Cycle Set Analysis
template<class GraphT, class GT>
void CycleEquivalenceAnalysis<GraphT, GT>::computeCycleEquivalence(GraphT F,
                                                                   BlockGraph
                                                                     &Graph) {

  // 1) Compute the DFSNum and the spanning tree over the undirected
  //    `GenericGraph` using a standard DFS
  computeDFSAndSpanningTree(Graph);

  // 2) Insert the capping backedges. We actually materialize capping backedges
  //    on the graph, instead of keeping multiple parallel data structures, as
  //    it is done on the paper algorithm.
  insertCappingBackedges(Graph);

  // Dump the internal `GenericGraph` for debug
  if (CycleEquivalenceAnalysisLogger.isEnabled()) {
    revng_log(CycleEquivalenceAnalysisLogger,
              "Dumping undirected graph for function: " << F->getName());
    // We need the `llvm::DOTGraphTraits` trait implemented on the template
    // class parameter of `CycleEquivalenceAnalysis` for the `llvm::WriteGraph`
    // primitive to work
    llvm::WriteGraph(&Graph, "BracketGraph.dot");
  }

  // `OpenBrackets` state for each `Block`
  std::map<BlockNode *, llvm::SetVector<BlockEdgeDescriptor>> OpenBracketsMap;

  // 2) Perform the bracket set cycle computation using a `llvm::po_iterator`
  for (BlockNode *BB :
       llvm::post_order(llvm::Undirected<BlockNode *>(Graph.getEntryNode()))) {

    auto &OpenBrackets = OpenBracketsMap[BB];

    // We should not have any bracket open at the beginning of the analysis
    revng_assert(OpenBrackets.empty());

    // Prepopulate the `OpenBrackets` `SetVector` with the brackets of all the
    // children of the current node
    for (const auto &[ForwardEdge, _, __] : getForwardEdges(BB)) {
      BlockNode *Successor = std::get<1>(ForwardEdge);
      auto &ChildBrackets = OpenBracketsMap.at(Successor);
      OpenBrackets.insert(ChildBrackets.begin(), ChildBrackets.end());
    }

    // We open all the brackets starting from the current node, iterating over
    // all the edges outgoing from the current `Block`
    for (const auto &[BackwardEdge, _, __] : getBackwardEdges(BB)) {
      OpenBrackets.insert(BackwardEdge);
    }

    // We then remove all the standard open brackets ending in the current node
    OpenBrackets.remove_if([&BB](const auto &Edge) {
      return std::get<1>(Edge) == BB;
    });

    // Every bracket (backedge on the undirected graph), gives origin to a new
    // cycle equivalence class composed by itself only (to start). In this way,
    // we can apply Theorem 4, which states that any tree edge `t` and backedge
    // `b` are cycle equivalent only if `b` is the only bracket for `t`.
    // Eventual tree edges that will be part of the cycle equivalence class
    // denoted by `(BackwardEdge, 1)` (i.e., the class where `BackwardEdge` is
    // the only bracket), will then be added to such class.
    for (const auto &[BackwardEdge, Label, IsInverted] : getBackwardEdges(BB)) {
      BracketDescriptor BD = std::make_pair(BackwardEdge, 1);
      insertEdge(BackwardEdge, Label, IsInverted, BD);
    }

    // We mark the edges to the ancestors with the result of the analysis
    for (const auto &[TreeEdge, Label, IsInverted] : getTreeEdges(BB)) {
      BlockEdgeDescriptor LastOpenBracket = OpenBrackets.back();
      BracketDescriptor BD = std::make_pair(LastOpenBracket,
                                            OpenBrackets.size());
      insertEdge(TreeEdge, Label, IsInverted, BD);
    }
  }
}

// Explicit template instantiation for the `llvm::Function *` parameter
template class CycleEquivalenceAnalysis<llvm::Function *>;

// CycleEquivalenceAnalysis<llvm::Function *>::BlockGraph DOTGraphTraits

using BlockNode = CycleEquivalenceAnalysis<llvm::Function *>::BlockNode;
using BlockGraph = CycleEquivalenceAnalysis<llvm::Function *>::BlockGraph;

static std::string getNodeLabel(const BlockNode *N) {
  return N->getName().str() + ", " + std::to_string(N->getDFSNum());
}

std::string
llvm::DOTGraphTraits<BlockGraph *>::getNodeLabel(const BlockNode *N,
                                                 const BlockGraph *G) {
  return ::getNodeLabel(N);
}

std::string
llvm::DOTGraphTraits<BlockGraph *>::getEdgeAttributes(const BlockNode *N,
                                                      EdgeIterator EI,
                                                      const BlockGraph *G) {
  std::string EdgeLabel = EI.getCurrent()->Label->edgeLabel();
  std::string EdgeStyle = EI.getCurrent()->Label->edgeStyle();

  std::string EdgeAttributes = "label=\"" + EdgeLabel + "\",style=" + EdgeStyle
                               + ",dir=none";
  return EdgeAttributes;
}

// CycleEquivalenceAnalysis<llvm::Function *>::BlockGraph DOTGraphTraits end

template<class GraphT>
CycleEquivalenceAnalysis<GraphT>::CycleEquivalenceResult
getEdgeToCycleEquivalenceClassIDMap(GraphT F) {
  CycleEquivalenceAnalysis<GraphT> CEA;
  CEA.run(F);

  return CEA.getEdgeToCycleEquivalenceClassIDMap();
}

template CycleEquivalenceAnalysis<Function *>::CycleEquivalenceResult
getEdgeToCycleEquivalenceClassIDMap(Function *F);
