#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iterator>

#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/GenericDomTreeConstruction.h"
#include "llvm/Support/raw_os_ostream.h"

#include "revng/ADT/ReversePostOrderTraversal.h"
#include "revng/BasicAnalyses/GeneratedCodeBasicInfo.h"
#include "revng/Support/Debug.h"
#include "revng/Support/IRHelpers.h"

#include "revng-c/RestructureCFG/ASTNodeUtils.h"
#include "revng-c/RestructureCFG/ASTTree.h"
#include "revng-c/RestructureCFG/BasicBlockNodeBB.h"
#include "revng-c/RestructureCFG/MetaRegionBB.h"
#include "revng-c/RestructureCFG/RegionCFGTree.h"
#include "revng-c/RestructureCFG/Utils.h"

#include "BasicBlockNode.h"

class GHASTDumper {
  size_t GraphLogCounter;
  Logger<> &Logger;
  std::string FunctionName;
  const ASTTree &AST;
  std::string FolderName;

public:
  GHASTDumper(::Logger<> &Logger,
              const llvm::Function &F,
              const ASTTree &TheAST,
              const std::string &FolderName) :
    GraphLogCounter(0),
    Logger(Logger),
    FunctionName(F.getName().str()),
    AST(TheAST),
    FolderName(FolderName) {}

  void log(const std::string &Filename) {
    if (Logger.isEnabled()) {
      AST.dumpASTOnFile(FunctionName,
                        "ast-" + FolderName,
                        std::to_string(GraphLogCounter++) + "-" + Filename);
    }
  }
};

// Helper function that visit an AST tree and creates the sequence nodes
inline ASTNode *createSequence(ASTTree &Tree, ASTNode *RootNode) {
  SequenceNode *RootSequenceNode = Tree.addSequenceNode();
  RootSequenceNode->addNode(RootNode);

  for (ASTNode *Node : RootSequenceNode->nodes()) {

    switch (Node->getKind()) {

    case ASTNode::NK_If: {
      auto *If = llvm::cast<IfNode>(Node);

      if (If->hasThen())
        If->setThen(createSequence(Tree, If->getThen()));
      if (If->hasElse())
        If->setElse(createSequence(Tree, If->getElse()));
    } break;

    case ASTNode::NK_Switch: {
      auto *Switch = llvm::cast<SwitchNode>(Node);
      for (auto &LabelCasePair : Switch->cases())
        LabelCasePair.second = createSequence(Tree, LabelCasePair.second);
    } break;

    case ASTNode::NK_Scs: {
      auto *Scs = llvm::cast<ScsNode>(Node);
      if (Scs->hasBody())
        Scs->setBody(createSequence(Tree, Scs->getBody()));
    } break;

    case ASTNode::NK_Code: {
      // TODO: confirm that doesn't make sense to process a code node.
    } break;

    case ASTNode::NK_Continue:
    case ASTNode::NK_Break:
    case ASTNode::NK_SwitchBreak:
    case ASTNode::NK_Set: {
      // Do nothing for these nodes
    } break;

    case ASTNode::NK_List:
    default:
      revng_abort("AST node type not expected");
    }
  }

  return RootSequenceNode;
}

// Helper function that simplifies useless dummy nodes
inline void simplifyDummies(ASTTree &AST, ASTNode *RootNode) {

  switch (RootNode->getKind()) {

  case ASTNode::NK_List: {
    auto *Sequence = llvm::cast<SequenceNode>(RootNode);
    std::vector<ASTNode *> UselessDummies;
    for (ASTNode *Node : Sequence->nodes()) {
      if (Node->isDummy()) {
        UselessDummies.push_back(Node);
      } else {
        simplifyDummies(AST, Node);
      }
    }
    for (ASTNode *Node : UselessDummies) {
      Sequence->removeNode(Node);
      AST.removeASTNode(Node);
    }
  } break;

  case ASTNode::NK_If: {
    auto *If = llvm::cast<IfNode>(RootNode);
    if (If->hasThen()) {
      simplifyDummies(AST, If->getThen());
    }
    if (If->hasElse()) {
      simplifyDummies(AST, If->getElse());
    }
  } break;

  case ASTNode::NK_Switch: {

    auto *Switch = llvm::cast<SwitchNode>(RootNode);

    for (auto &LabelCaseNodePair : Switch->cases())
      simplifyDummies(AST, LabelCaseNodePair.second);

  } break;

  case ASTNode::NK_Scs: {
    auto *Scs = llvm::cast<ScsNode>(RootNode);
    if (Scs->hasBody())
      simplifyDummies(AST, Scs->getBody());
  } break;

  case ASTNode::NK_Code:
  case ASTNode::NK_Continue:
  case ASTNode::NK_Break:
  case ASTNode::NK_SwitchBreak:
  case ASTNode::NK_Set:
    // Do nothing
    break;

  default:
    revng_unreachable();
  }
}

template<class NodeT>
inline BasicBlockNode<NodeT> *getDirectSuccessor(BasicBlockNode<NodeT> *Node) {
  BasicBlockNode<NodeT> *Successor = nullptr;
  if (Node->successor_size() == 1) {
    Successor = Node->getSuccessorI(0);
  }
  return Successor;
}

template<class NodeT>
inline BasicBlockNode<NodeT> *
getUniqueSuccessorOrNull(BasicBlockNode<NodeT> *Node) {
  BasicBlockNode<NodeT> *Successor = nullptr;
  revng_assert(Node->successor_size() <= 1);
  if (Node->successor_size() == 1) {
    Successor = Node->getSuccessorI(0);
  }
  return Successor;
}

template<class NodeT>
inline std::vector<BasicBlockNode<NodeT> *>
getNotInlinedSuccs(BasicBlockNode<NodeT> *Node) {
  using BasicBlockNodeT = BasicBlockNode<NodeT>;
  using ConstEdge = std::pair<const BasicBlockNodeT *, const BasicBlockNodeT *>;

  std::vector<BasicBlockNodeT *> NotInlinedSuccessors;
  for (BasicBlockNodeT *Successor : Node->successors()) {
    if (not isEdgeInlined(ConstEdge{ Node, Successor })) {
      NotInlinedSuccessors.push_back(Successor);
    }
  }
  return NotInlinedSuccessors;
}

template<class NodeT>
inline std::vector<BasicBlockNode<NodeT> *>
getNotInlinedNotTerminalSuccs(BasicBlockNode<NodeT> *Node) {
  using BasicBlockNodeT = BasicBlockNode<NodeT>;
  using ConstEdge = std::pair<const BasicBlockNodeT *, const BasicBlockNodeT *>;

  std::vector<BasicBlockNodeT *> NotInlinedNotTerminalSuccessors;
  for (BasicBlockNodeT *Successor : Node->successors()) {
    if (not isEdgeInlined(ConstEdge{ Node, Successor })
        and getUniqueSuccessorOrNull(Successor)) {
      NotInlinedNotTerminalSuccessors.push_back(Successor);
    }
  }
  return NotInlinedNotTerminalSuccessors;
}

template<class NodeT>
inline BasicBlockNode<NodeT> *
findCommonPostDom(BasicBlockNode<NodeT> *Succ1, BasicBlockNode<NodeT> *Succ2) {

  // Retrieve the successor of the two successors of the `IfNode`, and check
  // that or the retrieved node is equal for both the successors, or it does
  // not exists for both of them.
  BasicBlockNode<NodeT> *SuccOfSucc1 = nullptr;
  BasicBlockNode<NodeT> *SuccOfSucc2 = nullptr;

  revng_assert(Succ1->successor_size() < 2);
  revng_assert(Succ2->successor_size() < 2);

  if (Succ1->successor_size() == 1)
    SuccOfSucc1 = getDirectSuccessor(Succ1);

  if (Succ2->successor_size() == 1)
    SuccOfSucc2 = getDirectSuccessor(Succ2);

  revng_assert(SuccOfSucc1 == SuccOfSucc2);

  return SuccOfSucc1;
}

template<class NodeT>
inline ASTNode *findASTNode(ASTTree &AST,
                            typename RegionCFG<NodeT>::BBNodeMap &TileToNodeMap,
                            BasicBlockNode<NodeT> *Node) {
  if (auto It = TileToNodeMap.find(Node); It != TileToNodeMap.end())
    Node = It->second;

  return AST.findASTNode(Node);
}

template<class NodeT>
inline void
connectTile(llvm::DominatorTreeBase<BasicBlockNode<NodeT>, false> &ASTDT,
            BasicBlockNode<NodeT> *Node,
            BasicBlockNode<NodeT> *Tile) {
  using BBNodeT = BasicBlockNode<NodeT>;
  using BasicBlockNodeTVect = std::vector<BBNodeT *>;
  using EdgeDescriptor = typename BBNodeT::EdgeDescriptor;

  // Move all the edges incoming in the head of the collapsed region to the tile
  // node.
  BasicBlockNodeTVect Predecessors;
  for (BBNodeT *Predecessor : Node->predecessors())
    Predecessors.push_back(Predecessor);

  for (BBNodeT *Predecessor : Predecessors) {
    moveEdgeTarget(EdgeDescriptor{ Predecessor, Node }, Tile);

    // Update the dominator tree used for AST building.
    using DomUpdate = typename llvm::DominatorTreeBase<BasicBlockNode<NodeT>,
                                                       false>::UpdateType;
    const auto Insert = llvm::DominatorTree::Insert;
    const auto Delete = llvm::DominatorTree::Delete;
    std::vector<DomUpdate> Updates;
    Updates.push_back({ Delete, Predecessor, Node });
    Updates.push_back({ Insert, Predecessor, Tile });
    ASTDT.applyUpdates(Updates);
  }
}

template<class NodeT>
inline BasicBlockNode<NodeT> *
createTileImpl(RegionCFG<NodeT> &Graph,
               llvm::DominatorTreeBase<BasicBlockNode<NodeT>, false> &ASTDT,
               typename RegionCFG<NodeT>::BBNodeMap &TileToNodeMap,
               BasicBlockNode<NodeT> *Node,
               BasicBlockNode<NodeT> *End,
               bool EndIsPartOfTile) {
  using BBNodeT = BasicBlockNode<NodeT>;
  using BasicBlockNodeTVect = std::vector<BBNodeT *>;
  using EdgeDescriptor = typename BBNodeT::EdgeDescriptor;

  // Create the new tile node
  BBNodeT *Tile = Graph.addTile();

  // Connect the incoming edge to the newly created tile
  connectTile(ASTDT, Node, Tile);

  // The `End` node is considered as part of the tile that is being collapsed
  if (EndIsPartOfTile) {
    // If `End` is marked as part of the tile, it must be present
    revng_assert(End);

    // We move all the edges exiting from the `End` node of the collapsed region
    // to the tile node
    BasicBlockNodeTVect Successors;
    for (BBNodeT *Successor : End->successors())
      Successors.push_back(Successor);

    for (BBNodeT *Successor : Successors) {
      auto Edge = extractLabeledEdge(EdgeDescriptor{ End, Successor });
      ASTDT.deleteEdge(End, Successor);
      addEdge(EdgeDescriptor{ Tile, Successor }, Edge.second);
      ASTDT.insertEdge(Tile, Successor);
    }
  }

  // Update the map containing the mapping between tiles and the head node which
  // gave origin to a certain tile.
  TileToNodeMap[Tile] = Node;

  return Tile;
}

template<class NodeT>
inline BasicBlockNode<NodeT> *
createTile(RegionCFG<NodeT> &Graph,
           llvm::DominatorTreeBase<BasicBlockNode<NodeT>, false> &ASTDT,
           typename RegionCFG<NodeT>::BBNodeMap &TileToNodeMap,
           BasicBlockNode<NodeT> *Node,
           BasicBlockNode<NodeT> *End,
           bool EndIsPartOfTile) {

  using BBNodeT = BasicBlockNode<NodeT>;
  using EdgeDescriptor = typename BBNodeT::EdgeDescriptor;

  // Call the `Impl` function that is responsible of creating the tile node
  BBNodeT *Tile = createTileImpl(Graph,
                                 ASTDT,
                                 TileToNodeMap,
                                 Node,
                                 End,
                                 EndIsPartOfTile);

  // When `End` is present and not part of the collapsed tile, we need to
  // connect the `Tile` node to it, in order to preserve the original edges in
  // the control-flow
  if (not EndIsPartOfTile and End != nullptr) {
    moveEdgeSource(EdgeDescriptor(Node, End), Tile);
    ASTDT.deleteEdge(Node, End);
    ASTDT.insertEdge(Tile, End);
  }

  return Tile;
}

template<class NodeT>
inline BasicBlockNode<NodeT> *
createSwitchTile(RegionCFG<NodeT> &Graph,
                 llvm::DominatorTreeBase<BasicBlockNode<NodeT>, false> &ASTDT,
                 typename RegionCFG<NodeT>::BBNodeMap &TileToNodeMap,
                 BasicBlockNode<NodeT> *Node,
                 BasicBlockNode<NodeT> *End,
                 bool EndIsPartOfTile,
                 std::vector<BasicBlockNode<NodeT> *>
                   &NotInlinedNotTerminalSuccessors) {

  using BBNodeT = BasicBlockNode<NodeT>;
  using BasicBlockNodeT = BasicBlockNode<NodeT>;
  using EdgeDescriptor = typename BBNodeT::EdgeDescriptor;
  using EdgeInfo = typename BasicBlockNodeT::EdgeInfo;

  // Call the `Impl` function that is responsible of creating the tile node
  BBNodeT *Tile = createTileImpl(Graph,
                                 ASTDT,
                                 TileToNodeMap,
                                 Node,
                                 End,
                                 EndIsPartOfTile);

  // When `End` is present and not part of the collapsed tile, we need to
  // connect the `Tile` node to it, in order to preserve the original edges in
  // the control-flow
  if (not EndIsPartOfTile and End != nullptr) {
    // Step 1:
    // There is a common successor (which is the `End` node) for all the cases
    // of the switch, excluding the inlined cases and the case nodes without a
    // successor (see comments on the declaration of the
    // `NotInlinedNotTerminalSuccessors` variable for a motivation for this). We
    // need to remove all the edges exiting from the cases, and replace them
    // with an edge connect the `Tile` node to `End`.
    std::optional<EdgeInfo> EdgeInfoN;
    for (BasicBlockNodeT *Successor : NotInlinedNotTerminalSuccessors) {

      // We do not process `Successor` when it is equal to `End`, it will be
      // treated separately
      if (Successor == End) {
        continue;
      }

      BasicBlockNodeT *SuccOfSuccN = getUniqueSuccessorOrNull(Successor);
      revng_assert(SuccOfSuccN == End);
      auto Edge = extractLabeledEdge(EdgeDescriptor{ Successor, End });

      // Check that all the `EdgeInfo`s attached to the edges we are
      // going to remove, are equal.
      if (not EdgeInfoN.has_value()) {
        EdgeInfoN.emplace(Edge.second);
      } else {
        revng_assert(EdgeInfoN.value() == Edge.second);
      }

      ASTDT.deleteEdge(Successor, End);
    }

    // Connect the tile to the elected successor, if we found candidate nodes
    // eligible above
    if (EdgeInfoN.has_value()) {
      addEdge(EdgeDescriptor{ Tile, End }, EdgeInfoN.value());
      ASTDT.insertEdge(Tile, End);
    }

    // Step 2:
    // It may be that `End` was itself a successor of `Node`, and in Step 1 we
    // did not connect `Tile` with `End`. In such situation, we need to do that
    // now.
    if (Node->hasSuccessor(End) and not(Tile->hasSuccessor(End))) {
      moveEdgeSource(EdgeDescriptor(Node, End), Tile);
      ASTDT.deleteEdge(Node, End);
      ASTDT.insertEdge(Tile, End);
    }
  }

  return Tile;
}

inline void
generateAst(RegionCFG<llvm::BasicBlock *> &Region,
            ASTTree &AST,
            std::map<RegionCFG<llvm::BasicBlock *> *, ASTTree> &CollapsedMap) {
  // Define some using used in all the function body.
  using NodeT = llvm::BasicBlock *;
  using BasicBlockNodeT = typename RegionCFG<NodeT>::BasicBlockNodeT;

  // Get some fields of `RegionCFG`.
  std::string RegionName = Region.getRegionName();
  std::string FunctionName = Region.getFunctionName();

  Region.markUnreachableAsInlined();

  // Invoke the weave function.
  Region.weave();

  // Invoke the inflate function.
  Region.inflate();

  // After we are done with the combing, we need to pre-compute the weight of
  // the current RegionCFG, so that during the untangle phase of other
  // `RegionCFG` that contain a collapsed node pointing to the current
  // `RegionCFG`. the weight of collapsed node is ready to consume. Indeed,
  // after the tiling phase, the `RegionCFG` is destroyed, so the last place
  // where we can compute it is here.
  Region.computeUntangleWeight();

  // TODO: factorize out the AST generation phase.
  llvm::DominatorTreeBase<BasicBlockNode<NodeT>, false> ASTDT;
  ASTDT.recalculate(Region);

  CombLogger << DoLog;

  std::map<BasicBlockNode<NodeT> *, BasicBlockNode<NodeT> *> TileToNodeMap;

  using BasicBlockNodeTVect = typename RegionCFG<NodeT>::BasicBlockNodeTVect;
  BasicBlockNodeTVect PONodes;
  for (auto *N : post_order(&Region))
    PONodes.push_back(N);

  CFGDumper Dumper(Region, FunctionName, RegionName, "tile");

  for (BasicBlockNode<NodeT> *Node : PONodes) {
    Dumper.log("-node-" + Node->getNameStr());

    // Collect the children nodes in the dominator tree.
    llvm::SmallVector<decltype(Node), 8> Children;
    {
      const auto &DomTreeChildren = ASTDT[Node]->children();
      for (auto *DomTreeNode : DomTreeChildren)
        Children.push_back(DomTreeNode->getBlock());
    }

    // Collect the successor nodes of the current analyzed node.
    llvm::SmallVector<decltype(Node), 8> Successors;
    for (BasicBlockNode<NodeT> *Successor : Node->successors())
      Successors.push_back(Successor);

    // Handle collapsded node.
    ASTTree::ast_unique_ptr ASTObject;
    if (Node->isCollapsed()) {

      revng_assert(Children.size() <= 1);
      RegionCFG<NodeT> *BodyGraph = Node->getCollapsedCFG();
      revng_assert(BodyGraph != nullptr);
      revng_log(CombLogger,
                "Inspecting collapsed node: " << Node->getNameStr());

      // Call recursively the generation of the AST for the collapsed node.
      const auto &[It, New] = CollapsedMap.insert({ BodyGraph, ASTTree() });
      ASTTree &CollapsedAST = It->second;
      if (New)
        generateAst(*BodyGraph, CollapsedAST, CollapsedMap);

      ASTNode *Body = AST.copyASTNodesFrom(CollapsedAST);

      switch (Successors.size()) {

      case 0: {
        ASTObject.reset(new ScsNode(Node, Body));
      } break;

      case 1: {
        auto *Succ = Successors[0];
        ASTNode *ASTChild = nullptr;
        if (ASTDT.dominates(Node, Succ)) {
          ASTChild = findASTNode(AST, TileToNodeMap, Succ);
          createTile(Region, ASTDT, TileToNodeMap, Node, Succ, true);
        }
        ASTObject.reset(new ScsNode(Node, Body, ASTChild));
      } break;

      default:
        revng_abort();
      }

    } else if (Node->isDispatcher() or isASwitch(Node)) {

      revng_assert(Node->isCode() or Node->isDispatcher());

      llvm::Value *SwitchCondition = nullptr;
      if (not Node->isDispatcher()) {
        NodeT OriginalNode = Node->getOriginalNode();
        llvm::Instruction *Terminator = OriginalNode->getTerminator();
        llvm::SwitchInst *Switch = llvm::cast<llvm::SwitchInst>(Terminator);
        SwitchCondition = Switch->getCondition();
      }
      revng_assert(SwitchCondition or Node->isDispatcher());

      // Results variables
      bool EndIsPartOfTile = false;
      BasicBlockNodeT *PostDomBB = nullptr;
      BasicBlockNodeT *CandidateFallthroughBB = nullptr;

      // TODO: this helper variable is needed to represent the node that is the
      //       target of the inserted `SwitchBreak` nodes. This is needed until
      //       we eliminate the special casing for the weaved switches below.
      BasicBlockNodeT *SwitchBreakTarget = nullptr;

      BasicBlockNodeTVect NotInlinedSuccessors = getNotInlinedSuccs(Node);

      // In the following, we will also need to exclude successors of the switch
      // that do not have successors, but are not inlined either. An example
      // motivating this: Imagine node C that has two successors, that are both
      // inlined. That node C is the successor of node B. So at some point we
      // create tile A, which "inside" has B followed by C followed by its
      // inlined successors. This node A is indeed an example of a case node
      // without a successor which is not inlined, but that we still do not need
      // to consider when performing the search for the common fallthrough node.
      BasicBlockNodeTVect
        NotInlinedNotTerminalSuccessors = getNotInlinedNotTerminalSuccs(Node);

      // Count the non inlined successors
      unsigned NotInlined = NotInlinedSuccessors.size();

      // We special case the handling of the "all but one case inlined", due to
      // some assumptions we make about the nesting of the weaved `switch`es.
      // Specifically, we assume that each weaved switch Y, relative to switch
      // X, must be nested "inside" it, also in the resulting generated AST.
      // Therefore we cannot match this as we would do in the ordinary way, by
      // placing the weaved switch Y as fallthrough of switch X.
      // TODO: this special casing for the weaved switches can be dropped if we
      //       give up the assumption that each weaved switch is nested into
      //       its parent switch. This would lead to greatly reduce the
      //       complexity of switch tiling logic.
      if (NotInlined == 1) {

        // Handle the special case with one a single non inlined successor
        const auto NotInlined = [&Node](const auto *Child) {
          using ConstEdge = std::pair<const BasicBlockNodeT *,
                                      const BasicBlockNodeT *>;
          return not isEdgeInlined(ConstEdge{ Node, Child });
        };

        auto It = std::find_if(Successors.begin(),
                               Successors.end(),
                               NotInlined);

        // Assert that we found one
        revng_assert(It != Successors.end());
        CandidateFallthroughBB = *It;

        // Assert that we don't find more that one
        It = std::find_if(std::next(It), Successors.end(), NotInlined);
        revng_assert(It == Successors.end());

        // We now have two possibilities in front of us:
        // 1) The `CandidateFallthrough` is dominated. We treat it as a normal
        //    case, and when we create the tile we consider it as part of the
        //    current tile.
        // 2) The `CandidateFallthrough` is not dominated. It will not be part
        //    of the current tile.
        if (CandidateFallthroughBB
            and ASTDT.dominates(Node, CandidateFallthroughBB)) {
          PostDomBB = nullptr;
          EndIsPartOfTile = true;
          SwitchBreakTarget = nullptr;
        } else {
          PostDomBB = nullptr;
          EndIsPartOfTile = false;
          SwitchBreakTarget = CandidateFallthroughBB;
        }
      } else {

        // We pre-compute the successor of all the cases that are not inlined
        llvm::SmallVector<BasicBlockNodeT *> SuccOfCases;
        for (BasicBlockNodeT *Case : NotInlinedNotTerminalSuccessors) {
          BasicBlockNodeT *SuccOfCase = getUniqueSuccessorOrNull(Case);
          revng_assert(SuccOfCase);
          SuccOfCases.push_back(SuccOfCase);
        }

        // Criterion 1:
        // For each successor, check if a certain one is successor of all the
        // other cases
        for (BasicBlockNodeT *Case : NotInlinedSuccessors) {
          unsigned Count = std::count(SuccOfCases.begin(),
                                      SuccOfCases.end(),
                                      Case);
          if (Count > 0) {
            if ((getUniqueSuccessorOrNull(Case)
                 and Count == SuccOfCases.size() - 1)
                or (not getUniqueSuccessorOrNull(Case)
                    and Count == SuccOfCases.size())) {
              revng_assert(not CandidateFallthroughBB);
              CandidateFallthroughBB = Case;
            }
          }
        }

        // Criterion 2:
        // Search for node which is the successor for all the cases (excluding
        // the inlined ones), even if not a successor of `Node` itself
        std::map<BasicBlockNodeT *, unsigned> SuccCounterMap;
        for (BasicBlockNodeT *Elem : SuccOfCases) {
          SuccCounterMap[Elem]++;
        }
        for (const auto &[Key, Value] : SuccCounterMap) {
          if (Value == SuccOfCases.size()) {

            // Criteria 1 and 2 may overlap in some cases, but it is important
            // that if they do they elect the same `CandidateFallthroughBB`
            revng_assert(not CandidateFallthroughBB
                         or Key == CandidateFallthroughBB);
            CandidateFallthroughBB = Key;
          }
        }

        // We now have two possibilities in front of us:
        // 1) The `CandidateFallthrough` is dominated. We consider the node to
        //    be the postdominator of `Node`.
        // 2) The `CandidateFallthrough` is not dominated. It will not be part
        //    of the current tile.
        if (CandidateFallthroughBB
            and ASTDT.dominates(Node, CandidateFallthroughBB)) {
          PostDomBB = CandidateFallthroughBB;
          EndIsPartOfTile = true;
          SwitchBreakTarget = CandidateFallthroughBB;
        } else {
          PostDomBB = nullptr;
          EndIsPartOfTile = false;
          SwitchBreakTarget = CandidateFallthroughBB;
        }
      }

      // The only legit situation for not having elected a
      // `CandidateFallthroughBB` node, is when all the successors are inlined
      // or do not have a successor themselves
      revng_assert(CandidateFallthroughBB
                   or NotInlinedNotTerminalSuccessors.size() == 0);

      SwitchNode::case_container LabeledCases;
      llvm::SmallVector<ASTNode *> SwitchBreakVector;
      bool HasDefault = false;
      for (const auto &[SwitchSucc, EdgeInfos] : Node->labeled_successors()) {

        ASTNode *ASTPointer = nullptr;
        if (SwitchSucc == SwitchBreakTarget) {
          ASTPointer = AST.addSwitchBreak(nullptr);
          SwitchBreakVector.push_back(ASTPointer);
        } else {
          ASTPointer = findASTNode(AST, TileToNodeMap, SwitchSucc);
        }

        revng_assert(nullptr != ASTPointer);

        if (EdgeInfos.Labels.empty()) {
          revng_assert(HasDefault == false);
          HasDefault = true;
        }
        LabeledCases.push_back({ EdgeInfos.Labels, ASTPointer });
      }
      revng_assert(HasDefault or Node->isWeaved() or Node->isDispatcher());

      createSwitchTile(Region,
                       ASTDT,
                       TileToNodeMap,
                       Node,
                       CandidateFallthroughBB,
                       EndIsPartOfTile,
                       NotInlinedNotTerminalSuccessors);

      // If we elected a postdominator for the current tile, we retrieve the
      // corresponding AST node in order to set it as the postdominator on the
      // AST too
      ASTNode *PostDomAST = nullptr;
      if (PostDomBB) {
        PostDomAST = findASTNode(AST, TileToNodeMap, PostDomBB);
      }

      ASTObject.reset(new SwitchNode(Node,
                                     SwitchCondition,
                                     std::move(LabeledCases),
                                     PostDomAST));
      for (ASTNode *Break : SwitchBreakVector) {
        SwitchBreakNode *SwitchBreakCast = llvm::cast<SwitchBreakNode>(Break);
        SwitchNode *Switch = llvm::cast<SwitchNode>(ASTObject.get());
        SwitchBreakCast->setParentSwitch(Switch);
      }
    } else {
      switch (Successors.size()) {

      case 2: {

        switch (Children.size()) {

        case 0: {

          // This means that both our exiting edges have been inlined, and we
          // do not have any immediate postdominator. This situation should
          // not arise, since not having at least one of the two branches
          // dominated is a signal of an error.
          revng_assert(not Node->isBreak() and not Node->isContinue()
                       and not Node->isSet());
          revng_log(CombLogger,
                    "Node " << Node->getNameStr()
                            << " does not dominate any "
                               "node, but has two successors.");
          revng_unreachable("A node does not dominate any node, but has two "
                            "successors.");
        } break;
        case 1: {

          // This means that we have two successors, but we only dominate a
          // single node. This situation is possible only if we have an
          // inlined edge and we have nopostdominator in the tile.
          revng_assert(not Node->isBreak() and not Node->isContinue()
                       and not Node->isSet());
          BasicBlockNode<NodeT> *Successor1 = Successors[0];
          BasicBlockNode<NodeT> *Successor2 = Successors[1];

          ASTNode *Then = nullptr;
          ASTNode *Else = nullptr;
          BasicBlockNodeT *NotDominatedSucc = nullptr;

          using ConstEdge = std::pair<const BasicBlockNodeT *,
                                      const BasicBlockNodeT *>;

          bool Inlined1 = isEdgeInlined(ConstEdge{ Node, Successor1 });
          bool Inlined2 = isEdgeInlined(ConstEdge{ Node, Successor2 });

          if (Inlined1 and Inlined2) {
            revng_assert(ASTDT.dominates(Node, Successor1));
            revng_assert(ASTDT.dominates(Node, Successor2));
            Then = findASTNode(AST, TileToNodeMap, Successor1);
            Else = findASTNode(AST, TileToNodeMap, Successor2);
          } else if (Inlined1) {
            revng_assert(ASTDT.dominates(Node, Successor1));
            Then = findASTNode(AST, TileToNodeMap, Successor1);
            NotDominatedSucc = Successor2;
          } else if (Inlined2) {
            revng_assert(ASTDT.dominates(Node, Successor2));
            Else = findASTNode(AST, TileToNodeMap, Successor2);
            NotDominatedSucc = Successor1;
          } else {
            auto *DominatedSucc = Children[0];
            revng_assert(DominatedSucc == Successor1
                         or DominatedSucc == Successor2);
            NotDominatedSucc = DominatedSucc == Successor1 ? Successor2 :
                                                             Successor1;
            if (DominatedSucc == Successor1)
              Then = findASTNode(AST, TileToNodeMap, Successor1);
            if (DominatedSucc == Successor2)
              Else = findASTNode(AST, TileToNodeMap, Successor2);
          }
          createTile(Region,
                     ASTDT,
                     TileToNodeMap,
                     Node,
                     NotDominatedSucc,
                     false);

          // Build the `IfNode`.
          using UniqueExpr = ASTTree::expr_unique_ptr;
          using ExprDestruct = ASTTree::expr_destructor;
          auto *OriginalNode = Node->getOriginalNode();
          UniqueExpr CondExpr(new AtomicNode(OriginalNode), ExprDestruct());
          ExprNode *Condition = AST.addCondExpr(std::move(CondExpr));

          // Insert the postdominator if the current tile actually has it.
          ASTObject.reset(new IfNode(Node, Condition, Then, Else, nullptr));
        } break;
        case 2: {

          // TODO: Handle this case.
          // This means that we have two successors, and we dominate both the
          // two successors, or one successor and the postdominator.
          revng_assert(not Node->isBreak() and not Node->isContinue()
                       and not Node->isSet());
          BasicBlockNode<NodeT> *Successor1 = Successors[0];
          BasicBlockNode<NodeT> *Successor2 = Successors[1];

          // First of all, check if one of the successors can also be
          // considered as the immediate postdominator of the tile.
          auto *SuccOfSucc1 = getDirectSuccessor(Successor1);
          auto *SuccOfSucc2 = getDirectSuccessor(Successor2);

          using ConstEdge = std::pair<const BasicBlockNodeT *,
                                      const BasicBlockNodeT *>;
          bool Inlined1 = isEdgeInlined(ConstEdge{ Node, Successor1 });
          bool Inlined2 = isEdgeInlined(ConstEdge{ Node, Successor2 });

          ASTNode *Then = nullptr;
          ASTNode *Else = nullptr;
          BasicBlockNode<NodeT> *PostDomBB = nullptr;

          if (SuccOfSucc1 != SuccOfSucc2) {

            if (Inlined1 and Inlined2) {
              revng_assert(ASTDT.dominates(Node, Successor1));
              revng_assert(ASTDT.dominates(Node, Successor2));
              Then = findASTNode(AST, TileToNodeMap, Successor1);
              Else = findASTNode(AST, TileToNodeMap, Successor2);
            } else if (Inlined1) {
              revng_assert(ASTDT.dominates(Node, Successor1));
              Then = findASTNode(AST, TileToNodeMap, Successor1);
              PostDomBB = Successor2;
            } else if (Inlined2) {
              revng_assert(ASTDT.dominates(Node, Successor2));
              Else = findASTNode(AST, TileToNodeMap, Successor2);
              PostDomBB = Successor1;
            } else if (SuccOfSucc1 == Successor2) {
              revng_assert(SuccOfSucc2 != Successor1);
              revng_assert(ASTDT.dominates(Node, Successor1));
              Then = findASTNode(AST, TileToNodeMap, Successor1);
              PostDomBB = Successor2;
            } else if (SuccOfSucc2 == Successor1) {
              revng_assert(SuccOfSucc1 != Successor2);
              revng_assert(ASTDT.dominates(Node, Successor2));
              Else = findASTNode(AST, TileToNodeMap, Successor2);
              PostDomBB = Successor1;
            } else {
              revng_assert(ASTDT.dominates(Node, Successor1));
              revng_assert(ASTDT.dominates(Node, Successor2));
              Then = findASTNode(AST, TileToNodeMap, Successor1);
              Else = findASTNode(AST, TileToNodeMap, Successor2);
            }
          } else {
            revng_assert(ASTDT.dominates(Node, Successor1));
            revng_assert(ASTDT.dominates(Node, Successor2));
            Then = findASTNode(AST, TileToNodeMap, Successor1);
            Else = findASTNode(AST, TileToNodeMap, Successor2);
          }

          // Build the `IfNode`.
          using UniqueExpr = ASTTree::expr_unique_ptr;
          using ExprDestruct = ASTTree::expr_destructor;
          auto *OriginalNode = Node->getOriginalNode();
          UniqueExpr CondExpr(new AtomicNode(OriginalNode), ExprDestruct());
          ExprNode *Condition = AST.addCondExpr(std::move(CondExpr));

          // Insert the postdominator if the current tile actually has it.
          ASTNode *PostDom = nullptr;
          if (PostDomBB)
            PostDom = findASTNode(AST, TileToNodeMap, PostDomBB);

          ASTObject.reset(new IfNode(Node, Condition, Then, Else, PostDom));

          if (PostDomBB) {
            createTile(Region, ASTDT, TileToNodeMap, Node, PostDomBB, true);
          } else {
            createTile(Region, ASTDT, TileToNodeMap, Node, PostDomBB, false);
          }
        } break;
        case 3: {

          // This is the standard situation, we have two successors, we
          // dominate both of them and we also dominate the postdominator
          // node.
          revng_assert(not Node->isBreak() and not Node->isContinue()
                       and not Node->isSet());

          // Check that our successor nodes are also in the dominated node
          // vector.
          BasicBlockNode<NodeT> *Successor1 = Successors[0];
          BasicBlockNode<NodeT> *Successor2 = Successors[1];
          revng_assert(containsSmallVector(Children, Successor1));
          revng_assert(containsSmallVector(Children, Successor2));

          ASTNode *Then = findASTNode(AST, TileToNodeMap, Successor1);
          ASTNode *Else = findASTNode(AST, TileToNodeMap, Successor2);

          // Retrieve the successors of the `then` and `else` nodes. We expect
          // the successor to be identical due to the structure of the tile we
          // are covering. And we expect it to be the PostDom node of the
          // tile.
          BasicBlockNode<NodeT> *PostDomBB = findCommonPostDom(Successor1,
                                                               Successor2);
          ASTNode *PostDom = nullptr;
          if (PostDomBB != nullptr) {
            // Check that the postdom is between the nodes dominated by the
            // current node.
            revng_assert(containsSmallVector(Children, PostDomBB));
            PostDom = findASTNode(AST, TileToNodeMap, PostDomBB);
          }

          // Build the `IfNode`.
          using UniqueExpr = ASTTree::expr_unique_ptr;
          using ExprDestruct = ASTTree::expr_destructor;
          auto *OriginalNode = Node->getOriginalNode();
          UniqueExpr CondExpr(new AtomicNode(OriginalNode), ExprDestruct());
          ExprNode *Condition = AST.addCondExpr(std::move(CondExpr));
          ASTObject.reset(new IfNode(Node, Condition, Then, Else, PostDom));

          if (PostDomBB) {
            createTile(Region, ASTDT, TileToNodeMap, Node, PostDomBB, true);
          } else {
            createTile(Region, ASTDT, TileToNodeMap, Node, PostDomBB, false);
          }
        } break;

        default: {
          revng_log(CombLogger,
                    "Node: " << Node->getNameStr() << " dominates "
                             << Children.size() << " nodes");
          revng_unreachable("Node directly dominates more than 3 other "
                            "nodes");
        } break;
        }
      } break;

      case 1: {
        switch (Children.size()) {
        case 0: {

          // In this situation, we don't need to actually add as a successor
          // of the current node the single successor which is not dominated.
          // Therefore, the successor will not be a successor on the AST.
          revng_assert(not Node->isBreak() and not Node->isContinue());
          if (Node->isSet()) {
            ASTObject.reset(new SetNode(Node));
          } else {
            ASTObject.reset(new CodeNode(Node, nullptr));
          }
        } break;

        case 1: {

          // In this situation, we dominate the only successor of the current
          // node. The successor therefore will be an actual successor on the
          // AST.
          revng_assert(not Node->isBreak() and not Node->isContinue());
          revng_assert(Successors[0] == Children[0]);
          auto *Succ = findASTNode(AST, TileToNodeMap, Children[0]);
          if (Node->isSet()) {
            ASTObject.reset(new SetNode(Node, Succ));
          } else {
            ASTObject.reset(new CodeNode(Node, Succ));
          }
          createTile(Region, ASTDT, TileToNodeMap, Node, Children[0], true);
        } break;

        default: {
          revng_log(CombLogger,
                    "Node: " << Node->getNameStr() << " dominates "
                             << Children.size() << "nodes");
          revng_unreachable("Node with 1 successor dominates an incorrect "
                            "number of nodes");
        } break;
        }
      } break;

      case 0: {
        if (Node->isBreak())
          ASTObject.reset(new BreakNode(Node));
        else if (Node->isContinue())
          ASTObject.reset(new ContinueNode(Node));
        else if (Node->isSet())
          ASTObject.reset(new SetNode(Node));
        else if (Node->isEmpty() or Node->isCode())
          ASTObject.reset(new CodeNode(Node, nullptr));
        else
          revng_abort();
      } break;

      default: {
        revng_log(CombLogger,
                  "Node: " << Node->getNameStr() << " dominates "
                           << Children.size() << " nodes");
        revng_unreachable("Node directly dominates more than 3 other nodes");
      } break;
      }
    }
    AST.addASTNode(Node, std::move(ASTObject));
  }

  // Set in the ASTTree object the root node.
  BasicBlockNode<NodeT> *Root = ASTDT.getRootNode()->getBlock();
  revng_assert(Root);
  ASTNode *RootNode = AST.findASTNode(Root);
  AST.setRoot(RootNode);
}

inline void normalize(ASTTree &AST, const llvm::Function &F) {

  // AST dumper helper
  GHASTDumper Dumper(CombLogger, F, AST, "normalize");

  // Serialize the graph starting from the root node.
  CombLogger << "Serializing first AST draft:\n";
  Dumper.log("first-draft");

  // Create sequence nodes.
  CombLogger << "Performing sequence insertion:\n";
  ASTNode *RootNode = AST.getRoot();
  RootNode = createSequence(AST, RootNode);
  AST.setRoot(RootNode);
  Dumper.log("after-sequence");

  // Simplify useless sequence nodes.
  CombLogger << "Performing useless dummies simplification:\n";
  simplifyDummies(AST, RootNode);
  Dumper.log("after-dummies-removal");

  // Simplify useless sequence nodes.
  CombLogger << "Performing useless sequence simplification:\n";
  RootNode = simplifyAtomicSequence(AST, RootNode);
  AST.setRoot(RootNode);
  Dumper.log("after-sequence-simplification");

  Dumper.log("final");
}
