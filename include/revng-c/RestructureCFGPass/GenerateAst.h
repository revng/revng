#ifndef REVNGC_RESTRUCTURE_CFG_GENERATEAST_H
#define REVNGC_RESTRUCTURE_CFG_GENERATEAST_H

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <sys/stat.h>

// LLVM includes
#include <llvm/ADT/DepthFirstIterator.h>
#include <llvm/ADT/GraphTraits.h>
#include <llvm/ADT/PostOrderIterator.h>
#include <llvm/ADT/SCCIterator.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/SmallSet.h>
#include <llvm/IR/Dominators.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/GenericDomTreeConstruction.h>
#include <llvm/Support/raw_os_ostream.h>

// revng includes
#include <revng/BasicAnalyses/GeneratedCodeBasicInfo.h>
#include <revng/Support/IRHelpers.h>

// Local libraries includes
#include "revng-c/ADT/ReversePostOrderTraversal.h"
#include "revng-c/RestructureCFGPass/ASTTree.h"
#include "revng-c/RestructureCFGPass/BasicBlockNodeBB.h"
#include "revng-c/RestructureCFGPass/MetaRegionBB.h"
#include "revng-c/RestructureCFGPass/RegionCFGTree.h"
#include "revng-c/RestructureCFGPass/Utils.h"

// Helper function that visit an AST tree and creates the sequence nodes
inline ASTNode *createSequence(ASTTree &Tree, ASTNode *RootNode) {
  SequenceNode *RootSequenceNode = Tree.addSequenceNode();
  RootSequenceNode->addNode(RootNode);

  for (ASTNode *Node : RootSequenceNode->nodes()) {
    if (auto *If = llvm::dyn_cast<IfNode>(Node)) {
      if (If->hasThen()) {
        If->setThen(createSequence(Tree, If->getThen()));
      }
      if (If->hasElse()) {
        If->setElse(createSequence(Tree, If->getElse()));
      }
    } else if (llvm::isa<CodeNode>(Node)) {
      // TODO: confirm that doesn't make sense to process a code node.
    } else if (llvm::isa<ScsNode>(Node)) {
      // TODO: confirm that this phase is not needed since the processing is
      //       done inside the processing of each SCS region.
    } else if (auto *Switch = llvm::dyn_cast<SwitchNode>(Node)) {

      for (auto &LabelCasePair : Switch->cases())
        LabelCasePair.second = createSequence(Tree, LabelCasePair.second);

      if (ASTNode *Default = Switch->getDefault())
        Switch->replaceDefault(createSequence(Tree, Default));

    } else if (llvm::isa<BreakNode>(Node)) {
      // Stop here during the analysis.
    } else if (llvm::isa<ContinueNode>(Node)) {
      // Stop here during the analysis.
    } else if (llvm::isa<SequenceNode>(Node)) {
      // Stop here during the analysis.
    } else if (llvm::isa<SetNode>(Node)) {
      // Stop here during the analysis.
    } else {
      revng_abort("AST node type not expected");
    }
  }

  return RootSequenceNode;
}

// Helper function that simplifies useless dummy nodes
inline void simplifyDummies(ASTNode *RootNode) {

  switch (RootNode->getKind()) {

  case ASTNode::NK_List: {
    auto *Sequence = llvm::cast<SequenceNode>(RootNode);
    std::vector<ASTNode *> UselessDummies;
    for (ASTNode *Node : Sequence->nodes()) {
      if (Node->isEmpty()) {
        UselessDummies.push_back(Node);
      } else {
        simplifyDummies(Node);
      }
    }
    for (ASTNode *Node : UselessDummies) {
      Sequence->removeNode(Node);
    }
  } break;

  case ASTNode::NK_If: {
    auto *If = llvm::cast<IfNode>(RootNode);
    if (If->hasThen()) {
      simplifyDummies(If->getThen());
    }
    if (If->hasElse()) {
      simplifyDummies(If->getElse());
    }
  } break;

  case ASTNode::NK_Switch: {

    auto *Switch = llvm::cast<SwitchNode>(RootNode);

    for (auto &LabelCaseNodePair : Switch->cases())
      simplifyDummies(LabelCaseNodePair.second);

    if (auto *Default = Switch->getDefault())
      simplifyDummies(Default);

  } break;

  case ASTNode::NK_Scs:
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

// Helper function which simplifies sequence nodes composed by a single AST
// node.
inline ASTNode *simplifyAtomicSequence(ASTNode *RootNode) {
  switch (RootNode->getKind()) {

  case ASTNode::NK_List: {
    auto *Sequence = llvm::cast<SequenceNode>(RootNode);
    switch (Sequence->listSize()) {

    case 0:
      RootNode = nullptr;
      break;

    case 1:
      RootNode = simplifyAtomicSequence(Sequence->getNodeN(0));
      break;

    default:
      bool Empty = true;
      for (ASTNode *&Node : Sequence->nodes()) {
        Node = simplifyAtomicSequence(Node);
        if (nullptr != Node)
          Empty = false;
      }
      revng_assert(not Empty);
    }
  } break;

  case ASTNode::NK_If: {
    auto *If = llvm::cast<IfNode>(RootNode);

    if (If->hasThen())
      If->setThen(simplifyAtomicSequence(If->getThen()));

    if (If->hasElse())
      If->setElse(simplifyAtomicSequence(If->getElse()));

  } break;

  case ASTNode::NK_Switch: {

    auto *Switch = llvm::cast<SwitchNode>(RootNode);

    // In case the recursive call to `simplifyAtomicSequence` gives origin to a
    // complete simplification of the default node of the switch, setting its
    // corresponding `ASTNode` to `nullptr` already does the job, since having
    // the corresponding `Default` field set to `nullptr` means that the switch
    // node has no default.
    if (ASTNode *Default = Switch->getDefault()) {
      auto *NewDefault = simplifyAtomicSequence(Default);
      if (NewDefault != Default)
        Switch->replaceDefault(NewDefault);
    }

    auto LabelCasePairIt = Switch->cases().begin();
    auto LabelCasePairEnd = Switch->cases().end();
    while (LabelCasePairIt != LabelCasePairEnd) {
      auto *NewCaseNode = simplifyAtomicSequence(LabelCasePairIt->second);
      if (nullptr == NewCaseNode) {
        revng_assert(nullptr == Switch->getDefault());
        LabelCasePairIt = Switch->cases().erase(LabelCasePairIt);
        LabelCasePairEnd = Switch->cases().end();
      } else {
        LabelCasePairIt->second = NewCaseNode;
        ++LabelCasePairIt;
      }
    }

  } break;

  case ASTNode::NK_Scs: {
    // TODO: check if this is not needed as the simplification is done for each
    //       SCS region.
    // After flattening this situation may arise again.
    auto *Scs = llvm::cast<ScsNode>(RootNode);
    if (Scs->hasBody())
      Scs->setBody(simplifyAtomicSequence(Scs->getBody()));
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

  return RootNode;
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
findCommonPostDom(BasicBlockNode<NodeT> *Succ1, BasicBlockNode<NodeT> *Succ2) {

  // Retrieve the successor of the two successors of the `IfNode`, and check
  // that or the retrieved node is equal for both the successors, or it does
  // not exists for both of them.
  BasicBlockNode<NodeT> *SuccOfSucc1 = nullptr;
  BasicBlockNode<NodeT> *SuccOfSucc2 = nullptr;
  if (Succ1->successor_size() == 1)
    SuccOfSucc1 = getDirectSuccessor(Succ1);
  else
    revng_abort();
  if (Succ2->successor_size() == 1)
    SuccOfSucc2 = getDirectSuccessor(Succ2);
  else
    revng_abort();
  revng_assert(SuccOfSucc1 == SuccOfSucc2);

  return SuccOfSucc1;
}

template<class NodeT>
inline ASTNode *findASTNode(ASTTree &AST,
                            typename RegionCFG<NodeT>::BBNodeMap &TileToNodeMap,
                            BasicBlockNode<NodeT> *Node) {
  if (TileToNodeMap.count(Node) != 0) {
    Node = TileToNodeMap.at(Node);
  }

  return AST.findASTNode(Node);
}

template<class NodeT>
inline void createTile(RegionCFG<NodeT> &Graph,
                       typename RegionCFG<NodeT>::BBNodeMap &TileToNodeMap,
                       BasicBlockNode<NodeT> *Node,
                       BasicBlockNode<NodeT> *End) {

  using BBNodeT = BasicBlockNode<NodeT>;
  using BasicBlockNodeTVect = std::vector<BBNodeT *>;
  using EdgeDescriptor = typename BBNodeT::EdgeDescriptor;

  // Create the new tile node.
  BBNodeT *Tile = Graph.addTile();

  // Move all the edges incoming in the head of the collapsed region to the tile
  // node.
  BasicBlockNodeTVect Predecessors;
  for (BBNodeT *Predecessor : Node->predecessors()) {
    Predecessors.push_back(Predecessor);
  }
  for (BBNodeT *Predecessor : Predecessors) {
    moveEdgeTarget(EdgeDescriptor(Predecessor, Node), Tile);
  }

  // Move all the edges exiting from the postdominator node of the collapsed
  // region to the tile node, if the `End` passed as an argument is
  // `!= nullptr`.
  if (End != nullptr) {
    BasicBlockNodeTVect Successors;
    for (BBNodeT *Successor : End->successors()) {
      Successors.push_back(Successor);
    }
    for (BBNodeT *Successor : Successors) {
      moveEdgeSource(EdgeDescriptor(End, Successor), Tile);
    }
  }

  // Update the map containing the mapping between tiles and the head node which
  // gave origin to a certain tile.
  TileToNodeMap[Tile] = Node;
}

template<class NodeT>
inline void generateAst(RegionCFG<NodeT> &Region,
                        ASTTree &AST,
                        typename RegionCFG<NodeT>::DuplicationMap &NDuplicates,
                        std::map<BasicBlockNodeBB *, ASTTree> &CollapsedMap) {
  // Define some using used in all the function body.
  using BasicBlockNodeT = typename RegionCFG<NodeT>::BasicBlockNodeT;
  using BasicBlockNodeTVect = typename RegionCFG<NodeT>::BasicBlockNodeTVect;
  using BBNodeT = BasicBlockNodeT;

  // Get some fields of `RegionCFG`.
  std::string RegionName = Region.getRegionName();
  std::string FunctionName = Region.getFunctionName();

  RegionCFG<NodeT> &Graph = Region;

  Graph.markUnexpectedPCAsInlined();

  if (CombLogger.isEnabled()) {
    CombLogger << "Weaveing region " + RegionName + "\n";
    Graph.dumpDotOnFile("weaves", FunctionName, "PREWEAVE");
  }

  // Invoke the weave function.
  Graph.weave();

  if (CombLogger.isEnabled()) {
    Graph.dumpDotOnFile("weaves", FunctionName, "POSTWEAVE");

    CombLogger << "Inflating region " + RegionName + "\n";
    Graph.dumpDotOnFile("dots", FunctionName, "PRECOMB");
  }

  Graph.inflate();
  if (CombLogger.isEnabled()) {
    Graph.dumpDotOnFile("dots", FunctionName, "POSTCOMB");
  }

  // Compute the NDuplicates, which will be used later.
  // TODO: this now doesn't run multiple
  for (BasicBlockNodeBB *BBNode : Graph.nodes()) {
    if (BBNode->isCode()) {
      llvm::BasicBlock *BB = BBNode->getOriginalNode();
      NDuplicates[BB] += 1;
    }
  }

  // TODO: factorize out the AST generation phase.
  llvm::DominatorTreeBase<BasicBlockNode<NodeT>, false> ASTDT;
  ASTDT.recalculate(Graph);
  ASTDT.updateDFSNumbers();

  CombLogger << DoLog;

  std::map<BasicBlockNode<NodeT> *, BasicBlockNode<NodeT> *> TileToNodeMap;

  BasicBlockNodeTVect PONodes;
  for (auto POIt = po_begin(&Graph); POIt != po_end(&Graph); POIt++) {
    PONodes.push_back(*POIt);
  }

  unsigned Counter = 0;
  for (BasicBlockNode<NodeT> *Node : PONodes) {
    ASTDT.recalculate(Graph);

    if (CombLogger.isEnabled()) {
      Counter++;
      Graph.dumpDotOnFile("dots",
                          FunctionName,
                          "AST-" + std::to_string(Counter));
    }

    // Collect the children nodes in the dominator tree.
    llvm::SmallVector<decltype(Node), 8> Children;
    {
      const auto &DomTreeChildren = ASTDT[Node]->getChildren();
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
      const auto &[It, New] = CollapsedMap.insert({ Node, ASTTree() });
      ASTTree &CollapsedAST = It->second;
      if (New)
        generateAst(*BodyGraph, CollapsedAST, NDuplicates, CollapsedMap);
      ASTNode *Body = AST.copyASTNodesFrom(CollapsedAST);

      switch (Successors.size()) {

      case 0: {
        ASTObject.reset(new ScsNode(Node, Body));
      } break;

      case 1: {
        auto *Succ = Successors[0];
        if (ASTDT.dominates(Node, Succ)) {
          ASTNode *ASTChild = findASTNode(AST, TileToNodeMap, Succ);
          ASTObject.reset(new ScsNode(Node, Body, ASTChild));
          createTile(Graph, TileToNodeMap, Node, Succ);
        } else {
          ASTObject.reset(new ScsNode(Node, Body));
        }
      } break;

      default:
        revng_abort();
      }

    } else if (Node->isDispatcher() or isASwitch(Node)) {

      revng_assert(Node->isCode() or Node->isDispatcher());

      llvm::Value *SwitchCondition = nullptr;
      if (not Node->isDispatcher()) {
        llvm::BasicBlock *OriginalNode = Node->getOriginalNode();
        llvm::Instruction *Terminator = OriginalNode->getTerminator();
        llvm::SwitchInst *Switch = llvm::cast<llvm::SwitchInst>(Terminator);
        SwitchCondition = Switch->getCondition();
      }
      revng_assert(SwitchCondition or Node->isDispatcher());

      SwitchNode::case_container LabeledCases;
      BasicBlockNode<NodeT> *DefaultNode = nullptr;
      ASTNode *DefaultASTNode = nullptr;

      for (const auto &[SwitchSucc, EdgeInfos] : Node->labeled_successors()) {
        ASTNode *ASTPointer = findASTNode(AST, TileToNodeMap, SwitchSucc);
        revng_assert(nullptr != ASTPointer);

        if (EdgeInfos.Labels.empty()) {
          revng_assert(nullptr == DefaultNode);
          revng_assert(nullptr == DefaultASTNode);
          DefaultNode = SwitchSucc;
          DefaultASTNode = ASTPointer;
        }

        LabeledCases.push_back({ EdgeInfos.Labels, ASTPointer });
      }
      revng_assert(DefaultASTNode or Node->isWeaved() or Node->isDispatcher());
      revng_assert(DefaultNode or Node->isWeaved() or Node->isDispatcher());

      revng_assert(Node->successor_size() == LabeledCases.size());

      // TODO: verify wheter this assertion is really necessary and why. With
      //       the tiling, this criterion is not respected anymore. You are
      //       not obliged to dominate a node in order to consume it in your
      //       tile.
      // revng_assert(Children.size() >= Node->successor_size());
      ASTNode *PostDomASTNode = nullptr;
      BasicBlockNodeT *PostDomBB = nullptr;

      if (Children.size() > Node->successor_size()) {

        // There are some children on the dominator tree that are not
        // successors on the graph. It should be at most one, which is the
        // post-dominator.
        const auto NotSuccessor = [&Node](const auto *Child) {
          auto It = Node->successors().begin();
          auto End = Node->successors().end();
          return std::find(It, End, Child) == End;
        };

        auto It = std::find_if(Children.begin(), Children.end(), NotSuccessor);

        // Assert that we found one.
        revng_assert(It != Children.end());

        PostDomASTNode = findASTNode(AST, TileToNodeMap, *It);
        PostDomBB = *It;

        // Assert that we don't find more than one.
        It = std::find_if(std::next(It), Children.end(), NotSuccessor);
        revng_assert(It == Children.end());
      }

      createTile(Graph, TileToNodeMap, Node, PostDomBB);

      ASTObject.reset(new SwitchNode(Node,
                                     SwitchCondition,
                                     std::move(LabeledCases),
                                     DefaultASTNode,
                                     PostDomASTNode));
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

          ASTNode *Then = findASTNode(AST, TileToNodeMap, Successor1);
          ASTNode *Else = findASTNode(AST, TileToNodeMap, Successor2);

          using ConstEdge = std::pair<const BasicBlockNodeT *,
                                      const BasicBlockNodeT *>;
          if (isEdgeInlined(ConstEdge{ Node, Successor1 })) {

            Else = nullptr;
            createTile(Graph, TileToNodeMap, Node, Successor2);

          } else if (isEdgeInlined(ConstEdge{ Node, Successor2 })) {

            Then = nullptr;
            createTile(Graph, TileToNodeMap, Node, Successor1);

          } else {
            auto *DominatedSucc = Children[0];
            revng_assert(DominatedSucc == Successor1
                         or DominatedSucc == Successor2);
            auto *NotDominatedSucc = DominatedSucc == Successor1 ? Successor2 :
                                                                   Successor1;
            Then = DominatedSucc == Successor1 ? Then : nullptr;
            Else = DominatedSucc == Successor2 ? Else : nullptr;

            createTile(Graph, TileToNodeMap, Node, NotDominatedSucc);
          }

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

          ASTNode *Then = findASTNode(AST, TileToNodeMap, Successor1);
          ASTNode *Else = findASTNode(AST, TileToNodeMap, Successor2);

          // First of all, check if one of the successors can also be
          // considered as the immediate postdominator of the tile.
          auto *SuccOfSucc1 = getDirectSuccessor(Successor1);
          auto *SuccOfSucc2 = getDirectSuccessor(Successor2);
          revng_assert(SuccOfSucc1 != SuccOfSucc2 or nullptr == SuccOfSucc1);

          BasicBlockNode<NodeT> *PostDomBB = nullptr;

          using ConstEdge = std::pair<const BasicBlockNodeT *,
                                      const BasicBlockNodeT *>;
          if (isEdgeInlined(ConstEdge{ Node, Successor1 })) {
            Else = nullptr;
            PostDomBB = Successor2;
          } else if (isEdgeInlined(ConstEdge{ Node, Successor2 })) {
            Then = nullptr;
            PostDomBB = Successor1;
          } else if (SuccOfSucc1 == Successor2) {
            revng_assert(SuccOfSucc2 != Successor1);
            PostDomBB = Successor2;
            Else = nullptr;
          } else if (SuccOfSucc2 == Successor1) {
            revng_assert(SuccOfSucc1 != Successor2);
            PostDomBB = Successor1;
            Then = nullptr;
          } else {
            revng_assert(ASTDT.dominates(Node, Successor1));
            revng_assert(ASTDT.dominates(Node, Successor2));
          }

          // Build the `IfNode`.
          using UniqueExpr = ASTTree::expr_unique_ptr;
          using ExprDestruct = ASTTree::expr_destructor;
          auto *OriginalNode = Node->getOriginalNode();
          UniqueExpr CondExpr(new AtomicNode(OriginalNode), ExprDestruct());
          ExprNode *Condition = AST.addCondExpr(std::move(CondExpr));

          // Insert the postdominator if the current tile actually has it.
          if (PostDomBB) {
            ASTNode *PostDom = findASTNode(AST, TileToNodeMap, PostDomBB);
            ASTObject.reset(new IfNode(Node, Condition, Then, Else, PostDom));

          } else {
            ASTObject.reset(new IfNode(Node, Condition, Then, Else, nullptr));
          }

          createTile(Graph, TileToNodeMap, Node, PostDomBB);
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
          revng_assert(PostDomBB != nullptr);
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

          createTile(Graph, TileToNodeMap, Node, PostDomBB);
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
          // Therefore, the successor will not be a succesor on the AST.
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
          auto *Succ = findASTNode(AST, TileToNodeMap, Children[0]);
          BBNodeT *SuccBB = Children[0];
          if (Node->isSet()) {
            ASTObject.reset(new SetNode(Node, Succ));
          } else {
            ASTObject.reset(new CodeNode(Node, Succ));
          }

          createTile(Graph, TileToNodeMap, Node, SuccBB);
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
          ASTObject.reset(new BreakNode());
        else if (Node->isContinue())
          ASTObject.reset(new ContinueNode());
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
  ASTNode *RootNode = AST.findASTNode(Root);
  AST.setRoot(RootNode);
}

inline void normalize(ASTTree &AST, std::string FunctionName) {
  // Serialize the graph starting from the root node.
  CombLogger << "Serializing first AST draft:\n";
  if (CombLogger.isEnabled()) {
    AST.dumpOnFile("ast", FunctionName, "First-draft");
  }

  // Create sequence nodes.
  CombLogger << "Performing sequence insertion:\n";
  ASTNode *RootNode = AST.getRoot();
  RootNode = createSequence(AST, RootNode);
  AST.setRoot(RootNode);
  if (CombLogger.isEnabled()) {
    AST.dumpOnFile("ast", FunctionName, "After-sequence");
  }

  // Simplify useless sequence nodes.
  CombLogger << "Performing useless dummies simplification:\n";
  simplifyDummies(RootNode);
  if (CombLogger.isEnabled()) {
    AST.dumpOnFile("ast", FunctionName, "After-dummies-removal");
  }

  // Simplify useless sequence nodes.
  CombLogger << "Performing useless sequence simplification:\n";
  RootNode = simplifyAtomicSequence(RootNode);
  AST.setRoot(RootNode);
  if (CombLogger.isEnabled()) {
    AST.dumpOnFile("ast", FunctionName, "After-sequence-simplification");
  }

  // Remove danling nodes (possibly created by the de-optimization pass, after
  // disconnecting the first CFG node corresponding to the simplified AST
  // node), and superfluos dummy nodes.
  // TODO: disabled this phase so that the flattening can correctly skip over
  //       the tile nodes and correctly flatten the AST at least (while
  //       leaving the `RegionCFG` in a broken state, but no user of it exists
  //       anymore. Build the AST generation phase so that the AST is built
  //       complete on the fly, and not flattened in a successive phase.
  // removeNotReachables();
  // purgeTrivialDummies();
}
#endif // REVNGC_RESTRUCTURE_CFG_GENERATEAST_H
