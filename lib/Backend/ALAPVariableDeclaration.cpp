
/// \file ALAPVariableDeclaration.cpp
/// ALAP Variable Declaration Computation
///

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <list>
#include <type_traits>

#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/GenericDomTreeConstruction.h"

#include "revng/ADT/GenericGraph.h"
#include "revng/ADT/RecursiveCoroutine.h"

#include "revng-c/RestructureCFG/ASTNode.h"
#include "revng-c/RestructureCFG/ASTTree.h"
#include "revng-c/Support/DecompilationHelpers.h"
#include "revng-c/Support/FunctionTags.h"

#include "ALAPVariableDeclaration.h"

using BBGHASTNodeMap = std::multimap<const llvm::BasicBlock *, const ASTNode *>;

static RecursiveCoroutine<void>
collectExprBB(ExprNode *Expr, const ASTNode *Node, BBGHASTNodeMap &ResultMap) {
  switch (Expr->getKind()) {
  case ExprNode::NodeKind::NK_ValueCompare:
  case ExprNode::NodeKind::NK_LoopStateCompare: {
    // There is no associated `BasicBlock`
  } break;
  case ExprNode::NodeKind::NK_Atomic: {
    auto *Atomic = llvm::cast<AtomicNode>(Expr);
    llvm::BasicBlock *BB = Atomic->getConditionalBasicBlock();
    ResultMap.insert(std::make_pair(BB, Node));
  } break;
  case ExprNode::NodeKind::NK_Not: {
    auto *Not = llvm::cast<NotNode>(Expr);
    ExprNode *NegatedNode = Not->getNegatedNode();
    rc_recur collectExprBB(NegatedNode, Node, ResultMap);
  } break;
  case ExprNode::NodeKind::NK_And:
  case ExprNode::NodeKind::NK_Or: {
    auto *Binary = llvm::cast<BinaryNode>(Expr);
    const auto &[LHS, RHS] = Binary->getInternalNodes();
    rc_recur collectExprBB(LHS, Node, ResultMap);
    rc_recur collectExprBB(RHS, Node, ResultMap);
  } break;
  default:
    revng_unreachable();
  }

  rc_return;
}

class BBToASTNodeMapping {
  BBGHASTNodeMap BBToASTNode;
  const ASTTree &GHAST;

private:
  RecursiveCoroutine<void> computeImpl(const ASTNode *Node) {
    switch (Node->getKind()) {
    case ASTNode::NK_List: {
      auto *Seq = llvm::cast<SequenceNode>(Node);

      // A `SequenceNode` should not have an associated `BasicBlock`
      revng_assert(Seq->getOriginalBB() == nullptr);

      // Recursively visit on each element of the `SequenceNode`
      for (ASTNode *Child : Seq->nodes()) {
        rc_recur computeImpl(Child);
      }
    } break;
    case ASTNode::NK_Scs: {
      auto *Scs = llvm::cast<ScsNode>(Node);

      // An `ScsNode` should not have an associated `BasicBlock`
      revng_assert(Scs->getOriginalBB() == nullptr);

      // Inspect the related condition containing the `IfNode` associated to the
      // execution of the loop
      if (not Scs->isWhileTrue()) {
        IfNode *If = Scs->getRelatedCondition();
        ExprNode *IfExpr = If->getCondExpr();
        collectExprBB(IfExpr, Scs, BBToASTNode);
      }

      if (Scs->hasBody()) {
        rc_recur computeImpl(Scs->getBody());
      }
    } break;
    case ASTNode::NK_If: {
      auto *If = llvm::cast<IfNode>(Node);

      // Add the original `BB` in the `ResultMap`
      llvm::BasicBlock *BB = If->getOriginalBB();
      BBToASTNode.insert(std::make_pair(BB, If));

      ExprNode *IfExpr = If->getCondExpr();
      collectExprBB(IfExpr, If, BBToASTNode);

      if (If->hasThen()) {
        rc_recur computeImpl(If->getThen());
      }
      if (If->hasElse()) {
        rc_recur computeImpl(If->getElse());
      }
    } break;
    case ASTNode::NK_Switch: {
      auto *Switch = llvm::cast<SwitchNode>(Node);

      // Add the original `BB` in the `ResultMap`
      llvm::BasicBlock *BB = Switch->getOriginalBB();
      BBToASTNode.insert(std::make_pair(BB, Switch));

      for (auto &LabelCasePair : Switch->cases_const_range()) {
        ASTNode *Case = LabelCasePair.second;
        rc_recur computeImpl(Case);
      }
    } break;
    case ASTNode::NK_Code: {

      // Add the original `BB` in the `ResultMap`
      auto *Code = llvm::cast<CodeNode>(Node);
      llvm::BasicBlock *BB = Code->getOriginalBB();
      BBToASTNode.insert(std::make_pair(BB, Code));
    } break;
    case ASTNode::NK_Continue: {
      auto *Continue = llvm::cast<ContinueNode>(Node);

      // A `ContinueNode` should not have an associated `BasicBlock`
      revng_assert(Continue->getOriginalBB() == nullptr);

      if (Continue->hasComputation()) {
        auto *If = llvm::cast<IfNode>(Continue->getComputationIfNode());
        ExprNode *IfExpr = If->getCondExpr();
        collectExprBB(IfExpr, Continue, BBToASTNode);
      }
    } break;
    case ASTNode::NK_Set:
    case ASTNode::NK_SwitchBreak:
    case ASTNode::NK_Break: {

      // These nodes should not have an associated `BasicBlock`
      revng_assert(Node->getOriginalBB() == nullptr);
    } break;
    default:
      revng_unreachable();
    }

    rc_return;
  }

public:
  BBToASTNodeMapping(const ASTTree &GHAST) : GHAST(GHAST) {}

  const BBGHASTNodeMap &compute() {
    ASTNode *RootNode = GHAST.getRoot();
    computeImpl(RootNode);
    return BBToASTNode;
  }
};

struct ASTForwardNode {
  ASTForwardNode(const ASTNode *Node) : Node(Node) {}
  const ASTNode *Node;
  const ASTNode *getASTNode() { return Node; }
};

using Node = ForwardNode<ASTForwardNode>;
using ScopeReachabilityGenericGraphTy = GenericGraph<Node>;
using ASTToNodeMapType = std::map<const ASTNode *, Node *>;

// In this struct we compact both the underlying `GenericGraph` that we use to
// materialize the graph, and the mapping between the original `ASTNode` and the
// corresponding `Node` in the `GenericGraph`
struct ScopeReachabilityGraphTy {
  ScopeReachabilityGenericGraphTy Graph;
  ASTToNodeMapType ASTToNodeMap;
};

static RecursiveCoroutine<Node *>
buildNode(ScopeReachabilityGraphTy &ScopeReachabilityGraph,
          const ASTNode *ASTN) {
  // Create a new node in the `GenericGraph`
  Node *GNode = ScopeReachabilityGraph.Graph.addNode(ASTN);
  ScopeReachabilityGraph.ASTToNodeMap[ASTN] = GNode;

  switch (ASTN->getKind()) {
  case ASTNode::NK_List: {
    auto *Seq = llvm::cast<SequenceNode>(ASTN);

    Node *PreviousGChild = GNode;
    for (ASTNode *Child : Seq->nodes()) {
      Node *GChild = rc_recur buildNode(ScopeReachabilityGraph, Child);

      // Create the sibling-to-sibling edge
      if (PreviousGChild)
        PreviousGChild->addSuccessor(GChild);

      // Prepare for the next iteration
      PreviousGChild = GChild;
    }
  } break;
  case ASTNode::NK_Scs: {
    auto *Scs = llvm::cast<ScsNode>(ASTN);

    if (Scs->hasBody()) {
      Node *Body = rc_recur buildNode(ScopeReachabilityGraph, Scs->getBody());
      GNode->addSuccessor(Body);
    }
  } break;
  case ASTNode::NK_If: {
    auto *If = llvm::cast<IfNode>(ASTN);

    if (If->hasThen()) {
      Node *Then = rc_recur buildNode(ScopeReachabilityGraph, If->getThen());
      GNode->addSuccessor(Then);
    }
    if (If->hasElse()) {
      Node *Else = rc_recur buildNode(ScopeReachabilityGraph, If->getElse());
      GNode->addSuccessor(Else);
    }
  } break;
  case ASTNode::NK_Switch: {
    auto *Switch = llvm::cast<SwitchNode>(ASTN);

    for (auto &LabelCasePair : Switch->cases_const_range()) {
      Node *Case = rc_recur buildNode(ScopeReachabilityGraph,
                                      LabelCasePair.second);
      GNode->addSuccessor(Case);
    }
  } break;
  case ASTNode::NK_Code:
  case ASTNode::NK_Continue:
  case ASTNode::NK_Set:
  case ASTNode::NK_SwitchBreak:
  case ASTNode::NK_Break:
    break;
  default:
    revng_unreachable();
  }

  rc_return GNode;
}

/// This helper function can be used to collect all the transitive `User`s
/// starting from an `llvm::Instruction` and stopping at either: a `CallInst`,
/// an `Assign` or a terminator
static RecursiveCoroutine<void>
collectTransitiveUsers(const llvm::Instruction *I,
                       llvm::SmallSet<const llvm::Instruction *, 4> &Users) {

  // This dataflow visit will stop at certain collection points
  if (isAssignment(I) or isCallToIsolatedFunction(I) or I->isTerminator()) {
    // We stop at `@Assign` `TaggedCall`.
    // We stop at calls to isolated functions.
    // We stop at `Terminator` instructions.
    rc_return;
  }

  // In all the other cases, we continue exploring the dataflow
  for (const llvm::User *U : I->users()) {
    const llvm::Instruction *UserInstruction = llvm::cast<llvm::Instruction>(U);
    if (bool New = Users.insert(UserInstruction).second; New)
      rc_recur collectTransitiveUsers(UserInstruction, Users);
  }

  rc_return;
}

static llvm::SmallSet<const ASTNode *, 4>
collectUsageASTNodes(const llvm::Instruction *Variable,
                     const BBGHASTNodeMap &BBToASTNode) {
  // Collect the immediate `User`s of the `Variable`
  llvm::SmallSet<const llvm::Instruction *, 4> UsageInstructions;
  for (const llvm::User *VariableUser : Variable->users()) {
    const llvm::Instruction
      *UserInst = llvm::cast<llvm::Instruction>(VariableUser);
    UsageInstructions.insert(UserInst);
  }

  // For special aggregate types, we need to consider the call itself as
  // a usage
  if (isArtificialAggregateLocalVarDecl(Variable)
      or isHelperAggregateLocalVarDecl(Variable)) {
    UsageInstructions.insert(Variable);
  }

  // Collect the transitive `User`s of the already collected `User`s
  llvm::SmallSet<const llvm::Instruction *, 4> AdditionalUsers;
  for (const llvm::Instruction *UserInst : UsageInstructions) {
    collectTransitiveUsers(UserInst, AdditionalUsers);
  }

  // Merge the obtained results from the transitive `User`s collection
  UsageInstructions.insert(AdditionalUsers.begin(), AdditionalUsers.end());

  // We collect all the AST nodes that are users of the pending variable
  // we are analyzing
  llvm::SmallSet<const ASTNode *, 4> UsageASTNodes;
  for (const llvm::Instruction *UserInst : UsageInstructions) {
    const llvm::BasicBlock *UserBB = UserInst->getParent();

    // We retrieve all the `GHASTNode`s which encompass the `BasicBlock`
    // above
    auto Range = BBToASTNode.equal_range(UserBB);
    for (auto RangeIt = Range.first; RangeIt != Range.second; ++RangeIt) {
      UsageASTNodes.insert(RangeIt->second);
    }
  }

  // Ensure that we find usages for each `Variable` that we need to assign
  revng_assert(not UsageASTNodes.empty());

  return UsageASTNodes;
}

static ScopeReachabilityGraphTy
makeGHASTReachabilityGraph(const ASTTree &GHAST) {
  ScopeReachabilityGraphTy ScopeReachabilityGraph;
  const ASTNode *RootNode = GHAST.getRoot();
  Node *RootGNode = buildNode(ScopeReachabilityGraph, RootNode);
  ScopeReachabilityGraph.Graph.setEntryNode(RootGNode);

  return ScopeReachabilityGraph;
}

static std::map<const llvm::CallInst *, llvm::SmallSet<const ASTNode *, 4>>
computeVariableUsages(const ASTTree &GHAST,
                      PendingVariableListType &PendingVariables) {

  // Compute a `BasicBlock * -> GHASTNode *` multimap representing which
  // `GHASTNode`s covers the usage of a certain `BasicBlock`
  BBToASTNodeMapping Mapping(GHAST);
  const BBGHASTNodeMap &BBToASTNode = Mapping.compute();

  std::map<const llvm::CallInst *, llvm::SmallSet<const ASTNode *, 4>>
    VariableUsages;
  for (auto *Variable : PendingVariables) {
    VariableUsages[Variable] = collectUsageASTNodes(Variable, BBToASTNode);
  }

  return VariableUsages;
}

ASTVarDeclMap computeVarDeclMap(const ASTTree &GHAST,
                                PendingVariableListType &PendingVariables) {

  // 1: build a `GenericGraph` over the GHAST, representing the visibility
  // graph between `ASTNode`s. In this `GenericGraph`, no `SequenceNode` is
  // present, but instead each sibling in a `SequenceNode` has, as child, its
  // immediate successor. This type of view is useful for our analysis.
  ScopeReachabilityGraphTy
    ScopeReachabilityGraph = makeGHASTReachabilityGraph(GHAST);

  // 2: we instantiate the dominator tree over the `ScopeReachabilityGraph`. We
  // will make use of dominance queries as an alternative to performing
  // reachability queries over the same graph. Indeed, we postulate that they
  // are equivalent given the AST like structure of `ScopeReachabilityGraph`,
  // and the fact that the direct successor of each `ASTNode` is a child of it
  // in this representation.
  llvm::DominatorTreeBase<Node, false> DT;
  DT.recalculate(ScopeReachabilityGraph.Graph);

  // 3: pre-compute, for each variable, all the `ASTNode`s that contain an use
  // of the variable
  auto VariableUsages = computeVariableUsages(GHAST, PendingVariables);

  // 4: perform the `Variable` assignment operation.
  // We want to iterate over the nodes of the newly created `GenericGraph`in
  // post order.
  ASTVarDeclMap Result;
  for (auto *TheNode : llvm::post_order(&ScopeReachabilityGraph.Graph)) {

    // At each iteration, iterate over the list of variables that are still to
    // assign
    for (const llvm::CallInst *&Pending : PendingVariables) {

      // We use a `nullptr` in `PendingVariables` as a tombstone to mark the
      // fact that the variable has already been assigned
      if (not Pending) {
        continue;
      }

      const llvm::SmallSet<const ASTNode *, 4> &UsageASTNodes = VariableUsages
                                                                  .at(Pending);

      // If, from the current node, we dominate all the `ASTNode`s in
      // `UsageASTNodes`, we can mark the variable to be emitted in the current
      // `ASTNode`
      const auto TheNodeDominatesASTUse = [&](const ASTNode *UsageASTNode) {
        Node *UsageGraphNode = ScopeReachabilityGraph.ASTToNodeMap
                                 .at(UsageASTNode);
        return DT.dominates(TheNode, UsageGraphNode);
      };
      bool AllUsesDominated = llvm::all_of(UsageASTNodes,
                                           TheNodeDominatesASTUse);

      // If we the current TheNode dominates all the uses of the `Variable`,
      // we mark the current node as the place where to emit the variable
      // declaration
      if (AllUsesDominated) {
        Result[TheNode->getASTNode()].insert(Pending);
        Pending = nullptr;
      }
    }
  }

  // At the end of the processing, we should have assigned all the pending
  // variables
  const auto IsNullPtr = [](const llvm::CallInst *Variable) {
    return not Variable;
  };
  revng_assert(llvm::all_of(PendingVariables, IsNullPtr));

  return Result;
}
