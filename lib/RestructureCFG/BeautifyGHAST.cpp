/// \file CDecompilerBeautify.cpp
/// Beautify passes on the final AST
///

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/Instructions.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Path.h"

#include "revng/ADT/RecursiveCoroutine.h"
#include "revng/Model/IRHelpers.h"
#include "revng/Model/LoadModelPass.h"
#include "revng/Pipeline/RegisterAnalysis.h"
#include "revng/RestructureCFG/ASTNodeUtils.h"
#include "revng/RestructureCFG/ASTTree.h"
#include "revng/RestructureCFG/BeautifyGHAST.h"
#include "revng/RestructureCFG/ExprNode.h"
#include "revng/RestructureCFG/GenerateAst.h"
#include "revng/RestructureCFG/RegionCFGTree.h"
#include "revng/Support/Assert.h"
#include "revng/Support/Debug.h"
#include "revng/Support/DecompilationHelpers.h"

#include "FallThroughScopeAnalysis.h"
#include "InlineDispatcherSwitch.h"
#include "PromoteCallNoReturn.h"
#include "RemoveDeadCode.h"
#include "SimplifyCompareNode.h"
#include "SimplifyDualSwitch.h"
#include "SimplifyHybridNot.h"
#include "SimplifyImplicitStatement.h"

using std::unique_ptr;

using namespace llvm;

static Logger<> BeautifyLogger("beautify");

// Prefix for the short circuit metrics dir.
static cl::opt<std::string> OutputPath("short-circuit-metrics-output-dir",
                                       cl::desc("Short circuit metrics dir"),
                                       cl::value_desc("short-circuit-dir"),
                                       cl::cat(MainCategory),
                                       cl::Optional);

static std::unique_ptr<llvm::raw_fd_ostream>
openFunctionFile(const StringRef DirectoryPath,
                 const StringRef FunctionName,
                 const StringRef Suffix) {

  std::error_code Error;
  SmallString<32> FilePath = DirectoryPath;

  if (FilePath.empty())
    if ((Error = llvm::sys::fs::current_path(FilePath)))
      revng_abort(Error.message().c_str());

  if ((Error = llvm::sys::fs::make_absolute(FilePath)))
    revng_abort(Error.message().c_str());

  if ((Error = llvm::sys::fs::create_directories(FilePath)))
    revng_abort(Error.message().c_str());

  llvm::sys::path::append(FilePath, FunctionName + Suffix);
  auto FileOStream = std::make_unique<llvm::raw_fd_ostream>(FilePath, Error);
  if (Error) {
    FileOStream.reset();
    revng_abort(Error.message().c_str());
  }

  return FileOStream;
}

// Metrics counter variables
static unsigned ShortCircuitCounter = 0;
static unsigned TrivialShortCircuitCounter = 0;

static RecursiveCoroutine<bool> hasSideEffects(ExprNode *Expr) {
  switch (Expr->getKind()) {

  case ExprNode::NodeKind::NK_Atomic: {
    auto *Atomic = llvm::cast<AtomicNode>(Expr);
    llvm::BasicBlock *BB = Atomic->getConditionalBasicBlock();
    for (llvm::Instruction &I : *BB) {

      if (I.getType()->isVoidTy() and hasSideEffects(I)) {
        // For Instructions with void type, AddLocalVariablesDueToSideEffects
        // cannot properly assign them to LocalVariables because they have
        // void type, so we need to explicitly ask if they have side effects.
        rc_return true;
      } else {
        revng_assert(not isCallToTagged(&I, FunctionTags::Assign),
                     "call to assign should have matched void+hasSideEffects");
      }
    }
    rc_return false;
  }

  case ExprNode::NodeKind::NK_Not: {
    auto *Not = llvm::cast<NotNode>(Expr);
    rc_return rc_recur hasSideEffects(Not->getNegatedNode());
  }

  case ExprNode::NodeKind::NK_And: {
    auto *And = llvm::cast<AndNode>(Expr);
    auto &&[LHS, RHS] = And->getInternalNodes();
    rc_return rc_recur hasSideEffects(LHS) or rc_recur hasSideEffects(RHS);
  }

  case ExprNode::NodeKind::NK_Or: {
    auto *Or = llvm::cast<OrNode>(Expr);
    auto &&[LHS, RHS] = Or->getInternalNodes();
    rc_return rc_recur hasSideEffects(LHS) or rc_recur hasSideEffects(RHS);
  }

  default:
    revng_abort();
  }
  rc_return true;
}

static bool hasSideEffects(IfNode *If) {
  // Compute how many statement we need to serialize for the basicblock
  // associated with the internal `IfNode`.
  return hasSideEffects(If->getCondExpr());
}

using UniqueExpr = ASTTree::expr_unique_ptr;

// Helper function to simplify short-circuit IFs
static bool simplifyShortCircuit(ASTNode *RootNode, ASTTree &AST) {

  // The following should be an assert, but since the backend is in
  // maintenance mode, we have an early return to propagate an early
  // failure.
  if (RootNode == nullptr) {
    return false;
  }

  if (auto *Sequence = llvm::dyn_cast<SequenceNode>(RootNode)) {
    for (ASTNode *Node : Sequence->nodes()) {
      return simplifyShortCircuit(Node, AST);
    }

  } else if (auto *Scs = llvm::dyn_cast<ScsNode>(RootNode)) {
    return simplifyShortCircuit(Scs->getBody(), AST);
  } else if (auto *Switch = llvm::dyn_cast<SwitchNode>(RootNode)) {

    for (auto &LabelCasePair : Switch->cases())
      return simplifyShortCircuit(LabelCasePair.second, AST);

  } else if (auto *If = llvm::dyn_cast<IfNode>(RootNode)) {
    if (If->hasBothBranches()) {
      if (auto NestedIf = llvm::dyn_cast_or_null<IfNode>(If->getThen())) {

        // TODO: Refactor this with some kind of iterator
        if (NestedIf->getThen() != nullptr) {

          if (If->getElse()->isEqual(NestedIf->getThen())
              and not hasSideEffects(NestedIf)) {
            if (BeautifyLogger.isEnabled()) {
              BeautifyLogger << "Candidate for short-circuit reduction found:";
              BeautifyLogger << "\n";
              BeautifyLogger << "IF " << If->getName() << " and ";
              BeautifyLogger << "IF " << NestedIf->getName() << "\n";
              BeautifyLogger << "Nodes being simplified:\n";
              BeautifyLogger << If->getElse()->getName() << " and ";
              BeautifyLogger << NestedIf->getThen()->getName() << "\n";
            }
            If->setThen(NestedIf->getElse());
            If->setElse(NestedIf->getThen());

            // `if A and not B` situation.
            UniqueExpr NotB;
            NotB.reset(new NotNode(NestedIf->getCondExpr()));
            ExprNode *NotBNode = AST.addCondExpr(std::move(NotB));

            UniqueExpr AAndNotB;
            AAndNotB.reset(new AndNode(If->getCondExpr(), NotBNode));

            ExprNode *AAndNotBNode = AST.addCondExpr(std::move(AAndNotB));

            If->replaceCondExpr(AAndNotBNode);

            // Increment counter
            ShortCircuitCounter += 1;

            // Recursive call.
            return simplifyShortCircuit(If, AST);
          }
        }

        if (NestedIf->getElse() != nullptr) {
          if (If->getElse()->isEqual(NestedIf->getElse())
              and not hasSideEffects(NestedIf)) {
            if (BeautifyLogger.isEnabled()) {
              BeautifyLogger << "Candidate for short-circuit reduction found:";
              BeautifyLogger << "\n";
              BeautifyLogger << "IF " << If->getName() << " and ";
              BeautifyLogger << "IF " << NestedIf->getName() << "\n";
              BeautifyLogger << "Nodes being simplified:\n";
              BeautifyLogger << If->getElse()->getName() << " and ";
              BeautifyLogger << NestedIf->getElse()->getName() << "\n";
            }
            If->setThen(NestedIf->getThen());
            If->setElse(NestedIf->getElse());

            // `if A and B` situation.
            UniqueExpr AAndB;
            {
              ExprNode *E = new AndNode(If->getCondExpr(),
                                        NestedIf->getCondExpr());
              AAndB.reset(E);
            }

            ExprNode *AAndBNode = AST.addCondExpr(std::move(AAndB));

            If->replaceCondExpr(AAndBNode);

            // Increment counter
            ShortCircuitCounter += 1;

            return simplifyShortCircuit(If, AST);
          }
        }
      }
    }
    if (If->hasBothBranches()) {
      if (auto NestedIf = llvm::dyn_cast_or_null<IfNode>(If->getElse())) {
        // TODO: Refactor this with some kind of iterator
        if (NestedIf->getThen() != nullptr) {
          if (If->getThen()->isEqual(NestedIf->getThen())
              and not hasSideEffects(NestedIf)) {
            if (BeautifyLogger.isEnabled()) {
              BeautifyLogger << "Candidate for short-circuit reduction found:";
              BeautifyLogger << "\n";
              BeautifyLogger << "IF " << If->getName() << " and ";
              BeautifyLogger << "IF " << NestedIf->getName() << "\n";
              BeautifyLogger << "Nodes being simplified:\n";
              BeautifyLogger << If->getThen()->getName() << " and ";
              BeautifyLogger << NestedIf->getThen()->getName() << "\n";
            }
            If->setElse(NestedIf->getElse());
            If->setThen(NestedIf->getThen());

            // `if not A and not B` situation.
            UniqueExpr NotA;
            NotA.reset(new NotNode(If->getCondExpr()));
            ExprNode *NotANode = AST.addCondExpr(std::move(NotA));

            UniqueExpr NotB;
            NotB.reset(new NotNode(NestedIf->getCondExpr()));
            ExprNode *NotBNode = AST.addCondExpr(std::move(NotB));

            UniqueExpr NotAAndNotB;
            NotAAndNotB.reset(new AndNode(NotANode, NotBNode));
            ExprNode *NotAAndNotBNode = AST.addCondExpr(std::move(NotAAndNotB));

            If->replaceCondExpr(NotAAndNotBNode);

            // Increment counter
            ShortCircuitCounter += 1;

            return simplifyShortCircuit(If, AST);
          }
        }

        if (NestedIf->getElse() != nullptr) {
          if (If->getThen()->isEqual(NestedIf->getElse())
              and not hasSideEffects(NestedIf)) {
            if (BeautifyLogger.isEnabled()) {
              BeautifyLogger << "Candidate for short-circuit reduction found:";
              BeautifyLogger << "\n";
              BeautifyLogger << "IF " << If->getName() << " and ";
              BeautifyLogger << "IF " << NestedIf->getName() << "\n";
              BeautifyLogger << "Nodes being simplified:\n";
              BeautifyLogger << If->getThen()->getName() << " and ";
              BeautifyLogger << NestedIf->getElse()->getName() << "\n";
            }
            If->setElse(NestedIf->getThen());
            If->setThen(NestedIf->getElse());

            // `if not A and B` situation.
            UniqueExpr NotA;
            NotA.reset(new NotNode(If->getCondExpr()));
            ExprNode *NotANode = AST.addCondExpr(std::move(NotA));

            UniqueExpr NotAAndB;
            NotAAndB.reset(new AndNode(NotANode, NestedIf->getCondExpr()));
            ExprNode *NotAAndBNode = AST.addCondExpr(std::move(NotAAndB));

            If->replaceCondExpr(NotAAndBNode);

            // Increment counter
            ShortCircuitCounter += 1;

            return simplifyShortCircuit(If, AST);
          }
        }
      }
    }

    if (If->hasThen())
      return simplifyShortCircuit(If->getThen(), AST);
    if (If->hasElse())
      return simplifyShortCircuit(If->getElse(), AST);
  }

  // We return true to notify that no `simplifyShortCircuit` failure arose
  return true;
}

static bool simplifyTrivialShortCircuit(ASTNode *RootNode, ASTTree &AST) {

  // The following should be an assert, but since the backend is in
  // maintenance mode, we have an early return to propagate an early
  // failure.
  if (RootNode == nullptr) {
    return false;
  }

  if (auto *Sequence = llvm::dyn_cast<SequenceNode>(RootNode)) {
    for (ASTNode *Node : Sequence->nodes()) {
      return simplifyTrivialShortCircuit(Node, AST);
    }
  } else if (auto *Scs = llvm::dyn_cast<ScsNode>(RootNode)) {
    return simplifyTrivialShortCircuit(Scs->getBody(), AST);

  } else if (auto *Switch = llvm::dyn_cast<SwitchNode>(RootNode)) {

    for (auto &LabelCasePair : Switch->cases())
      return simplifyTrivialShortCircuit(LabelCasePair.second, AST);

  } else if (auto *If = llvm::dyn_cast<IfNode>(RootNode)) {
    if (!If->hasElse()) {
      if (auto *InternalIf = llvm::dyn_cast<IfNode>(If->getThen())) {
        if (!InternalIf->hasElse() and not hasSideEffects(InternalIf)) {
          if (BeautifyLogger.isEnabled()) {
            BeautifyLogger << "Candidate for trivial short-circuit reduction";
            BeautifyLogger << "found:\n";
            BeautifyLogger << "IF " << If->getName() << " and ";
            BeautifyLogger << "If " << InternalIf->getName() << "\n";
            BeautifyLogger << "Nodes being simplified:\n";
            BeautifyLogger << If->getThen()->getName() << " and ";
            BeautifyLogger << InternalIf->getThen()->getName() << "\n";
          }
          If->setThen(InternalIf->getThen());

          // `if A and B` situation.
          UniqueExpr AAndB;
          {
            ExprNode *E = new AndNode(If->getCondExpr(),
                                      InternalIf->getCondExpr());
            AAndB.reset(E);
          }
          ExprNode *AAndBNode = AST.addCondExpr(std::move(AAndB));

          If->replaceCondExpr(AAndBNode);

          // Increment counter
          TrivialShortCircuitCounter += 1;

          return simplifyTrivialShortCircuit(RootNode, AST);
        }
      }
    }

    if (If->hasThen())
      return simplifyTrivialShortCircuit(If->getThen(), AST);
    if (If->hasElse())
      return simplifyTrivialShortCircuit(If->getElse(), AST);
  }

  // We return true to notify that no `simplifyShortCircuit` failure arose
  return true;
}

static void matchDoWhile(ASTNode *RootNode, ASTTree &AST) {

  BeautifyLogger << "Matching do whiles"
                 << "\n";
  if (auto *Sequence = llvm::dyn_cast<SequenceNode>(RootNode)) {
    for (ASTNode *Node : Sequence->nodes()) {
      matchDoWhile(Node, AST);
    }
  } else if (auto *If = llvm::dyn_cast<IfNode>(RootNode)) {
    if (If->hasThen()) {
      matchDoWhile(If->getThen(), AST);
    }
    if (If->hasElse()) {
      matchDoWhile(If->getElse(), AST);
    }

  } else if (auto *Switch = llvm::dyn_cast<SwitchNode>(RootNode)) {

    for (auto &LabelCasePair : Switch->cases())
      matchDoWhile(LabelCasePair.second, AST);

  } else if (auto *Scs = llvm::dyn_cast<ScsNode>(RootNode)) {
    ASTNode *Body = Scs->getBody();

    // Body could be nullptr (previous while/dowhile semplification)
    if (Body == nullptr)
      return;

    // Recursive scs nesting handling
    matchDoWhile(Body, AST);

    // We don't want to transform a do-while in a while
    if (Scs->isWhile())
      return;

    ASTNode *LastNode = Body;
    auto *SequenceBody = llvm::dyn_cast<SequenceNode>(Body);
    if (SequenceBody) {
      revng_assert(not SequenceBody->nodes().empty());
      LastNode = *std::prev(SequenceBody->nodes().end());
    }
    revng_assert(LastNode);

    auto *NestedIf = llvm::dyn_cast<IfNode>(LastNode);
    if (not NestedIf)
      return;

    ASTNode *Then = NestedIf->getThen();
    ASTNode *Else = NestedIf->getElse();
    auto *ThenBreak = llvm::dyn_cast_or_null<BreakNode>(Then);
    auto *ElseBreak = llvm::dyn_cast_or_null<BreakNode>(Else);
    auto *ThenContinue = llvm::dyn_cast_or_null<ContinueNode>(Then);
    auto *ElseContinue = llvm::dyn_cast_or_null<ContinueNode>(Else);

    bool HandledCases = (ThenBreak and ElseContinue)
                        or (ThenContinue and ElseBreak);
    if (not HandledCases)
      return;

    Scs->setDoWhile(NestedIf);

    if (ThenBreak and ElseContinue) {
      // Invert the conditional expression of the current `IfNode`.
      UniqueExpr Not;
      Not.reset(new NotNode(NestedIf->getCondExpr()));
      ExprNode *NotNode = AST.addCondExpr(std::move(Not));
      NestedIf->replaceCondExpr(NotNode);

    } else {
      revng_assert(ElseBreak and ThenContinue);
    }

    // Remove the if node
    if (SequenceBody) {
      SequenceBody->removeNode(NestedIf);
    } else {
      Scs->setBody(nullptr);
    }

  } else {
    BeautifyLogger << "No matching done\n";
  }
}

static void addComputationToContinue(ASTNode *RootNode, IfNode *ConditionIf) {
  BeautifyLogger << "Adding computation code to continue node"
                 << "\n";
  if (auto *Sequence = llvm::dyn_cast<SequenceNode>(RootNode)) {
    for (ASTNode *Node : Sequence->nodes()) {
      addComputationToContinue(Node, ConditionIf);
    }
  } else if (auto *If = llvm::dyn_cast<IfNode>(RootNode)) {
    if (If->hasThen()) {
      addComputationToContinue(If->getThen(), ConditionIf);
    }
    if (If->hasElse()) {
      addComputationToContinue(If->getElse(), ConditionIf);
    }
  } else if (auto *Switch = llvm::dyn_cast<SwitchNode>(RootNode)) {

    for (auto &LabelCasePair : Switch->cases())
      addComputationToContinue(LabelCasePair.second, ConditionIf);

  } else if (auto *Continue = llvm::dyn_cast<ContinueNode>(RootNode)) {
    Continue->addComputationIfNode(ConditionIf);
  }
}

static void matchWhile(ASTNode *RootNode, ASTTree &AST) {
  if (auto *Sequence = llvm::dyn_cast<SequenceNode>(RootNode)) {
    for (ASTNode *Node : Sequence->nodes()) {
      matchWhile(Node, AST);
    }
  } else if (auto *If = llvm::dyn_cast<IfNode>(RootNode)) {
    if (If->hasThen()) {
      matchWhile(If->getThen(), AST);
    }
    if (If->hasElse()) {
      matchWhile(If->getElse(), AST);
    }

  } else if (auto *Switch = llvm::dyn_cast<SwitchNode>(RootNode)) {

    for (auto &LabelCasePair : Switch->cases())
      matchWhile(LabelCasePair.second, AST);

  } else if (auto *Scs = llvm::dyn_cast<ScsNode>(RootNode)) {
    ASTNode *Body = Scs->getBody();

    // Body could be nullptr (previous while/dowhile semplification)
    if (Body == nullptr)
      return;

    // Recursive scs nesting handling
    matchWhile(Body, AST);

    // We don't want to transform a while in a do-while
    if (Scs->isDoWhile())
      return;

    ASTNode *FirstNode = Body;
    auto *SequenceBody = llvm::dyn_cast<SequenceNode>(Body);
    if (SequenceBody) {
      revng_assert(not SequenceBody->nodes().empty());
      FirstNode = *SequenceBody->nodes().begin();
    }
    revng_assert(FirstNode);

    auto *NestedIf = llvm::dyn_cast<IfNode>(FirstNode);
    if (not NestedIf)
      return;

    ASTNode *Then = NestedIf->getThen();
    ASTNode *Else = NestedIf->getElse();
    auto *ThenBreak = llvm::dyn_cast_or_null<BreakNode>(Then);
    auto *ElseBreak = llvm::dyn_cast_or_null<BreakNode>(Else);

    // Without a break, this if cannot become a while
    if (not ThenBreak and not ElseBreak)
      return;

    // This is a while
    Scs->setWhile(NestedIf);

    ASTNode *BranchThatStaysInside = nullptr;
    if (ElseBreak) {
      BranchThatStaysInside = Then;

    } else {
      revng_assert(llvm::isa<BreakNode>(Then));
      BranchThatStaysInside = Else;

      // If the break node is the then branch, we should invert the
      // conditional expression of the current `IfNode`.
      UniqueExpr Not;
      Not.reset(new NotNode(NestedIf->getCondExpr()));
      ExprNode *NotNode = AST.addCondExpr(std::move(Not));
      NestedIf->replaceCondExpr(NotNode);
    }

    // Remove the if node
    if (SequenceBody) {
      SequenceBody->removeNode(NestedIf);
      if (BranchThatStaysInside) {
        auto &Seq = SequenceBody->getChildVec();
        Seq.insert(Seq.begin(), BranchThatStaysInside);
      }
    } else {
      Scs->setBody(BranchThatStaysInside);
    }
    // Add computation before the continue nodes
    addComputationToContinue(Scs->getBody(), NestedIf);
  } else {
    BeautifyLogger << "No matching done\n";
  }
}

class SwitchBreaksFixer {

protected:
  using SwitchStackT = llvm::SmallVector<SwitchNode *, 2>;
  using LoopStackEntryT = std::pair<ScsNode *, SwitchStackT>;
  using LoopStackT = llvm::SmallVector<LoopStackEntryT, 8>;

public:
  SwitchBreaksFixer() = default;
  ~SwitchBreaksFixer() = default;

  void run(ASTNode *RootNode, ASTTree &AST) {
    LoopStack.clear();
    exec(RootNode, AST);
  }

protected:
  void exec(ASTNode *Node, ASTTree &AST) {
    if (Node == nullptr)
      return;
    switch (Node->getKind()) {
    case ASTNode::NK_If: {
      IfNode *If = llvm::cast<IfNode>(Node);
      exec(If->getThen(), AST);
      exec(If->getElse(), AST);
    } break;
    case ASTNode::NK_Scs: {
      ScsNode *Loop = llvm::cast<ScsNode>(Node);
      LoopStack.push_back({ Loop, {} });
      exec(Loop->getBody(), AST);
      revng_assert(LoopStack.back().second.empty());
      LoopStack.pop_back();
    } break;
    case ASTNode::NK_List: {
      SequenceNode *Seq = llvm::cast<SequenceNode>(Node);
      for (ASTNode *N : Seq->nodes())
        exec(N, AST);
    } break;
    case ASTNode::NK_Switch: {
      SwitchNode *Switch = llvm::cast<SwitchNode>(Node);
      if (not LoopStack.empty())
        LoopStack.back().second.push_back(Switch);
      for (auto &LabelCasePair : Switch->cases())
        exec(LabelCasePair.second, AST);
      if (not LoopStack.empty())
        LoopStack.back().second.pop_back();
    } break;
    case ASTNode::NK_Break: {
      revng_assert(not LoopStack.empty()); // assert that we're in a loop
      BreakNode *B = llvm::cast<BreakNode>(Node);
      SwitchStackT &ActiveSwitches = LoopStack.back().second;
      if (not ActiveSwitches.empty()) {
        // The outer switch needs a declaration for the state variable necessary
        // to break directly out of the loop from within the switches
        ActiveSwitches.front()->setNeedsStateVariable(true);
        B->setBreakFromWithinSwitch(true);
        for (SwitchNode *S : LoopStack.back().second) {
          // this loop break is inside one (or possibly more nested) switch(es),
          // contained in the loop, hence all the active switches need a
          // dispatcher to be inserted right after the switch, to use the state
          // variable to dispatch the break out of the loop.
          S->setNeedsLoopBreakDispatcher(true);
        }
      }
    } break;
    case ASTNode::NK_SwitchBreak:
      // assert that we're either not in a loop, or, if we're in a loop we're
      // also inside a switch which is nested in the loop
      revng_assert(LoopStack.empty() or not LoopStack.back().second.empty());
      break;
    case ASTNode::NK_Set:
    case ASTNode::NK_Code:
    case ASTNode::NK_Continue:
      break; // do nothing
    }
  }

protected:
  LoopStackT LoopStack{};
};

// This node weight computation routine uses a reasonable and at the same time
// very basilar criterion, which assign a point for each node in the AST
// subtree. In the future, we might considering using something closer to the
// definition of the cyclomatic Complexity itself, cfr.
// https://www.sonarsource.com/resources/white-papers/cognitive-complexity.html
static RecursiveCoroutine<unsigned>
computeCumulativeNodeWeight(ASTNode *Node,
                            std::map<const ASTNode *, unsigned> &NodeWeight) {
  switch (Node->getKind()) {
  case ASTNode::NK_List: {
    SequenceNode *Seq = llvm::cast<SequenceNode>(Node);

    unsigned Accum = 0;
    for (ASTNode *N : Seq->nodes()) {
      unsigned NWeight = rc_recur computeCumulativeNodeWeight(N, NodeWeight);
      NodeWeight[N] = NWeight;

      // Accumulate the weight of all the nodes in the sequence, in order to
      // compute the weight of the sequence itself.
      Accum += NWeight;
    }
    rc_return Accum;
  }
  case ASTNode::NK_Scs: {
    ScsNode *Loop = llvm::cast<ScsNode>(Node);
    if (Loop->hasBody()) {
      ASTNode *Body = Loop->getBody();
      unsigned BodyWeight = rc_recur computeCumulativeNodeWeight(Body,
                                                                 NodeWeight);
      NodeWeight[Body] = BodyWeight;
      rc_return BodyWeight + 1;
    } else {
      rc_return 1;
    }
  }
  case ASTNode::NK_If: {
    IfNode *If = llvm::cast<IfNode>(Node);

    unsigned ThenWeight = 0;
    unsigned ElseWeight = 0;
    if (If->hasThen()) {
      ASTNode *Then = If->getThen();
      ThenWeight = rc_recur computeCumulativeNodeWeight(Then, NodeWeight);
      NodeWeight[Then] = ThenWeight;
    }
    if (If->hasElse()) {
      ASTNode *Else = If->getElse();
      ElseWeight = rc_recur computeCumulativeNodeWeight(Else, NodeWeight);
      NodeWeight[Else] = ElseWeight;
    }
    rc_return ThenWeight + ElseWeight + 1;
  }
  case ASTNode::NK_Switch: {
    SwitchNode *Switch = llvm::cast<SwitchNode>(Node);

    unsigned SwitchWeight = 0;
    for (auto &LabelCasePair : Switch->cases()) {
      ASTNode *Case = LabelCasePair.second;
      unsigned CaseWeight = rc_recur computeCumulativeNodeWeight(Case,
                                                                 NodeWeight);
      NodeWeight[Case] = CaseWeight;
      SwitchWeight += CaseWeight;
    }
    rc_return SwitchWeight + 1;
  }
  case ASTNode::NK_Code: {

    // TODO: At the moment we use the BasicBlock size to assign a weight to the
    //       code nodes. In future, we would want to use the number of statement
    //       emitted in the decompiled code as weight (and use
    //       `AssignmentMarker`s to do that).
    CodeNode *Code = llvm::cast<CodeNode>(Node);
    llvm::BasicBlock *BB = Code->getBB();
    rc_return BB->size();
  }
  case ASTNode::NK_Continue: {

    // The weight of a continue node, contrary to what intuition would suggest,
    // is not always constant. In fact, due to a previous beautification pass,
    // a continue node could gain a computation node, which represents the code
    // which represents the computations needed to update the condition of the
    // corresponding while/do-while cycle.
    // In this setting, we need to take into account also the weight of this
    // computation node, because that code will become part of the scope ending
    // with the continue. If we do not take into account this contribute, we
    // could end up promoting as fallthrough the break scope, even though its
    // scope is smaller in terms of decompiled code.
    ContinueNode *Continue = llvm::cast<ContinueNode>(Node);
    if (Continue->hasComputation()) {
      IfNode *If = Continue->getComputationIfNode();
      llvm::BasicBlock *BB = If->getOriginalBB();
      revng_assert(BB != nullptr);
      rc_return BB->size() + 1;
    }
  } break;
  case ASTNode::NK_Set:
  case ASTNode::NK_SwitchBreak:
  case ASTNode::NK_Break: {

    // If we assign weight 1 to all these cases, no distinction is needed for
    // them.
    rc_return 1;
  }
  default:
    revng_abort();
  }

  rc_return 0;
}

static RecursiveCoroutine<ASTNode *>
promoteNoFallthrough(ASTTree &AST,
                     ASTNode *Node,
                     FallThroughScopeTypeMap &FallThroughScopeMap,
                     std::map<const ASTNode *, unsigned> &NodeWeight) {
  // Visit the current node.
  switch (Node->getKind()) {
  case ASTNode::NK_List: {
    SequenceNode *Seq = llvm::cast<SequenceNode>(Node);

    // In place of a sequence node, we need just to inspect all the nodes in the
    // sequence.
    for (ASTNode *&N : Seq->nodes()) {
      N = rc_recur promoteNoFallthrough(AST,
                                        N,
                                        FallThroughScopeMap,
                                        NodeWeight);
    }
  } break;
  case ASTNode::NK_Scs: {
    ScsNode *Scs = llvm::cast<ScsNode>(Node);
    if (Scs->hasBody()) {
      ASTNode *Body = Scs->getBody();
      ASTNode *NewBody = rc_recur promoteNoFallthrough(AST,
                                                       Body,
                                                       FallThroughScopeMap,
                                                       NodeWeight);
      Scs->setBody(NewBody);
    }
  } break;
  case ASTNode::NK_If: {
    IfNode *If = llvm::cast<IfNode>(Node);

    // First of all, we recursively invoke the analysis on the children of the
    // `IfNode` (we discussed and said that further simplifications down in
    // the AST do not alter the `nofallthrough property`).
    if (If->hasThen()) {

      // We only have a `then` branch, proceed with the recursive visit.
      ASTNode *Then = If->getThen();
      ASTNode *NewThen = rc_recur promoteNoFallthrough(AST,
                                                       Then,
                                                       FallThroughScopeMap,
                                                       NodeWeight);
      If->setThen(NewThen);
    }
    if (If->hasElse()) {

      // We only have a `else` branch, proceed with the recursive visit.
      ASTNode *Else = If->getElse();
      ASTNode *NewElse = rc_recur promoteNoFallthrough(AST,
                                                       Else,
                                                       FallThroughScopeMap,
                                                       NodeWeight);
      If->setElse(NewElse);
    }

    // Whenever we have both then and else branches, and one of them is
    // no-fallthrough, we try to promote the other to a successor of the if, to
    // reduce nesting.
    if (If->hasThen() and If->hasElse()) {

      // In this case, we need to promote the `else` branch to fallthrough if
      // the `then` branch is a `nofallthrough` scope.
      ASTNode *Then = If->getThen();
      ASTNode *Else = If->getElse();

      // Define two temporary variables which will be used to perform the `then`
      // or `else` promotion.
      bool PromoteThen = false;
      bool PromoteElse = false;
      // First of all, check if both the branches are eligible for promotion.
      if (not fallsThrough(FallThroughScopeMap.at(Then))
          and not fallsThrough(FallThroughScopeMap.at(Else))) {

        if (NodeWeight.at(Then) >= NodeWeight.at(Else)) {
          // If the previous criterion did not match, we use the weight
          // criterion to decide which branch should be promoted
          PromoteThen = true;
        } else {
          PromoteElse = true;
        }
      } else if (not fallsThrough(FallThroughScopeMap.at(Then))) {
        PromoteElse = true;
      } else if (not fallsThrough(FallThroughScopeMap.at(Else))) {
        PromoteThen = true;
      }

      if (PromoteElse) {
        revng_assert(not PromoteThen);
        // The `then` branch is a `nofallthrough` branch.
        // Blank the `else` field, and substitute the current `IfNode` node
        // with the newly created `SequenceNode`.
        If->setElse(nullptr);
        SequenceNode *NewSequence = AST.addSequenceNode();
        NewSequence->addNode(If);

        // We need to assign a state for the `fallthrough` attribute of the
        // newly created `SequenceNode`. We also need to assign the `weight`
        // attribute for the same reason.
        FallThroughScopeMap[NewSequence] = FallThroughScopeMap.at(If);
        NodeWeight[NewSequence] = NodeWeight[If];
        NewSequence->addNode(Else);

        rc_return NewSequence;
      } else if (PromoteThen) {
        revng_assert(not PromoteElse);
        // The `else` branch is a `nofallthrough` branch.
        // Blank the `then` field, and substitute the current `IfNode` node
        // with the newly created `SequenceNode`.
        If->setThen(nullptr);
        SequenceNode *NewSequence = AST.addSequenceNode();
        NewSequence->addNode(If);

        // We need to assign a state for the `fallthrough` attribute of the
        // newly created `SequenceNode`.
        FallThroughScopeMap[NewSequence] = FallThroughScopeMap.at(If);
        NodeWeight[NewSequence] = NodeWeight[If];
        NewSequence->addNode(Then);

        rc_return NewSequence;
      } else {
        revng_assert(not PromoteThen);
        revng_assert(not PromoteElse);
      }
    }
  } break;
  case ASTNode::NK_Switch: {
    auto *Switch = llvm::cast<SwitchNode>(Node);
    for (auto &LabelCasePair : Switch->cases())
      LabelCasePair.second = rc_recur promoteNoFallthrough(AST,
                                                           LabelCasePair.second,
                                                           FallThroughScopeMap,
                                                           NodeWeight);
  } break;
  case ASTNode::NK_Continue: {
    auto *Continue = llvm::cast<ContinueNode>(Node);

    // This transformation changes heavily the structure of the AST, and can
    // invalidate the `implicitContinue` analysis assumptions. Therefore, we
    // check that at this stage no implicit `continue` has been set.
    revng_assert(not Continue->isImplicit());
  } break;
  case ASTNode::NK_Code:
  case ASTNode::NK_Set:
  case ASTNode::NK_SwitchBreak:
  case ASTNode::NK_Break:
    // Do nothing.
    break;
  default:
    revng_unreachable();
  }
  rc_return Node;
}

static ASTNode *promoteNoFallthroughIf(const model::Binary &Model,
                                       ASTNode *RootNode,
                                       ASTTree &AST) {

  // Perform the computation of fallthrough scopes type
  FallThroughScopeTypeMap
    FallThroughScopeMap = computeFallThroughScope(Model, RootNode);

  // In this map, we store the weight of the AST starting from a node and
  // going down.
  std::map<const ASTNode *, unsigned> NodeWeight;

  // Run the analysis which computes the AST weight of the nodes on the tree.
  unsigned RootWeight = computeCumulativeNodeWeight(RootNode, NodeWeight);
  NodeWeight[RootNode] = RootWeight;

  // Run the fallthrough promotion.
  RootNode = promoteNoFallthrough(AST,
                                  RootNode,
                                  FallThroughScopeMap,
                                  NodeWeight);

  // Run the sequence nodes collapse.
  RootNode = collapseSequences(AST, RootNode);

  // Update the root field of the AST.
  AST.setRoot(RootNode);

  return RootNode;
}

bool beautifyAST(const model::Binary &Model, Function &F, ASTTree &CombedAST) {

  // If the --short-circuit-metrics-output-dir=dir argument was passed from
  // command line, we need to print the statistics for the short circuit metrics
  // into a file with the function name, inside the directory 'dir'.
  std::unique_ptr<llvm::raw_fd_ostream> StatsFileStream;
  if (OutputPath.getNumOccurrences())
    StatsFileStream = openFunctionFile(OutputPath, F.getName(), ".csv");

  ShortCircuitCounter = 0;
  TrivialShortCircuitCounter = 0;

  ASTNode *RootNode = CombedAST.getRoot();

  // AST dumper helper
  GHASTDumper Dumper(BeautifyLogger, F, CombedAST, "beautify");

  Dumper.log("before-beautify");

  // Simplify short-circuit nodes.
  revng_log(BeautifyLogger, "Performing short-circuit simplification\n");

  // The following call may return `false` as a signal of failure, and in
  // that case we propagate the error upwards
  if (not(simplifyShortCircuit(RootNode, CombedAST))) {
    return false;
  }
  Dumper.log("after-short-circuit");

  // Flip IFs with empty then branches.
  // We need to do it before simplifyTrivialShortCircuit, otherwise that
  // functions will need to check every possible combination of then-else to
  // simplify. In this way we can keep it simple.
  revng_log(BeautifyLogger,
            "Performing IFs with empty then branches flipping\n");
  flipEmptyThen(CombedAST, RootNode);
  Dumper.log("after-if-flip");

  // Simplify trivial short-circuit nodes.
  revng_log(BeautifyLogger,
            "Performing trivial short-circuit simplification\n");

  // The following call may return `false` as a signal of failure, and in
  // that case we propagate the error upwards
  if (not(simplifyTrivialShortCircuit(RootNode, CombedAST))) {
    return false;
  }
  Dumper.log("after-trivial-short-circuit");

  // Flip IFs with empty then branches.
  // We need to do it here again, after simplifyTrivialShortCircuit, because
  // that functions can create empty then branches in some situations, and we
  // want to flip them as well.
  revng_log(BeautifyLogger,
            "Performing IFs with empty then branches flipping\n");
  flipEmptyThen(CombedAST, RootNode);
  Dumper.log("after-if-flip");

  // Perform the `SwitchBreak` simplification
  revng_log(BeautifyLogger, "Performing SwitchBreak simplification");
  RootNode = simplifySwitchBreak(CombedAST);
  Dumper.log("After-switchbreak-simplify");

  // Perform the dispatcher `switch` inlining
  revng_log(BeautifyLogger, "Performing dispatcher switch inlining\n");
  RootNode = inlineDispatcherSwitch(CombedAST);
  Dumper.log("after-dispatcher-switch-inlining");

  // Perform the dead code simplification.
  // We invoke this pass here because the dispatcher case inlining may have
  // moved around some non local control flow statements like `return`, in such
  // a way that a dead code simplification step is needed.
  revng_log(BeautifyLogger, "Performing dead code simplification\n");
  RootNode = removeDeadCode(Model, CombedAST);
  Dumper.log("after-dead-code-simplify");

  // Perform the simplification of `switch` with two entries in a `if`
  revng_log(BeautifyLogger, "Performing the dual switch simplification\n");
  RootNode = simplifyDualSwitch(CombedAST, RootNode);
  Dumper.log("after-dual-switch-simplify");

  // Remove empty sequences.
  revng_log(BeautifyLogger, "Removing empty sequence nodes\n");
  RootNode = simplifyAtomicSequence(CombedAST, RootNode);
  Dumper.log("after-empty-sequences-removal");

  // Match dowhile.
  revng_log(BeautifyLogger, "Matching do-while\n");
  matchDoWhile(RootNode, CombedAST);
  Dumper.log("after-match-do-while");

  // Match while.
  revng_log(BeautifyLogger, "Matching while\n");
  matchWhile(RootNode, CombedAST);
  Dumper.log("after-match-while");

  // Remove unnecessary scopes under the fallthrough analysis.
  revng_log(BeautifyLogger, "Analyzing fallthrough scopes\n");
  RootNode = promoteNoFallthroughIf(Model, RootNode, CombedAST);
  Dumper.log("after-fallthrough-scope-analysis");

  // Flip IFs with empty then branches.
  // We need to do it here again, after the promotion due to the `nofallthroguh`
  // analysis run before.
  revng_log(BeautifyLogger,
            "Performing IFs with empty then branches flipping\n");
  flipEmptyThen(CombedAST, RootNode);
  Dumper.log("after-if-flip");

  // Run the `promoteCallNoReturn` analysis.
  revng_log(BeautifyLogger, "Perform the CallNoReturn promotion\n");
  RootNode = promoteCallNoReturn(Model, CombedAST, RootNode);
  Dumper.log("after-callnoreturn-promotion");

  // Perform the double `not` simplification (`not` on the GHAST and `not` in
  // the IR).
  revng_log(BeautifyLogger, "Performing the double not simplification\n");
  RootNode = simplifyHybridNot(CombedAST, RootNode);
  Dumper.log("after-double-not-simplify");

  // Perform the `CompareNode` simplification. A `CompareNode` preceded by a
  // `not` is transformed in the `CompareNode` itself with the flipped
  // comparison predicate
  revng_log(BeautifyLogger, "Performing the compare node simplification\n");
  simplifyCompareNode(CombedAST, RootNode);
  Dumper.log("after-compare-node-simplify");

  // Remove useless continues.
  revng_log(BeautifyLogger, "Removing useless continue nodes\n");
  simplifyImplicitContinue(CombedAST);
  Dumper.log("after-continue-removal");

  // Perform the simplification of the implicit `return`, i.e., a `return` of
  // type `void`, which lies on a path followed by no other statements.
  revng_log(BeautifyLogger, "Performing the implicit return simplification\n");
  simplifyImplicitReturn(CombedAST, RootNode);
  Dumper.log("after-implicit-return-simplify");

  // Fix loop breaks from within switches
  revng_log(BeautifyLogger, "Fixing loop breaks inside switches\n");
  SwitchBreaksFixer().run(RootNode, CombedAST);
  Dumper.log("after-fix-switch-breaks");

  // Serialize the collected metrics in the statistics file if necessary
  if (StatsFileStream) {
    *StatsFileStream << "function,short-circuit,trivial-short-circuit\n"
                     << F.getName().data() << "," << ShortCircuitCounter << ","
                     << TrivialShortCircuitCounter << "\n";
  }

  // We return true to notify that not restructuring error arose
  return true;
}
