/// \file CDecompilerBeautify.cpp
/// \brief Bautify passes on the final AST
///

//
// Copyright rev.ng Srls. See LICENSE.md for details.
//

#include "llvm/IR/Instructions.h"
#include "llvm/Support/Casting.h"

#include "revng/Support/Assert.h"
#include "revng/Support/Debug.h"

#include "revng-c/RestructureCFGPass/ASTTree.h"
#include "revng-c/RestructureCFGPass/ExprNode.h"
#include "revng-c/RestructureCFGPass/GenerateAst.h"
#include "revng-c/RestructureCFGPass/RegionCFGTree.h"

#include "CDecompilerBeautify.h"

#include "MarkForSerialization.h"

static Logger<> BeautifyLogger("beautify");

using namespace llvm;
using std::make_unique;
using std::unique_ptr;

// Metrics counter variables
unsigned ShortCircuitCounter = 0;
unsigned TrivialShortCircuitCounter = 0;

using UniqueExpr = ASTTree::expr_unique_ptr;

static void flipEmptyThen(ASTNode *RootNode, ASTTree &AST) {
  if (auto *Sequence = llvm::dyn_cast<SequenceNode>(RootNode)) {
    for (ASTNode *Node : Sequence->nodes()) {
      flipEmptyThen(Node, AST);
    }
  } else if (auto *If = llvm::dyn_cast<IfNode>(RootNode)) {
    if (!If->hasThen()) {
      if (BeautifyLogger.isEnabled()) {
        BeautifyLogger << "Flipping then and else branches for : ";
        BeautifyLogger << If->getName() << "\n";
      }
      If->setThen(If->getElse());
      If->setElse(nullptr);

      // Invert the conditional expression of the current `IfNode`.
      UniqueExpr Not;
      Not.reset(new NotNode(If->getCondExpr()));
      ExprNode *NotNode = AST.addCondExpr(std::move(Not));
      If->replaceCondExpr(NotNode);

      flipEmptyThen(If->getThen(), AST);
    } else {

      // We are sure to have the `then` branch since the previous check did
      // not verify
      flipEmptyThen(If->getThen(), AST);

      // We have not the same assurance for the `else` branch
      if (If->hasElse()) {
        flipEmptyThen(If->getElse(), AST);
      }
    }
  } else if (auto *Scs = llvm::dyn_cast<ScsNode>(RootNode)) {
    flipEmptyThen(Scs->getBody(), AST);
  } else if (auto *Switch = llvm::dyn_cast<SwitchNode>(RootNode)) {

    for (auto &LabelCasePair : Switch->cases())
      flipEmptyThen(LabelCasePair.second, AST);

    if (ASTNode *Default = Switch->getDefault())
      flipEmptyThen(Default, AST);
  }
}

using Marker = MarkForSerialization::Analysis;

static bool requiresNoStatement(IfNode *If, Marker &Mark) {
  // HACK: this is not correct, because it enables short-circuit even in cases
  // where the IfNode really does require some statements, hence producing
  // code that is not semantically equivalent
  return true;

  // Compute how many statement we need to serialize for the basicblock
  // associated with the internal `IfNode`.
  ExprNode *ExprBB = If->getCondExpr();
  if (auto *Atomic = llvm::dyn_cast<AtomicNode>(ExprBB)) {
    llvm::BasicBlock *BB = Atomic->getConditionalBasicBlock();
    if (Mark.getToSerialize(BB).size() == 0) {
      return true;
    }
  }
  return false;
}

// Helper function to simplify short-circuit IFs
static void
simplifyShortCircuit(ASTNode *RootNode, ASTTree &AST, Marker &Mark) {

  if (auto *Sequence = llvm::dyn_cast<SequenceNode>(RootNode)) {
    for (ASTNode *Node : Sequence->nodes()) {
      simplifyShortCircuit(Node, AST, Mark);
    }

  } else if (auto *Scs = llvm::dyn_cast<ScsNode>(RootNode)) {
    simplifyShortCircuit(Scs->getBody(), AST, Mark);
  } else if (auto *Switch = llvm::dyn_cast<SwitchNode>(RootNode)) {

    for (auto &LabelCasePair : Switch->cases())
      simplifyShortCircuit(LabelCasePair.second, AST, Mark);
    if (ASTNode *Default = Switch->getDefault())
      simplifyShortCircuit(Default, AST, Mark);

  } else if (auto *If = llvm::dyn_cast<IfNode>(RootNode)) {
    if (If->hasBothBranches()) {
      if (auto NestedIf = llvm::dyn_cast_or_null<IfNode>(If->getThen())) {

        // TODO: Refactor this with some kind of iterator
        if (NestedIf->getThen() != nullptr) {

          if (If->getElse()->isEqual(NestedIf->getThen())
              and requiresNoStatement(NestedIf, Mark)) {
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
            simplifyShortCircuit(If, AST, Mark);
          }
        }

        if (NestedIf->getElse() != nullptr) {
          if (If->getElse()->isEqual(NestedIf->getElse())
              and requiresNoStatement(NestedIf, Mark)) {
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

            simplifyShortCircuit(If, AST, Mark);
          }
        }
      }
    }
    if (If->hasBothBranches()) {
      if (auto NestedIf = llvm::dyn_cast_or_null<IfNode>(If->getElse())) {
        // TODO: Refactor this with some kind of iterator
        if (NestedIf->getThen() != nullptr) {
          if (If->getThen()->isEqual(NestedIf->getThen())
              and requiresNoStatement(NestedIf, Mark)) {
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
            AST.addCondExpr(std::move(NotAAndNotB));

            // Increment counter
            ShortCircuitCounter += 1;

            simplifyShortCircuit(If, AST, Mark);
          }
        }

        if (NestedIf->getElse() != nullptr) {
          if (If->getThen()->isEqual(NestedIf->getElse())
              and requiresNoStatement(NestedIf, Mark)) {
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
            AST.addCondExpr(std::move(NotAAndB));

            // Increment counter
            ShortCircuitCounter += 1;

            simplifyShortCircuit(If, AST, Mark);
          }
        }
      }
    }
  }
}

static void
simplifyTrivialShortCircuit(ASTNode *RootNode, ASTTree &AST, Marker &Mark) {
  if (auto *Sequence = llvm::dyn_cast<SequenceNode>(RootNode)) {
    for (ASTNode *Node : Sequence->nodes()) {
      simplifyTrivialShortCircuit(Node, AST, Mark);
    }
  } else if (auto *Scs = llvm::dyn_cast<ScsNode>(RootNode)) {
    simplifyTrivialShortCircuit(Scs->getBody(), AST, Mark);

  } else if (auto *Switch = llvm::dyn_cast<SwitchNode>(RootNode)) {

    for (auto &LabelCasePair : Switch->cases())
      simplifyTrivialShortCircuit(LabelCasePair.second, AST, Mark);
    if (ASTNode *Default = Switch->getDefault())
      simplifyTrivialShortCircuit(Default, AST, Mark);

  } else if (auto *If = llvm::dyn_cast<IfNode>(RootNode)) {
    if (!If->hasElse()) {
      if (auto *InternalIf = llvm::dyn_cast<IfNode>(If->getThen())) {
        if (!InternalIf->hasElse() and requiresNoStatement(InternalIf, Mark)) {
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

          simplifyTrivialShortCircuit(RootNode, AST, Mark);
        }
      }
    }

    if (If->hasThen())
      simplifyTrivialShortCircuit(If->getThen(), AST, Mark);
    if (If->hasElse())
      simplifyTrivialShortCircuit(If->getElse(), AST, Mark);
  }
}

static ASTNode *matchSwitch(ASTTree &AST, ASTNode *RootNode, Marker &Mark) {

  // Inspect all the nodes composing a sequence node.
  if (auto *Sequence = llvm::dyn_cast<SequenceNode>(RootNode)) {
    for (ASTNode *&Node : Sequence->nodes()) {
      Node = matchSwitch(AST, Node, Mark);
    }
  } else if (auto *Scs = llvm::dyn_cast<ScsNode>(RootNode)) {
    // Inspect the body of a SCS region.
    Scs->setBody(matchSwitch(AST, Scs->getBody(), Mark));
  } else if (auto *If = llvm::dyn_cast<IfNode>(RootNode)) {

    // Inspect the body of an if construct.
    if (If->hasThen()) {
      If->setThen(matchSwitch(AST, If->getThen(), Mark));
    }
    if (If->hasElse()) {
      If->setElse(matchSwitch(AST, If->getElse(), Mark));
    }
  } else if (auto *Switch = llvm::dyn_cast<SwitchNode>(RootNode)) {
    // TODO: in the current situation, we should not find any switch node
    //       composed by only two case nodes. This check is only a safeguard,
    //       consider removing it altogether.
    // revng_assert(Switch->CaseSize() >= 2);
    for (auto &LabelCasePair : Switch->cases())
      LabelCasePair.second = matchSwitch(AST, LabelCasePair.second, Mark);

    if (ASTNode *Default = Switch->getDefault())
      Default = matchSwitch(AST, Default, Mark);
  }
  return RootNode;
}

static void simplifyLastContinue(ASTTree &AST) {
  for (ASTNode *Node : AST.nodes()) {
    auto *Scs = llvm::dyn_cast<ScsNode>(Node);
    if (not Scs or not Scs->hasBody())
      continue;

    auto *Seq = dyn_cast<SequenceNode>(Scs->getBody());
    if (not Seq)
      continue;

    auto ListSize = Seq->listSize();
    revng_assert(ListSize);
    ASTNode *LastNode = Seq->getNodeN(ListSize - 1);
    if (auto *Continue = llvm::dyn_cast<ContinueNode>(LastNode))
      if (Continue->hasComputation())
        Continue->setImplicit();
  }
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

    if (ASTNode *Default = Switch->getDefault())
      matchDoWhile(Default, AST);

  } else if (auto *Scs = llvm::dyn_cast<ScsNode>(RootNode)) {
    ASTNode *Body = Scs->getBody();

    // Recursive scs nesting handling
    matchDoWhile(Body, AST);

    ASTNode *LastNode = nullptr;
    bool InsideSequence = false;
    if (auto *Seq = llvm::dyn_cast<SequenceNode>(Body)) {
      using SeqSizeType = SequenceNode::links_container::size_type;
      SeqSizeType SequenceSize = Seq->listSize();
      LastNode = Seq->getNodeN(SequenceSize - 1);
      InsideSequence = true;
    } else {
      LastNode = Body;
    }

    revng_assert(LastNode != nullptr);

    if (auto *NestedIf = llvm::dyn_cast<IfNode>(LastNode)) {

      // Only if nodes with both branches are candidates.
      if (NestedIf->hasBothBranches()) {
        ASTNode *Then = NestedIf->getThen();
        ASTNode *Else = NestedIf->getElse();

        if (llvm::isa<BreakNode>(Then) and llvm::isa<ContinueNode>(Else)) {

          Scs->setDoWhile(NestedIf);
          // Invert the conditional expression of the current `IfNode`.
          UniqueExpr Not;
          Not.reset(new NotNode(NestedIf->getCondExpr()));
          ExprNode *NotNode = AST.addCondExpr(std::move(Not));
          NestedIf->replaceCondExpr(NotNode);

          // Remove the if node
          if (InsideSequence) {
            cast<SequenceNode>(Body)->removeNode(NestedIf);
          } else {
            Scs->setBody(nullptr);
          }
        } else if (llvm::isa<BreakNode>(Else)
                   and llvm::isa<ContinueNode>(Then)) {
          Scs->setDoWhile(NestedIf);

          // Remove the if node
          if (InsideSequence) {
            cast<SequenceNode>(Body)->removeNode(NestedIf);
          } else {
            Scs->setBody(nullptr);
          }
        }
      }
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
    if (ASTNode *Default = Switch->getDefault())
      addComputationToContinue(Default, ConditionIf);

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
    if (ASTNode *Default = Switch->getDefault())
      matchWhile(Default, AST);

  } else if (auto *Scs = llvm::dyn_cast<ScsNode>(RootNode)) {
    ASTNode *Body = Scs->getBody();

    // Body could be nullptr (previous while/dowhile semplification)
    if (Body == nullptr) {
      return;
    }

    // Recursive scs nesting handling
    matchWhile(Body, AST);

    ASTNode *FirstNode = nullptr;
    bool InsideSequence = false;
    if (auto *Seq = llvm::dyn_cast<SequenceNode>(Body)) {
      FirstNode = Seq->getNodeN(0);
      InsideSequence = true;
    } else {
      FirstNode = Body;
    }

    revng_assert(FirstNode != nullptr);

    if (auto *NestedIf = llvm::dyn_cast<IfNode>(FirstNode)) {

      // Only if nodes with both branches are candidates.
      if (NestedIf->hasBothBranches()) {
        ASTNode *Then = NestedIf->getThen();
        ASTNode *Else = NestedIf->getElse();

        if (llvm::isa<BreakNode>(Then)) {

          Scs->setWhile(NestedIf);
          // Invert the conditional expression of the current `IfNode`.
          UniqueExpr Not;
          Not.reset(new NotNode(NestedIf->getCondExpr()));
          ExprNode *NotNode = AST.addCondExpr(std::move(Not));
          NestedIf->replaceCondExpr(NotNode);

          // Remove the if node
          if (InsideSequence) {
            cast<SequenceNode>(Body)->removeNode(NestedIf);
          } else {
            Scs->setBody(NestedIf->getElse());

            // Add computation before the continue nodes
            addComputationToContinue(Scs->getBody(), NestedIf);
          }
        } else if (llvm::isa<BreakNode>(Else)) {
          Scs->setWhile(NestedIf);

          // Remove the if node
          if (InsideSequence) {
            cast<SequenceNode>(Body)->removeNode(NestedIf);
          } else {
            Scs->setBody(NestedIf->getThen());

            // Add computation before the continue nodes
            addComputationToContinue(Scs->getBody(), NestedIf);
          }
        }
      }
    }
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
      if (ASTNode *Default = Switch->getDefault())
        exec(Default, AST);
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

void beautifyAST(Function &F, ASTTree &CombedAST, Marker &Mark) {

  ASTNode *RootNode = CombedAST.getRoot();

  if (BeautifyLogger.isEnabled()) {
    CombedAST.dumpASTOnFile(F.getName(), "ast", "Before-beautify");
  }

  // Simplify short-circuit nodes.
  revng_log(BeautifyLogger, "Performing short-circuit simplification\n");
  simplifyShortCircuit(RootNode, CombedAST, Mark);
  if (BeautifyLogger.isEnabled()) {
    CombedAST.dumpASTOnFile(F.getName(), "ast", "After-short-circuit");
  }

  // Flip IFs with empty then branches.
  // We need to do it before simplifyTrivialShortCircuit, otherwise that
  // functions will need to check every possile combination of then-else to
  // simplify. In this way we can keep it simple.
  revng_log(BeautifyLogger,
            "Performing IFs with empty then branches flipping\n");
  flipEmptyThen(RootNode, CombedAST);
  if (BeautifyLogger.isEnabled()) {
    CombedAST.dumpASTOnFile(F.getName(), "ast", "After-if-flip");
  }

  // Simplify trivial short-circuit nodes.
  revng_log(BeautifyLogger,
            "Performing trivial short-circuit simplification\n");
  simplifyTrivialShortCircuit(RootNode, CombedAST, Mark);
  if (BeautifyLogger.isEnabled()) {
    CombedAST.dumpASTOnFile(F.getName(), "ast", "After-trivial-short-circuit");
  }

  // Flip IFs with empty then branches.
  // We need to do it here again, after simplifyTrivialShortCircuit, because
  // that functions can create empty then branches in some situations, and we
  // want to flip them as well.
  revng_log(BeautifyLogger,
            "Performing IFs with empty then branches flipping\n");
  flipEmptyThen(RootNode, CombedAST);
  if (BeautifyLogger.isEnabled()) {
    CombedAST.dumpASTOnFile(F.getName(), "ast", "After-if-flip");
  }

  // Match switch node.
  revng_log(BeautifyLogger, "Performing switch nodes matching\n");
  RootNode = matchSwitch(CombedAST, RootNode, Mark);
  if (BeautifyLogger.isEnabled()) {
    CombedAST.dumpASTOnFile(F.getName(), "ast", "After-switch-match");
  }

  // Match dowhile.
  revng_log(BeautifyLogger, "Matching do-while\n");
  matchDoWhile(RootNode, CombedAST);
  if (BeautifyLogger.isEnabled()) {
    CombedAST.dumpASTOnFile(F.getName(), "ast", "After-match-do-while");
  }

  // Match while.
  revng_log(BeautifyLogger, "Matching while\n");
  matchWhile(RootNode, CombedAST);
  if (BeautifyLogger.isEnabled()) {
    CombedAST.dumpASTOnFile(F.getName(), "ast", "After-match-while");
  }

  // Remove useless continues.
  revng_log(BeautifyLogger, "Removing useless continue nodes\n");
  simplifyLastContinue(CombedAST);
  if (BeautifyLogger.isEnabled()) {
    CombedAST.dumpASTOnFile(F.getName(), "ast", "After-continue-removal");
  }

  // Fix loop breaks from within switches
  revng_log(BeautifyLogger, "Fixing loop breaks inside switches\n");
  SwitchBreaksFixer().run(RootNode, CombedAST);
  if (BeautifyLogger.isEnabled())
    CombedAST.dumpASTOnFile(F.getName(), "ast", "After-fix-switch-breaks");

  // Remove empty sequences.
  revng_log(BeautifyLogger, "Removing emtpy sequence nodes\n");
  simplifyAtomicSequence(CombedAST, RootNode);
  if (BeautifyLogger.isEnabled()) {
    CombedAST.dumpASTOnFile(F.getName(),
                            "ast",
                            "After-removal-empty-sequences");
  }
}
