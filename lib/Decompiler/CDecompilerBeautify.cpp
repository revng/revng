/// \file CDecompilerBeautify.cpp
/// \brief Bautify passes on the final AST
///

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// LLVM includes
#include "llvm/Support/Casting.h"
#include <llvm/IR/Instructions.h>

// revng includes
#include "revng/Support/Debug.h"
#include <revng/Support/Assert.h>
//#include "revng/Support/MonotoneFramework.h"

// local libraries includes
#include "revng-c/RestructureCFGPass/ASTTree.h"
#include "revng-c/RestructureCFGPass/ExprNode.h"
#include "revng-c/RestructureCFGPass/RegionCFGTree.h"

// local includes
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
  }
}

using Marker = MarkForSerialization::Analysis;

#if 0
static bool requiresNoStatement(IfNode *If, Marker &Mark) {
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
#endif

static bool requiresNoStatement(IfNode *, Marker &) {
  // HACK: this is not correct, because it enables short-circuit even in cases
  // where the IfNode really does require some statements, hence producing
  // code that is not semantically equivalent
  return true;
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

    for (auto &Case : Switch->unordered_cases())
      simplifyShortCircuit(Case, AST, Mark);
    if (ASTNode *Default = Switch->getDefault())
      simplifyShortCircuit(Default, AST, Mark);

  } else if (auto *If = llvm::dyn_cast<IfNode>(RootNode)) {
    if (If->hasBothBranches()) {

      if (auto NestedIf = llvm::dyn_cast<IfNode>(If->getThen())) {

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

      if (auto NestedIf = llvm::dyn_cast<IfNode>(If->getElse())) {

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

    for (auto &Case : Switch->unordered_cases())
      simplifyTrivialShortCircuit(Case, AST, Mark);
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

static ConstantInt *getCaseConstant(ASTNode *Node) {
  llvm::BasicBlock *BB = Node->getOriginalBB();
  revng_assert(BB->size() == 2);

  llvm::Instruction &CompareInst = BB->front();
  llvm::Instruction &BranchInst = BB->back();
  revng_assert(llvm::isa<llvm::ICmpInst>(CompareInst));
  revng_assert(llvm::isa<llvm::BranchInst>(BranchInst));
  revng_assert(llvm::cast<llvm::BranchInst>(BranchInst).isConditional());

  auto *Compare = llvm::cast<llvm::ICmpInst>(&CompareInst);
  revng_assert(Compare->getNumOperands() == 2);

  llvm::ConstantInt *CI = llvm::cast<llvm::ConstantInt>(Compare->getOperand(1));
  return CI;
}

static llvm::Value *getCaseValue(ASTNode *Node) {
  llvm::BasicBlock *BB = Node->getOriginalBB();
  revng_assert(BB->size() == 2);

  llvm::Instruction &CompareInst = BB->front();
  llvm::Instruction &BranchInst = BB->back();
  revng_assert(llvm::isa<llvm::ICmpInst>(CompareInst));
  revng_assert(llvm::isa<llvm::BranchInst>(BranchInst));
  revng_assert(llvm::cast<llvm::BranchInst>(BranchInst).isConditional());

  auto *Compare = llvm::cast<llvm::ICmpInst>(&CompareInst);
  revng_assert(Compare->getNumOperands() == 2);

  return Compare->getOperand(0);
}

static bool wasOriginalSwitch(ASTNode *Candidate) {
  llvm::BasicBlock *BB = Candidate->getOriginalBB();

  // Check that the if corresponds to an original basic block.
  if (BB != nullptr) {

    // Check that the body contains an `icmp` instruction over the condition
    // of the switch and a constant, and then a conditional branch.
    if (BB->size() == 2 and BB->getName().startswith("switch check")) {
      llvm::Instruction &First = BB->front();
      llvm::Instruction &Second = BB->back();

      if (llvm::isa<llvm::ICmpInst>(First)) {
        if (auto *Branch = llvm::dyn_cast<llvm::BranchInst>(&Second)) {
          if (Branch->isConditional()) {
            return true;
          }
        }
      }
    }
  }
  return false;
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

    // Switch matching routine.
    if (wasOriginalSwitch(If)) {

      // Get the Value of the condition.
      llvm::Value *SwitchValue = getCaseValue(If);

      // An IfEqualNode represents the start of a switch statement.
      llvm::SmallVector<IfNode *, 8> Candidates;
      Candidates.push_back(If);

      // Continue to accumulate the IfEqual nodes until it is possible.
      while (If->getElse() and wasOriginalSwitch(If->getElse())
             and (SwitchValue == getCaseValue(If->getElse()))) {
        If = llvm::cast<IfNode>(If->getElse());
        Candidates.push_back(If);
      }

      RegularSwitchNode::case_container Cases;
      RegularSwitchNode::case_value_container CaseValues;
      for (IfNode *Candidate : Candidates) {
        Cases.push_back(Candidate->getThen());
        CaseValues.push_back(getCaseConstant(Candidate));
      }

      // Collect the last else (which will become the default case).
      ASTNode *DefaultCase = Candidates.back()->getElse();
      if (DefaultCase == nullptr)
        DefaultCase = AST.addSequenceNode();

      // Create the switch node.
      ASTTree::ast_unique_ptr Switch;
      {
        ASTNode *Tmp = new RegularSwitchNode(SwitchValue,
                                             Cases,
                                             CaseValues,
                                             DefaultCase);
        Switch.reset(Tmp);
      }

      // Invoke the switch matching on the switch just reconstructed.
      matchSwitch(AST, Switch.get(), Mark);

      // Return the new object.
      return AST.addSwitch(std::move(Switch));

    } else {

      // Analyze a standard IfNode.
      if (If->hasThen()) {
        If->setThen(matchSwitch(AST, If->getThen(), Mark));
      }
      if (If->hasElse()) {
        If->setElse(matchSwitch(AST, If->getElse(), Mark));
      }
    }

  } else if (auto *Switch = llvm::dyn_cast<SwitchNode>(RootNode)) {
    for (auto &Case : Switch->unordered_cases())
      Case = matchSwitch(AST, Case, Mark);
    if (ASTNode *Default = Switch->getDefault())
      Default = matchSwitch(AST, Default, Mark);
  }

  return RootNode;
}

static ASTNode *matchDispatcher(ASTTree &AST, ASTNode *RootNode, Marker &Mark) {
  // Inspect all the nodes composing a sequence node.
  if (auto *Sequence = llvm::dyn_cast<SequenceNode>(RootNode)) {
    for (ASTNode *&Node : Sequence->nodes()) {
      Node = matchDispatcher(AST, Node, Mark);
    }

  } else if (auto *Scs = llvm::dyn_cast<ScsNode>(RootNode)) {
    // Inspect the body of a SCS region.
    Scs->setBody(matchDispatcher(AST, Scs->getBody(), Mark));

  } else if (auto *IfDispatcher = llvm::dyn_cast<IfDispatcherNode>(RootNode)) {

    // An `IfDispatcherNode` represents the start of a dispatcher, which we want to
    // represent using a `SwitchDispatcherNode`.
    std::vector<IfDispatcherNode *> Candidates;
    Candidates.push_back(IfDispatcher);

    // Continue to accumulate the `IfDispatcher` nodes until it is possible.
    while (auto *SuccChk = dyn_cast_or_null<IfDispatcherNode>(IfDispatcher->getElse())) {
      Candidates.push_back(SuccChk);
      IfDispatcher = SuccChk;
    }

    SwitchDispatcherNode::case_container Cases;
    SwitchDispatcherNode::case_value_container CaseValues;
    for (IfDispatcherNode *Candidate : Candidates) {
      Cases.push_back(Candidate->getThen());
      CaseValues.push_back(Candidate->getCaseValue());
    }

    // Collect the last else which always corresponds to case 0
    unsigned Zero = 0;
    if (ASTNode *DefaultCase = Candidates.back()->getElse()) {
      Cases.push_back(DefaultCase);
      CaseValues.push_back(Zero);
    }

    // Create the `SwitchDispatcherNode`.
    ASTTree::ast_unique_ptr Switch;
    {
      ASTNode *Tmp = new SwitchDispatcherNode(Cases, CaseValues);
      Switch.reset(Tmp);
    }

    // Invoke the dispatcher matching on the switch just reconstructed.
    matchDispatcher(AST, Switch.get(), Mark);

    // Return the new object.
    return AST.addSwitchDispatcher(std::move(Switch));

  } else if (auto *If = llvm::dyn_cast<IfNode>(RootNode)) {

    // Analyze a standard `IfNode`.
    if (If->hasThen()) {
      If->setThen(matchDispatcher(AST, If->getThen(), Mark));
    }
    if (If->hasElse()) {
      If->setElse(matchDispatcher(AST, If->getElse(), Mark));
    }

  } else if (auto *Switch = llvm::dyn_cast<SwitchNode>(RootNode)) {
    for (auto &Case : Switch->unordered_cases())
      Case = matchDispatcher(AST, Case, Mark);
    if (ASTNode *Default = Switch->getDefault())
      Default = matchDispatcher(AST, Default, Mark);
  }
  return RootNode;
}

static ASTNode *getLastOfSequenceOrSelf(ASTNode *RootNode) {
  ASTNode *LastNode = nullptr;
  if (auto *Sequence = llvm::dyn_cast<SequenceNode>(RootNode)) {
    using SeqSizeType = SequenceNode::links_container::size_type;
    SeqSizeType SequenceSize = Sequence->listSize();
    LastNode = Sequence->getNodeN(SequenceSize - 1);
  } else {
    LastNode = RootNode;
  }
  revng_assert(LastNode != nullptr);
  return LastNode;
}

// HACK: districate this mutual calling tree.
static void simplifyLastContinue(ASTNode *RootNode, ASTTree &AST);

static void removeLastContinue(ASTNode *RootNode, ASTTree &AST) {
  if (auto *LastIf = llvm::dyn_cast<IfNode>(RootNode)) {

    // TODO: unify this
    // Handle then
    if (LastIf->hasThen()) {
      ASTNode *Then = LastIf->getThen();
      if (ContinueNode *ThenContinue = llvm::dyn_cast<ContinueNode>(Then)) {
        if (ThenContinue->hasComputation()) {
          ThenContinue->setImplicit();
        } else {
          LastIf->setThen(LastIf->getElse());
          LastIf->setElse(nullptr);

          // Manual flip when removing then.
          // TODO: handle this in the flipIfEmpty phase.
          // Invert the conditional expression of the current `IfNode`.
          UniqueExpr Not;
          Not.reset(new NotNode(LastIf->getCondExpr()));
          ExprNode *NotNode = AST.addCondExpr(std::move(Not));
          LastIf->replaceCondExpr(NotNode);
        }
      } else {
        removeLastContinue(Then, AST);
      }
    }

    // Handle else
    if (LastIf->hasElse()) {
      ASTNode *Else = LastIf->getElse();
      if (ContinueNode *ElseContinue = llvm::dyn_cast<ContinueNode>(Else)) {
        if (ElseContinue->hasComputation()) {
          ElseContinue->setImplicit();
        } else {
          LastIf->setElse(nullptr);
        }
      } else {
        removeLastContinue(Else, AST);
      }
    }
  } else if (auto *Sequence = llvm::dyn_cast<SequenceNode>(RootNode)) {
    ASTNode *LastNode = getLastOfSequenceOrSelf(Sequence);
    if (ContinueNode *LastContinue = llvm::dyn_cast<ContinueNode>(LastNode)) {
      LastContinue->setImplicit();
    } else {
      removeLastContinue(LastNode, AST);
    }
  } else if (auto *Scs = llvm::dyn_cast<ScsNode>(RootNode)) {

    // Body could be nullptr (previous while/dowhile semplification)
    if (Scs->getBody() == nullptr) {
      return;
    }

    simplifyLastContinue(Scs->getBody(), AST);
  }
}

static void simplifyLastContinue(ASTNode *RootNode, ASTTree &AST) {
  if (auto *Sequence = llvm::dyn_cast<SequenceNode>(RootNode)) {
    for (ASTNode *Node : Sequence->nodes()) {
      simplifyLastContinue(Node, AST);
    }
    removeLastContinue(Sequence, AST);
  } else if (auto *If = llvm::dyn_cast<IfNode>(RootNode)) {
    if (If->hasThen()) {
      simplifyLastContinue(If->getThen(), AST);
    }
    if (If->hasElse()) {
      simplifyLastContinue(If->getElse(), AST);
    }
  } else if (auto *Scs = llvm::dyn_cast<ScsNode>(RootNode)) {

    // Body could be nullptr (previous while/dowhile semplification)
    if (Scs->getBody() == nullptr) {
      return;
    }

    // Recursive invocation on the body
    simplifyLastContinue(Scs->getBody(), AST);

    ASTNode *LastNode = getLastOfSequenceOrSelf(Scs->getBody());
    removeLastContinue(LastNode, AST);
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

    for (auto &Case : Switch->unordered_cases())
      matchDoWhile(Case, AST);
    if (ASTNode *Default = Switch->getDefault())
      matchDoWhile(Default, AST);

  } else if (auto *Scs = llvm::dyn_cast<ScsNode>(RootNode)) {
    ASTNode *Body = Scs->getBody();

    // Recursive scs nesting handling
    matchDoWhile(Body, AST);

    // ASTNode *LastNode = getLastOfSequenceOrSelf(Scs->getBody());
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

    for (auto &Case : Switch->unordered_cases())
      addComputationToContinue(Case, ConditionIf);
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

    for (auto &Case : Switch->unordered_cases())
      matchWhile(Case, AST);
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

    // ASTNode *FirstNode = getLastOfSequenceOrSelf(Scs->getBody());
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
    case ASTNode::NK_SwitchDispatcher:
    case ASTNode::NK_SwitchRegular: {
      SwitchNode *Switch = llvm::cast<SwitchNode>(Node);
      if (not LoopStack.empty())
        LoopStack.back().second.push_back(Switch);
      for (ASTNode *Case : Switch->unordered_cases())
        exec(Case, AST);
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
    case ASTNode::NK_IfDispatcher:
      BeautifyLogger << "Unexpected: IfDispatcher\n";
      revng_unreachable("unexpected node kind");
      break;
    }
  }

protected:
  LoopStackT LoopStack{};
};

void beautifyAST(Function &F, ASTTree &CombedAST, Marker &Mark) {

  ASTNode *RootNode = CombedAST.getRoot();

  // Flip IFs with empty then branches.
  revng_log(BeautifyLogger,
            "Performing IFs with empty then branches flipping\n");
  flipEmptyThen(RootNode, CombedAST);
  if (BeautifyLogger.isEnabled()) {
    CombedAST.dumpOnFile("ast", F.getName(), "After-if-flip");
  }

  // Match switch node.
  revng_log(BeautifyLogger, "Performing dispatcher nodes matching\n");
  RootNode = matchDispatcher(CombedAST, RootNode, Mark);
  if (BeautifyLogger.isEnabled()) {
    CombedAST.dumpOnFile("ast", F.getName(), "After-dispatcher-match");
  }

  // Simplify short-circuit nodes.
  revng_log(BeautifyLogger, "Performing short-circuit simplification\n");
  simplifyShortCircuit(RootNode, CombedAST, Mark);
  if (BeautifyLogger.isEnabled()) {
    CombedAST.dumpOnFile("ast", F.getName(), "After-short-circuit");
  }

  // Simplify trivial short-circuit nodes.
  revng_log(BeautifyLogger,
            "Performing trivial short-circuit simplification\n");
  simplifyTrivialShortCircuit(RootNode, CombedAST, Mark);
  if (BeautifyLogger.isEnabled()) {
    CombedAST.dumpOnFile("ast", F.getName(), "After-trivial-short-circuit");
  }

  // Match switch node.
  revng_log(BeautifyLogger, "Performing switch nodes matching\n");
  RootNode = matchSwitch(CombedAST, RootNode, Mark);
  if (BeautifyLogger.isEnabled()) {
    CombedAST.dumpOnFile("ast", F.getName(), "After-switch-match");
  }

  // Match dowhile.
  revng_log(BeautifyLogger, "Matching do-while\n");
  matchDoWhile(RootNode, CombedAST);
  if (BeautifyLogger.isEnabled()) {
    CombedAST.dumpOnFile("ast", F.getName(), "After-match-do-while");
  }

  // Match while.
  revng_log(BeautifyLogger, "Matching while\n");
  matchWhile(RootNode, CombedAST);
  if (BeautifyLogger.isEnabled()) {
    CombedAST.dumpOnFile("ast", F.getName(), "After-match-while");
  }

  // Remove useless continues.
  revng_log(BeautifyLogger, "Removing useless continue nodes\n");
  simplifyLastContinue(RootNode, CombedAST);
  if (BeautifyLogger.isEnabled()) {
    CombedAST.dumpOnFile("ast", F.getName(), "After-continue-removal");
  }

  // Fix loop breaks from within switches
  revng_log(BeautifyLogger, "Fixing loop breaks inside switches\n");
  SwitchBreaksFixer().run(RootNode, CombedAST);
  if (BeautifyLogger.isEnabled())
    CombedAST.dumpOnFile("ast", F.getName(), "After-fix-switch-breaks");

  // Remove empty sequences.
  revng_log(BeautifyLogger, "Removing emtpy sequence nodes\n");
  simplifyAtomicSequence(RootNode);
  if (BeautifyLogger.isEnabled()) {
    CombedAST.dumpOnFile("ast", F.getName(), "After-removal-empty-sequences");
  }
}
