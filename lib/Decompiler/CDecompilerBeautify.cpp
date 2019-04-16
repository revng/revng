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

Logger<> BeautifyLogger("beautify");

using namespace llvm;
using std::make_unique;
using std::unique_ptr;

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
      auto Not = std::make_unique<NotNode>(If->getCondExpr());
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
    for (auto &Case : Switch->cases()) {
      simplifyShortCircuit(Case.second, AST, Mark);
    }

  } else if (auto *SwitchCheck = llvm::dyn_cast<SwitchCheckNode>(RootNode)) {
    for (auto &Case : SwitchCheck->cases()) {
      simplifyShortCircuit(Case.second, AST, Mark);
    }

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
            auto NotB = std::make_unique<NotNode>(NestedIf->getCondExpr());
            ExprNode *NotBNode = AST.addCondExpr(std::move(NotB));

            auto AAndNotB = std::make_unique<AndNode>(If->getCondExpr(),
                                                      NotBNode);
            ExprNode *AAndNotBNode = AST.addCondExpr(std::move(AAndNotB));

            If->replaceCondExpr(AAndNotBNode);

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
            auto AAndB = std::make_unique<AndNode>(If->getCondExpr(),
                                                   NestedIf->getCondExpr());
            ExprNode *AAndBNode = AST.addCondExpr(std::move(AAndB));

            If->replaceCondExpr(AAndBNode);

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
            auto NotA = std::make_unique<NotNode>(If->getCondExpr());
            ExprNode *NotANode = AST.addCondExpr(std::move(NotA));

            auto NotB = std::make_unique<NotNode>(NestedIf->getCondExpr());
            ExprNode *NotBNode = AST.addCondExpr(std::move(NotB));

            auto NotAAndNotB = std::make_unique<AndNode>(NotANode, NotBNode);
            ExprNode *NotAAndNotBNode = AST.addCondExpr(std::move(NotAAndNotB));

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
            auto NotA = std::make_unique<NotNode>(If->getCondExpr());
            ExprNode *NotANode = AST.addCondExpr(std::move(NotA));

            auto NotAAndB = std::make_unique<AndNode>(NotANode,
                                                      NestedIf->getCondExpr());
            ExprNode *NotAAndBNode = AST.addCondExpr(std::move(NotAAndB));

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
    for (auto &Case : Switch->cases()) {
      simplifyTrivialShortCircuit(Case.second, AST, Mark);
    }

  } else if (auto *SwitchCheck = llvm::dyn_cast<SwitchCheckNode>(RootNode)) {
    for (auto &Case : SwitchCheck->cases()) {
      simplifyTrivialShortCircuit(Case.second, AST, Mark);
    }

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
          auto AAndB = std::make_unique<AndNode>(If->getCondExpr(),
                                                 InternalIf->getCondExpr());
          ExprNode *AAndBNode = AST.addCondExpr(std::move(AAndB));

          If->replaceCondExpr(AAndBNode);

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

static bool isSwitchCheck(ASTNode *Candidate) {
  llvm::BasicBlock *BB = Candidate->getOriginalBB();

  // Check that the if corresponds to an original basic block.
  if (BB != nullptr) {

    // Check that the body contains an `icmp` instruction over the condition
    // of the switch and a constant, and then a conditional branch.
    if (BB->size() == 2) {
      auto &InstList = BB->getInstList();
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
    if (isSwitchCheck(If)) {

      // Get the Value of the condition.
      llvm::Value *SwitchValue = getCaseValue(If);

      // An IfEqualNode represents the start of a switch statement.
      llvm::SmallVector<IfNode *, 8> Candidates;
      Candidates.push_back(If);

      // Continue to accumulate the IfEqual nodes until it is possible.
      while (If->getElse() and isSwitchCheck(If->getElse())
             and (SwitchValue == getCaseValue(If->getElse()))) {
        If = llvm::cast<IfNode>(If->getElse());
        Candidates.push_back(If);
      }

      std::vector<std::pair<ConstantInt *, ASTNode *>> CandidatesCases;
      for (IfNode *Candidate : Candidates) {
        ConstantInt *CaseConstant = getCaseConstant(Candidate);
        CandidatesCases.push_back({CaseConstant, Candidate->getThen()});
      }

      // Collect the last else (which will become the default case).
      llvm::Type *SwitchType = SwitchValue->getType();
      auto *Zero = cast<ConstantInt>(llvm::Constant::getNullValue(SwitchType));
      ASTNode *DefaultCase = Candidates.back()->getElse();
      if (DefaultCase == nullptr)
        DefaultCase = AST.addSequenceNode();
      CandidatesCases.push_back(std::make_pair(Zero, DefaultCase));

      // Create the switch node.
      auto Switch = std::make_unique<SwitchNode>(SwitchValue, CandidatesCases);

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
    for (auto &Case : Switch->cases()) {
      Case.second = matchSwitch(AST, Case.second, Mark);
    }
  } else if (auto *SwitchCheck = llvm::dyn_cast<SwitchCheckNode>(RootNode)) {
    for (auto &Case : SwitchCheck->cases()) {
      Case.second = matchSwitch(AST, Case.second, Mark);
    }
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

  } else if (auto *IfCheck = llvm::dyn_cast<IfCheckNode>(RootNode)) {

    // An `IfCheckNode` represents the start of a dispatcher, which we want to
    // represent using a `SwitchCheckNode`.
    std::vector<IfCheckNode *> Candidates;
    Candidates.push_back(IfCheck);

    // Continue to accumulate the `IfCheck` nodes until it is possible.
    while (auto *SuccChk = dyn_cast_or_null<IfCheckNode>(IfCheck->getElse())) {
      Candidates.push_back(SuccChk);
      IfCheck = SuccChk;
    }

    std::vector<std::pair<unsigned, ASTNode *>> CandidatesCases;
    for (IfCheckNode *Candidate : Candidates) {
      unsigned CaseConstant = Candidate->getCaseValue();
      CandidatesCases.push_back({CaseConstant, Candidate->getThen()});
    }

    // Collect the last else (which will become the default case).
    unsigned Zero = 0;
    if (ASTNode *DefaultCase = Candidates.back()->getElse())
      CandidatesCases.push_back(std::make_pair(Zero, DefaultCase));

    // Create the `SwitchCheckNode`.
    unique_ptr<SwitchCheckNode> Switch(new SwitchCheckNode(CandidatesCases));

    // Invoke the dispatcher matching on the switch just reconstructed.
    matchDispatcher(AST, Switch.get(), Mark);

    // Return the new object.
    return AST.addSwitchCheck(std::move(Switch));

  } else if (auto *If = llvm::dyn_cast<IfNode>(RootNode)) {

    // Analyze a standard `IfNode`.
    if (If->hasThen()) {
      If->setThen(matchDispatcher(AST, If->getThen(), Mark));
    }
    if (If->hasElse()) {
      If->setElse(matchDispatcher(AST, If->getElse(), Mark));
    }

  } else if (auto *Switch = llvm::dyn_cast<SwitchNode>(RootNode)) {
    for (auto &Case : Switch->cases()) {
      Case.second = matchDispatcher(AST, Case.second, Mark);
    }

  } else if (auto *SwitchCheck = llvm::dyn_cast<SwitchCheckNode>(RootNode)) {
    for (auto &Case : SwitchCheck->cases()) {
      Case.second = matchDispatcher(AST, Case.second, Mark);
    }
  }

  return RootNode;
}

static ASTNode *getLastOfSequenceOrSelf(ASTNode *RootNode) {
  ASTNode *LastNode = nullptr;
  if (auto *Sequence = llvm::dyn_cast<SequenceNode>(RootNode)) {
    int SequenceSize = Sequence->listSize();
    LastNode = Sequence->getNodeN(SequenceSize - 1);
  } else {
    LastNode = RootNode;
  }
  revng_assert(LastNode != nullptr);
  return LastNode;
}

// HACK: districate this mutual calling tree.
static void simplifyLastContinue(ASTNode *RootNode);

static void removeLastContinue(ASTNode *RootNode) {
  if (auto *LastIf = llvm::dyn_cast<IfNode>(RootNode)) {

    // TODO: unify this
    // Handle then
    if (LastIf->hasThen()) {
      ASTNode *Then = LastIf->getThen();
      if (llvm::isa<ContinueNode>(Then)) {

        // Manual flip when removing then.
        // TODO: handle this in the flipIfEmpty phase.
        LastIf->setThen(LastIf->getElse());
        LastIf->setElse(nullptr);
      } else {
        removeLastContinue(Then);
      }
    }

    // Handle else
    if (LastIf->hasElse()) {
      ASTNode *Else = LastIf->getElse();
      if (llvm::isa<ContinueNode>(Else)) {
        LastIf->setElse(nullptr);
      } else {
        removeLastContinue(Else);
      }
    }
  } else if (auto *Sequence = llvm::dyn_cast<SequenceNode>(RootNode)) {
    ASTNode *LastNode = getLastOfSequenceOrSelf(Sequence);
    if (llvm::isa<ContinueNode>(LastNode)) {
      Sequence->removeNode(LastNode);
    } else {
      removeLastContinue(LastNode);
    }
  } else if (auto *Scs = llvm::dyn_cast<ScsNode>(RootNode)) {

    // Body could be nullptr (previous while/dowhile semplification)
    if (Scs->getBody() == nullptr) {
      return;
    }

    simplifyLastContinue(Scs->getBody());
  }
}

static void simplifyLastContinue(ASTNode *RootNode) {
  if (auto *Sequence = llvm::dyn_cast<SequenceNode>(RootNode)) {
    for (ASTNode *Node : Sequence->nodes()) {
      simplifyLastContinue(Node);
    }
    removeLastContinue(Sequence);
  } else if (auto *If = llvm::dyn_cast<IfNode>(RootNode)) {
    if (If->hasThen()) {
      simplifyLastContinue(If->getThen());
    }
    if (If->hasElse()) {
      simplifyLastContinue(If->getElse());
    }
  } else if (auto *Scs = llvm::dyn_cast<ScsNode>(RootNode)) {

    // Body could be nullptr (previous while/dowhile semplification)
    if (Scs->getBody() == nullptr) {
      return;
    }

    // Recursive invocation on the body
    simplifyLastContinue(Scs->getBody());

    ASTNode *LastNode = getLastOfSequenceOrSelf(Scs->getBody());
    removeLastContinue(LastNode);
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
    for (auto &Case : Switch->cases()) {
      matchDoWhile(Case.second, AST);
    }

  } else if (auto *SwitchCheck = llvm::dyn_cast<SwitchCheckNode>(RootNode)) {
    for (auto &Case : SwitchCheck->cases()) {
      matchDoWhile(Case.second, AST);
    }

  } else if (auto *Scs = llvm::dyn_cast<ScsNode>(RootNode)) {
    ASTNode *Body = Scs->getBody();

    // Recursive scs nesting handling
    matchDoWhile(Body, AST);

    // ASTNode *LastNode = getLastOfSequenceOrSelf(Scs->getBody());
    ASTNode *LastNode = nullptr;
    bool InsideSequence = false;
    if (auto *Sequence = llvm::dyn_cast<SequenceNode>(Body)) {
      int SequenceSize = Sequence->listSize();
      LastNode = Sequence->getNodeN(SequenceSize - 1);
      InsideSequence = true;
    } else {
      LastNode = Body;
    }

    revng_assert(LastNode != nullptr);

    if (auto *If = llvm::dyn_cast<IfNode>(LastNode)) {

      // Only if nodes with both branches are candidates.
      if (If->hasBothBranches()) {
        ASTNode *Then = If->getThen();
        ASTNode *Else = If->getElse();

        if (llvm::isa<BreakNode>(Then) and llvm::isa<ContinueNode>(Else)) {

          Scs->setDoWhile(If);
          // Invert the conditional expression of the current `IfNode`.
          auto Not = std::make_unique<NotNode>(If->getCondExpr());
          ExprNode *NotNode = AST.addCondExpr(std::move(Not));
          If->replaceCondExpr(NotNode);

          // Remove the if node
          if (InsideSequence) {
            cast<SequenceNode>(Body)->removeNode(If);
          } else {
            Scs->setBody(nullptr);
          }
        } else if (llvm::isa<BreakNode>(Else)
                   and llvm::isa<ContinueNode>(Then)) {
          Scs->setDoWhile(If);

          // Remove the if node
          if (InsideSequence) {
            cast<SequenceNode>(Body)->removeNode(If);
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
    for (auto &Case : Switch->cases()) {
      addComputationToContinue(Case.second, ConditionIf);
    }

  } else if (auto *SwitchCheck = llvm::dyn_cast<SwitchCheckNode>(RootNode)) {
    for (auto &Case : SwitchCheck->cases()) {
      addComputationToContinue(Case.second, ConditionIf);
    }
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
    for (auto &Case : Switch->cases()) {
      matchWhile(Case.second, AST);
    }

  } else if (auto *SwitchCheck = llvm::dyn_cast<SwitchCheckNode>(RootNode)) {
    for (auto &Case : SwitchCheck->cases()) {
      matchWhile(Case.second, AST);
    }

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
    if (auto *Sequence = llvm::dyn_cast<SequenceNode>(Body)) {
      FirstNode = Sequence->getNodeN(0);
      InsideSequence = true;
    } else {
      FirstNode = Body;
    }

    revng_assert(FirstNode != nullptr);

    if (auto *If = llvm::dyn_cast<IfNode>(FirstNode)) {

      // Only if nodes with both branches are candidates.
      if (If->hasBothBranches()) {
        ASTNode *Then = If->getThen();
        ASTNode *Else = If->getElse();

        if (llvm::isa<BreakNode>(Then)) {

          Scs->setWhile(If);
          // Invert the conditional expression of the current `IfNode`.
          auto Not = std::make_unique<NotNode>(If->getCondExpr());
          ExprNode *NotNode = AST.addCondExpr(std::move(Not));
          If->replaceCondExpr(NotNode);

          // Remove the if node
          if (InsideSequence) {
            cast<SequenceNode>(Body)->removeNode(If);
          } else {
            Scs->setBody(If->getElse());

            // Add computation before the continue nodes
            addComputationToContinue(Scs->getBody(), If);
          }
        } else if (llvm::isa<BreakNode>(Else)) {
          Scs->setWhile(If);

          // Remove the if node
          if (InsideSequence) {
            cast<SequenceNode>(Body)->removeNode(If);
          } else {
            Scs->setBody(If->getThen());

            // Add computation before the continue nodes
            addComputationToContinue(Scs->getBody(), If);
          }
        }
      }
    }
  } else {
    BeautifyLogger << "No matching done\n";
  }
}

static void fixSwitchBreaks(ASTNode *RootNode, ASTTree &AST) {
  if (RootNode == nullptr)
    return;
  switch (RootNode->getKind()) {
  case ASTNode::NK_If: {
    IfNode *If = llvm::cast<IfNode>(RootNode);
    fixSwitchBreaks(If->getThen(), AST);
    fixSwitchBreaks(If->getElse(), AST);
  } break;
  case ASTNode::NK_Scs: {
    ScsNode *Loop = llvm::cast<ScsNode>(RootNode);
    fixSwitchBreaks(Loop->getBody(), AST);
  } break;
  case ASTNode::NK_List: {
    SequenceNode *Seq = llvm::cast<SequenceNode>(RootNode);
    for (ASTNode *N : Seq->nodes())
      fixSwitchBreaks(N, AST);
  } break;
  case ASTNode::NK_Switch: {
    SwitchNode *Switch = llvm::cast<SwitchNode>(RootNode);
    for (auto &Pair : Switch->cases()) {
      SwitchBreakNode *Break = AST.addSwitchBreak();
      if (SequenceNode *Seq = llvm::dyn_cast<SequenceNode>(Pair.second)) {
        Seq->addNode(Break);
      } else {
        SequenceNode *Case = AST.addSequenceNode();
        Case->addNode(Pair.second);
        Case->addNode(Break);
      }
    }
  } break;
  case ASTNode::NK_SwitchCheck: {
    SwitchCheckNode *Check = llvm::cast<SwitchCheckNode>(RootNode);
    for (auto &Pair : Check->cases())
      fixSwitchBreaks(Pair.second, AST);
  } break;
  case ASTNode::NK_Set:
  case ASTNode::NK_Code:
  case ASTNode::NK_Break:
  case ASTNode::NK_Continue:
  case ASTNode::NK_SwitchBreak:
    break; // do nothing
  case ASTNode::NK_IfCheck:
    BeautifyLogger << "Unexpected: IfCheck\n";
  default:
    revng_unreachable("unexpected node kind");
  }
}

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
  simplifyLastContinue(RootNode);
  if (BeautifyLogger.isEnabled()) {
    CombedAST.dumpOnFile("ast", F.getName(), "After-continue-removal");
  }

  // Fix loop breaks from within switches
  revng_log(BeautifyLogger, "Fixing loop breaks inside switches\n");
  fixSwitchBreaks(RootNode, CombedAST);
  if (BeautifyLogger.isEnabled())
    CombedAST.dumpOnFile("ast", F.getName(), "After-fix-switch-breaks");

  // Remove empty sequences.
  revng_log(BeautifyLogger, "Removing emtpy sequence nodes\n");
  simplifyAtomicSequence(RootNode);
  if (BeautifyLogger.isEnabled()) {
    CombedAST.dumpOnFile("ast", F.getName(), "After-removal-empty-sequences");
  }
}
