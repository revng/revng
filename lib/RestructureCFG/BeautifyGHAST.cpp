/// \file CDecompilerBeautify.cpp
/// Beautify passes on the final AST
///

//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

#include "llvm/IR/Instructions.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Path.h"

#include "revng/ADT/RecursiveCoroutine.h"
#include "revng/Support/Assert.h"
#include "revng/Support/Debug.h"

#include "revng-c/RestructureCFG/ASTTree.h"
#include "revng-c/RestructureCFG/BeautifyGHAST.h"
#include "revng-c/RestructureCFG/ExprNode.h"
#include "revng-c/RestructureCFG/GenerateAst.h"
#include "revng-c/RestructureCFG/RegionCFGTree.h"
#include "revng-c/Support/DecompilationHelpers.h"

#include "SimplifyCompareNode.h"
#include "SimplifyDualSwitch.h"
#include "SimplifyHybridNot.h"
#include "SimplifyImplicitStatement.h"

using std::make_unique;
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
      revng_assert(If->getCondExpr());
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
    if (Scs->hasBody())
      flipEmptyThen(Scs->getBody(), AST);
  } else if (auto *Switch = llvm::dyn_cast<SwitchNode>(RootNode)) {

    for (auto &LabelCasePair : Switch->cases())
      flipEmptyThen(LabelCasePair.second, AST);
  }
}

static RecursiveCoroutine<bool> hasSideEffects(ExprNode *Expr) {
  switch (Expr->getKind()) {

  case ExprNode::NodeKind::NK_Atomic: {
    auto *Atomic = llvm::cast<AtomicNode>(Expr);
    llvm::BasicBlock *BB = Atomic->getConditionalBasicBlock();
    for (llvm::Instruction &I : *BB) {

      if (I.getType()->isVoidTy() and hasSideEffects(I)) {
        // For Instructions with void type, the MarkAssignment pass cannot
        // properly wrap them in calls to AssignmentMarker, so we need to
        // explicitly ask if they have side effects.
        rc_return true;
      } else {
        revng_assert(not isCallToTagged(&I, FunctionTags::Assign),
                     "call to assign should have matched void+hasSideEffects");
      }
    }
    rc_return false;
  } break;

  case ExprNode::NodeKind::NK_Not: {
    auto *Not = llvm::cast<NotNode>(Expr);
    rc_return rc_recur hasSideEffects(Not->getNegatedNode());
  } break;

  case ExprNode::NodeKind::NK_And: {
    auto *And = llvm::cast<AndNode>(Expr);
    const auto [LHS, RHS] = And->getInternalNodes();
    rc_return rc_recur hasSideEffects(LHS) or rc_recur hasSideEffects(RHS);
  } break;

  case ExprNode::NodeKind::NK_Or: {
    auto *Or = llvm::cast<OrNode>(Expr);
    const auto [LHS, RHS] = Or->getInternalNodes();
    rc_return rc_recur hasSideEffects(LHS) or rc_recur hasSideEffects(RHS);
  } break;

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

// Helper function to simplify short-circuit IFs
static void simplifyShortCircuit(ASTNode *RootNode, ASTTree &AST) {

  if (auto *Sequence = llvm::dyn_cast<SequenceNode>(RootNode)) {
    for (ASTNode *Node : Sequence->nodes()) {
      simplifyShortCircuit(Node, AST);
    }

  } else if (auto *Scs = llvm::dyn_cast<ScsNode>(RootNode)) {
    simplifyShortCircuit(Scs->getBody(), AST);
  } else if (auto *Switch = llvm::dyn_cast<SwitchNode>(RootNode)) {

    for (auto &LabelCasePair : Switch->cases())
      simplifyShortCircuit(LabelCasePair.second, AST);

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
            simplifyShortCircuit(If, AST);
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

            simplifyShortCircuit(If, AST);
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

            simplifyShortCircuit(If, AST);
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

            simplifyShortCircuit(If, AST);
          }
        }
      }
    }

    if (If->hasThen())
      simplifyShortCircuit(If->getThen(), AST);
    if (If->hasElse())
      simplifyShortCircuit(If->getElse(), AST);
  }
}

static void simplifyTrivialShortCircuit(ASTNode *RootNode, ASTTree &AST) {
  if (auto *Sequence = llvm::dyn_cast<SequenceNode>(RootNode)) {
    for (ASTNode *Node : Sequence->nodes()) {
      simplifyTrivialShortCircuit(Node, AST);
    }
  } else if (auto *Scs = llvm::dyn_cast<ScsNode>(RootNode)) {
    simplifyTrivialShortCircuit(Scs->getBody(), AST);

  } else if (auto *Switch = llvm::dyn_cast<SwitchNode>(RootNode)) {

    for (auto &LabelCasePair : Switch->cases())
      simplifyTrivialShortCircuit(LabelCasePair.second, AST);

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

          simplifyTrivialShortCircuit(RootNode, AST);
        }
      }
    }

    if (If->hasThen())
      simplifyTrivialShortCircuit(If->getThen(), AST);
    if (If->hasElse())
      simplifyTrivialShortCircuit(If->getElse(), AST);
  }
}

static ASTNode *matchSwitch(ASTTree &AST, ASTNode *RootNode) {

  // Inspect all the nodes composing a sequence node.
  if (auto *Sequence = llvm::dyn_cast<SequenceNode>(RootNode)) {
    for (ASTNode *&Node : Sequence->nodes()) {
      Node = matchSwitch(AST, Node);
    }
  } else if (auto *Scs = llvm::dyn_cast<ScsNode>(RootNode)) {
    // Inspect the body of a SCS region.
    Scs->setBody(matchSwitch(AST, Scs->getBody()));
  } else if (auto *If = llvm::dyn_cast<IfNode>(RootNode)) {

    // Inspect the body of an if construct.
    if (If->hasThen()) {
      If->setThen(matchSwitch(AST, If->getThen()));
    }
    if (If->hasElse()) {
      If->setElse(matchSwitch(AST, If->getElse()));
    }
  } else if (auto *Switch = llvm::dyn_cast<SwitchNode>(RootNode)) {
    // TODO: in the current situation, we should not find any switch node
    //       composed by only two case nodes. This check is only a safeguard,
    //       consider removing it altogether.
    // revng_assert(Switch->CaseSize() >= 2);
    for (auto &LabelCasePair : Switch->cases())
      LabelCasePair.second = matchSwitch(AST, LabelCasePair.second);
  }
  return RootNode;
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
    if (Body == nullptr) {
      return;
    }

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
  } break;
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
  } break;
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
  } break;
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
  } break;
  case ASTNode::NK_Code: {

    // FIXME: At the moment we use the BasicBlock size to assign a weight to the
    //       code nodes. In future, we would want to use the number of statement
    //       emitted in the decompiled code as weight (and use
    //       `AssignmentMarker`s to do that).
    CodeNode *Code = llvm::cast<CodeNode>(Node);
    llvm::BasicBlock *BB = Code->getBB();
    rc_return BB->size();
  } break;
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
  } break;
  default:
    revng_abort();
  }

  rc_return 0;
}

static RecursiveCoroutine<bool>
fallThroughScope(ASTNode *Node,
                 std::map<const ASTNode *, bool> &FallThroughMap) {
  switch (Node->getKind()) {
  case ASTNode::NK_List: {
    SequenceNode *Seq = llvm::cast<SequenceNode>(Node);

    // Invoke the fallthrough analysis on all the nodes in the sequence node.
    // Even though, after analyzing the sequence node we only use the value of
    // the last node of the sequence, it is important to recursively invoke this
    // routine on all the nodes in the sequence, since in part of the sub-tree
    // other portions of the AST benefiting from this analysis and
    // transformation could exist.
    for (ASTNode *N : Seq->nodes()) {
      bool NFallThrough = rc_recur fallThroughScope(N, FallThroughMap);
      FallThroughMap[N] = NFallThrough;
    }

    // The current sequence node is nofallthrough only if the last node of the
    // sequence node is nofallthrough.
    ASTNode *Last = Seq->getNodeN(Seq->length() - 1);
    rc_return FallThroughMap.at(Last);
  } break;
  case ASTNode::NK_Scs: {
    ScsNode *Loop = llvm::cast<ScsNode>(Node);

    // The loop node inherits the attribute from the body node of the SCS.
    if (Loop->hasBody()) {
      ASTNode *Body = Loop->getBody();
      bool BFallThrough = rc_recur fallThroughScope(Body, FallThroughMap);
      FallThroughMap[Body] = BFallThrough;
      rc_return BFallThrough;
    } else {
      rc_return true;
    }
  } break;
  case ASTNode::NK_If: {
    IfNode *If = llvm::cast<IfNode>(Node);

    // An IfNode is nofallthrough only if both its branches are nofallthrough.
    bool ThenFallThrough = false;
    if (If->hasThen()) {
      ASTNode *Then = If->getThen();
      ThenFallThrough = rc_recur fallThroughScope(Then, FallThroughMap);
      FallThroughMap[Then] = ThenFallThrough;
    }

    bool ElseFallThrough = false;
    if (If->hasElse()) {
      ASTNode *Else = If->getElse();
      ElseFallThrough = rc_recur fallThroughScope(Else, FallThroughMap);
      FallThroughMap[Else] = ElseFallThrough;
    }

    rc_return ThenFallThrough or ElseFallThrough;
  } break;
  case ASTNode::NK_Switch: {
    SwitchNode *Switch = llvm::cast<SwitchNode>(Node);

    // A SwitchNode is nofallthrough only if all its cases are nofallthrough.
    bool AllNoFallthrough = true;
    for (auto &LabelCasePair : Switch->cases()) {
      ASTNode *Case = LabelCasePair.second;
      bool CaseFallThrough = rc_recur fallThroughScope(Case, FallThroughMap);
      FallThroughMap[Case] = CaseFallThrough;
      AllNoFallthrough &= not CaseFallThrough;
    }

    // TODO: consider flipping a number of `not` in the code.
    rc_return not AllNoFallthrough;
  } break;
  case ASTNode::NK_Code: {
    CodeNode *Code = llvm::cast<CodeNode>(Node);
    llvm::BasicBlock *BB = Code->getBB();
    llvm::Instruction &I = BB->back();
    bool ReturnEnd = llvm::isa<ReturnInst>(&I);
    rc_return not ReturnEnd;
  } break;
  case ASTNode::NK_Set: {
    rc_return true;
  } break;
  case ASTNode::NK_SwitchBreak:
  case ASTNode::NK_Continue:
  case ASTNode::NK_Break: {
    rc_return false;
  } break;
  default:
    revng_abort();
  }

  rc_return true;
}

static RecursiveCoroutine<ASTNode *>
promoteNoFallthrough(ASTTree &AST,
                     ASTNode *Node,
                     std::map<const ASTNode *, bool> &FallThroughMap,
                     std::map<const ASTNode *, unsigned> &NodeWeight) {
  // Visit the current node.
  switch (Node->getKind()) {
  case ASTNode::NK_List: {
    SequenceNode *Seq = llvm::cast<SequenceNode>(Node);

    // In place of a sequence node, we need just to inspect all the nodes in the
    // sequence.
    for (ASTNode *&N : Seq->nodes()) {
      N = rc_recur promoteNoFallthrough(AST, N, FallThroughMap, NodeWeight);
    }
  } break;
  case ASTNode::NK_Scs: {
    ScsNode *Scs = llvm::cast<ScsNode>(Node);
    if (Scs->hasBody()) {
      ASTNode *Body = Scs->getBody();
      ASTNode *NewBody = rc_recur promoteNoFallthrough(AST,
                                                       Body,
                                                       FallThroughMap,
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
                                                       FallThroughMap,
                                                       NodeWeight);
      If->setThen(NewThen);
    }
    if (If->hasElse()) {

      // We only have a `else` branch, proceed with the recursive visit.
      ASTNode *Else = If->getElse();
      ASTNode *NewElse = rc_recur promoteNoFallthrough(AST,
                                                       Else,
                                                       FallThroughMap,
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
      if (FallThroughMap.at(Then) == false
          and FallThroughMap.at(Else) == false) {
        if (NodeWeight.at(Then) >= NodeWeight.at(Else))
          PromoteThen = true;
        else
          PromoteElse = true;
      } else if (FallThroughMap.at(Then) == false) {
        PromoteElse = true;
      } else if (FallThroughMap.at(Else) == false) {
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
        FallThroughMap[NewSequence] = FallThroughMap[If];
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
        FallThroughMap[NewSequence] = FallThroughMap[If];
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
                                                           FallThroughMap,
                                                           NodeWeight);
  } break;
  case ASTNode::NK_Code:
  case ASTNode::NK_Set:
  case ASTNode::NK_SwitchBreak:
  case ASTNode::NK_Continue:
  case ASTNode::NK_Break:
    // Do nothing.
    break;
  default:
    revng_unreachable();
  }
  rc_return Node;
}

static RecursiveCoroutine<ASTNode *> collapseSequences(ASTTree &AST,
                                                       ASTNode *Node) {
  switch (Node->getKind()) {
  case ASTNode::NK_List: {
    SequenceNode *Seq = llvm::cast<SequenceNode>(Node);
    SequenceNode::links_container &SeqVec = Seq->getChildVec();

    // In place of a sequence node, we need just to inspect all the nodes in the
    // sequence.

    // In this support vector, we will place the index and the size for each
    // sequence replacement list.
    std::vector<std::pair<unsigned, unsigned>> ReplacementVector;

    // This index is used to keep track of all children sequence nodes.
    unsigned I = 0;
    unsigned TotalNestedChildren = 0;
    for (ASTNode *&N : Seq->nodes()) {
      N = rc_recur collapseSequences(AST, N);

      // After analyzing the node, we check if the node is a sequence node
      // itself. If that's the case, we annotate the fact, in order to collapse
      // them in the current sequence node after resizing the vector.
      if (auto *SubSeq = llvm::dyn_cast<SequenceNode>(N)) {
        ReplacementVector.push_back(std::make_pair(I, SubSeq->length()));
        TotalNestedChildren += SubSeq->length();
      }
      I++;
    }

    // Reserve the required size in the sequence node child vector, in order to
    // avoid excessive reallocations. In the computation of the required new
    // size, remember that for every sublist we add we actually need to subtract
    // one (the spot of the sub sequence node that is being removed, which will
    // now disappear and whose place will be taken by the first node of the
    // sublist).
    SeqVec.reserve(SeqVec.size() + TotalNestedChildren
                   - ReplacementVector.size());

    // Replace in the original sequence list the child sequence node with the
    // content of the node itself.

    // This offset is used to compute the relative position of successive
    // sequence node (with respect to their original position), once the vector
    // increases in size due to the previous insertions (we actually do a -1 to
    // keep into account the sequence node itself which is being replaced).
    unsigned Offset = 0;
    for (auto &Pair : ReplacementVector) {
      unsigned Index = Pair.first + Offset;
      unsigned VecSize = Pair.second;
      Offset += VecSize - 1;

      // The substitution is done by taking an iterator the the old sequence
      // node, erasing it from the node list vector of the parent sequence,
      // inserting the nodes of the collapsed sequence node, and then removing
      // it from the AST.
      auto InternalSeqIt = SeqVec.begin() + Index;
      auto *InternalSeq = llvm::cast<SequenceNode>(*InternalSeqIt);
      auto It = SeqVec.erase(InternalSeqIt);
      SeqVec.insert(It,
                    InternalSeq->nodes().begin(),
                    InternalSeq->nodes().end());
      AST.removeASTNode(InternalSeq);
    }
  } break;
  case ASTNode::NK_Scs: {
    ScsNode *Scs = llvm::cast<ScsNode>(Node);
    if (Scs->hasBody()) {
      ASTNode *Body = Scs->getBody();
      ASTNode *NewBody = rc_recur collapseSequences(AST, Body);
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
      ASTNode *NewThen = rc_recur collapseSequences(AST, Then);
      If->setThen(NewThen);
    }
    if (If->hasElse()) {

      // We only have a `else` branch, proceed with the recursive visit.
      ASTNode *Else = If->getElse();
      ASTNode *NewElse = rc_recur collapseSequences(AST, Else);
      If->setElse(NewElse);
    }
  } break;
  case ASTNode::NK_Switch: {
    auto *Switch = llvm::cast<SwitchNode>(Node);
    for (auto &LabelCasePair : Switch->cases())
      LabelCasePair.second = rc_recur collapseSequences(AST,
                                                        LabelCasePair.second);
  } break;
  case ASTNode::NK_Code:
  case ASTNode::NK_Set:
  case ASTNode::NK_SwitchBreak:
  case ASTNode::NK_Continue:
  case ASTNode::NK_Break:
    // Do nothing.
    break;
  default:
    revng_unreachable();
  }
  rc_return Node;
}

static ASTNode *promoteNoFallthroughIf(ASTNode *RootNode, ASTTree &AST) {

  // This map will contain the result of the fallthough analysis.
  // We considered using a `std::set` in place of the `std::map`, but the `map`
  // has the advantage of making us able to assert that the fallthrough
  // information has been computed for every node in the AST. If we only have a
  // `set`, where a node is present in the set only if its scope does
  // fallthrough, we cannot distinguish the situation where a node does not
  // fallthrough, or simply a bug in the algorithm which didn't compute the
  // fallthrough value for the specific node.
  std::map<const ASTNode *, bool> FallThroughMap;

  // In this map, we store the weight of the AST starting from a node and
  // going down.
  std::map<const ASTNode *, unsigned> NodeWeight;

  // Run the analysis which marks the fallthrough property of the nodes.
  bool RootFallThrough = fallThroughScope(RootNode, FallThroughMap);
  FallThroughMap[RootNode] = RootFallThrough;

  // Run the analysis which computes the AST weight of the nodes on the tree.
  unsigned RootWeight = computeCumulativeNodeWeight(RootNode, NodeWeight);
  NodeWeight[RootNode] = RootWeight;

  // Run the fallthrough promotion.
  RootNode = promoteNoFallthrough(AST, RootNode, FallThroughMap, NodeWeight);

  // Run the sequence nodes collapse.
  RootNode = collapseSequences(AST, RootNode);

  // Update the root field of the AST.
  AST.setRoot(RootNode);

  return RootNode;
}

void beautifyAST(Function &F, ASTTree &CombedAST) {

  // If the --short-circuit-metrics-output-dir=dir argument was passed from
  // command line, we need to print the statistics for the short circuit metrics
  // into a file with the function name, inside the directory 'dir'.
  std::unique_ptr<llvm::raw_fd_ostream> StatsFileStream;
  if (OutputPath.getNumOccurrences())
    StatsFileStream = openFunctionFile(OutputPath, F.getName(), ".csv");

  ShortCircuitCounter = 0;
  TrivialShortCircuitCounter = 0;

  ASTNode *RootNode = CombedAST.getRoot();

  if (BeautifyLogger.isEnabled()) {
    CombedAST.dumpASTOnFile(F.getName().str(), "ast", "01-Before-beautify");
  }

  // Simplify short-circuit nodes.
  revng_log(BeautifyLogger, "Performing short-circuit simplification\n");
  simplifyShortCircuit(RootNode, CombedAST);
  if (BeautifyLogger.isEnabled()) {
    CombedAST.dumpASTOnFile(F.getName().str(), "ast", "02-After-short-circuit");
  }

  // Flip IFs with empty then branches.
  // We need to do it before simplifyTrivialShortCircuit, otherwise that
  // functions will need to check every possile combination of then-else to
  // simplify. In this way we can keep it simple.
  revng_log(BeautifyLogger,
            "Performing IFs with empty then branches flipping\n");
  flipEmptyThen(RootNode, CombedAST);
  if (BeautifyLogger.isEnabled()) {
    CombedAST.dumpASTOnFile(F.getName().str(), "ast", "03-After-if-flip-1");
  }

  // Simplify trivial short-circuit nodes.
  revng_log(BeautifyLogger,
            "Performing trivial short-circuit simplification\n");
  simplifyTrivialShortCircuit(RootNode, CombedAST);
  if (BeautifyLogger.isEnabled()) {
    CombedAST.dumpASTOnFile(F.getName().str(),
                            "ast",
                            "04-After-trivial-short-circuit");
  }

  // Flip IFs with empty then branches.
  // We need to do it here again, after simplifyTrivialShortCircuit, because
  // that functions can create empty then branches in some situations, and we
  // want to flip them as well.
  revng_log(BeautifyLogger,
            "Performing IFs with empty then branches flipping\n");
  flipEmptyThen(RootNode, CombedAST);
  if (BeautifyLogger.isEnabled()) {
    CombedAST.dumpASTOnFile(F.getName().str(), "ast", "05-After-if-flip-2");
  }

  // Match switch node.
  revng_log(BeautifyLogger, "Performing switch nodes matching\n");
  RootNode = matchSwitch(CombedAST, RootNode);
  if (BeautifyLogger.isEnabled()) {
    CombedAST.dumpASTOnFile(F.getName().str(), "ast", "06-After-switch-match");
  }

  // Match while.
  revng_log(BeautifyLogger, "Matching while\n");
  matchWhile(RootNode, CombedAST);
  if (BeautifyLogger.isEnabled()) {
    CombedAST.dumpASTOnFile(F.getName().str(), "ast", "07-After-match-while");
  }

  // Match dowhile.
  revng_log(BeautifyLogger, "Matching do-while\n");
  matchDoWhile(RootNode, CombedAST);
  if (BeautifyLogger.isEnabled()) {
    CombedAST.dumpASTOnFile(F.getName().str(),
                            "ast",
                            "08-After-match-do-while");
  }

  // Remove useless continues.
  revng_log(BeautifyLogger, "Removing useless continue nodes\n");
  simplifyImplicitContinue(CombedAST);
  if (BeautifyLogger.isEnabled()) {
    CombedAST.dumpASTOnFile(F.getName().str(),
                            "ast",
                            "09-After-continue-removal");
  }

  // Perform the simplification of `switch` with two entries in a `if`
  revng_log(BeautifyLogger, "Performing the dual switch simplification\n");
  RootNode = simplifyDualSwitch(CombedAST, RootNode);
  if (BeautifyLogger.isEnabled()) {
    CombedAST.dumpASTOnFile(F.getName().str(),
                            "ast",
                            "10-After-dual-switch-simplify");
  }

  // Fix loop breaks from within switches
  revng_log(BeautifyLogger, "Fixing loop breaks inside switches\n");
  SwitchBreaksFixer().run(RootNode, CombedAST);
  if (BeautifyLogger.isEnabled())
    CombedAST.dumpASTOnFile(F.getName().str(),
                            "ast",
                            "11-After-fix-switch-breaks");

  // Remove empty sequences.
  revng_log(BeautifyLogger, "Removing empty sequence nodes\n");
  simplifyAtomicSequence(CombedAST, RootNode);
  if (BeautifyLogger.isEnabled()) {
    CombedAST.dumpASTOnFile(F.getName().str(),
                            "ast",
                            "12-After-removal-empty-sequences");
  }

  // Remove unnecessary scopes under the fallthrough analysis.
  revng_log(BeautifyLogger, "Analyzing fallthrough scopes\n");
  RootNode = promoteNoFallthroughIf(RootNode, CombedAST);
  if (BeautifyLogger.isEnabled()) {
    CombedAST.dumpASTOnFile(F.getName().str(),
                            "ast",
                            "13-After-fallthrough-scope-analysis");
  }

  // Flip IFs with empty then branches.
  // We need to do it here again, after the promotion due to the `nofallthroguh`
  // analysis run before.
  revng_log(BeautifyLogger,
            "Performing IFs with empty then branches flipping\n");
  flipEmptyThen(RootNode, CombedAST);
  if (BeautifyLogger.isEnabled()) {
    CombedAST.dumpASTOnFile(F.getName().str(), "ast", "14-After-if-flip-3");
  }

  // Perform the double `not` simplification (`not` on the GHAST and `not` in
  // the IR).
  revng_log(BeautifyLogger, "Performing the double not simplification\n");
  RootNode = simplifyHybridNot(CombedAST, RootNode);
  if (BeautifyLogger.isEnabled()) {
    CombedAST.dumpASTOnFile(F.getName().str(),
                            "ast",
                            "15-After-double-not-simplify");
  }

  // Perform the `CompareNode` simplification. A `CompareNode` preceded by a
  // `not` is transformed in the `CompareNode` itself with the flipped
  // comparison predicate
  revng_log(BeautifyLogger, "Performing the compare node simplification\n");
  simplifyCompareNode(CombedAST, RootNode);
  if (BeautifyLogger.isEnabled()) {
    CombedAST.dumpASTOnFile(F.getName().str(),
                            "ast",
                            "16-After-compare-node-simplify-3");
  }

  // Perform the simplification of the implicit `return`, i.e., a `return` of
  // type `void`, which lies on a path followed by no other statements.
  revng_log(BeautifyLogger, "Performing the implicit return simplification\n");
  simplifyImplicitReturn(CombedAST, RootNode);
  if (BeautifyLogger.isEnabled()) {
    CombedAST.dumpASTOnFile(F.getName().str(),
                            "ast",
                            "17-After-implicit-return-simplify");
  }

  // Serialize the collected metrics in the statistics file if necessary
  if (StatsFileStream) {
    *StatsFileStream << "function,short-circuit,trivial-short-circuit\n"
                     << F.getName().data() << "," << ShortCircuitCounter << ","
                     << TrivialShortCircuitCounter << "\n";
  }
}
