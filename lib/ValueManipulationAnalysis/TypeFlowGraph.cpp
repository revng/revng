//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/ADT/ReversePostOrderTraversal.h"
#include "revng/Support/Assert.h"
#include "revng/Support/Debug.h"
#include "revng/Support/FunctionTags.h"

#include "revng-c/Support/FunctionTags.h"
#include "revng-c/Support/ModelHelpers.h"
#include "revng-c/ValueManipulationAnalysis/TypeColors.h"

#include "TypeFlowGraph.h"
#include "TypeFlowGraphWriter.h"
#include "TypeFlowNode.h"

using namespace vma;
using namespace llvm;

static Logger<> TGLog("vma-tg");

/// Returns a ColorSet with the types that can be assigned to a given use or
/// value.
static ColorSet getAcceptedColors(FunctionMetadataCache &Cache,
                                  const UseOrValue &Content,
                                  const model::Binary *Model) {

  // Instructions and operand uses should be the only thing remaining
  bool IsContentInst = isInst(Content);
  auto *ContentInst = IsContentInst ? dyn_cast<Instruction>(getValue(Content)) :
                                      nullptr;

  // Do we have strong model information about this node? If so, use that
  if (Model) {

    if (isValue(Content)) {
      const Value *V = getValue(Content);
      // TODO arguments should take the model types
      if (isa<Argument>(V))
        return ALL_COLORS;
    }

    if (IsContentInst or isUse(Content)) {

      // Deduce type for the use or for the value, depending on which type of
      // node we are looking at
      auto DeducedTypes = IsContentInst ?
                            getStrongModelInfo(Cache, ContentInst, *Model) :
                            getExpectedModelType(Cache,
                                                 getUse(Content),
                                                 *Model);

      // If we weren't able to deduce anything, fallthrough to the default
      // handling when there is no model.
      if (not DeducedTypes.empty()) {

        if (DeducedTypes.size() == 1) {
          ColorSet Result = QTToColor(DeducedTypes.back());
          if (isUse(Content)
              and getUse(Content)->get()->getType()->isIntegerTy(1))
            Result.addColor(BOOLNESS);

          return Result;
        }

        // There are cases in which we can associate to an LLVM value (typically
        // an aggregate) more than one model type, e.g. for values returned by
        // RawFunctionTypes or for calls to StructInitializer.
        // In these cases, the aggregate itself has no color. Not that the value
        // extracted from that will instead have a color, which is inferred by
        // `getStrongModelInfo()`.
        return NO_COLOR;
      }
    }
  }

  // If the content of the node is an Instruction's Value, assign colors
  // based on the instruction's opcode. Otherwise, if we are creating a node for
  // one of the operands, find which the user of the operand and check its
  // opcode.
  const Instruction *I = IsContentInst ?
                           ContentInst :
                           cast<Instruction>(getUse(Content)->getUser());

  // Arguments, constants, globals etc.
  if (isValue(Content)) {
    const Value *V = getValue(Content);

    if (isa<Argument>(V))
      return ALL_COLORS;

    // Constants and globals should not be infected, since they don't belong to
    // a single function.
    if (not isa<Instruction>(V))
      return NO_COLOR;
  }

  // Fallback to manually matching LLVM instructions that provides us with rich
  // type information
  switch (I->getOpcode()) {
  case Instruction::FNeg:
  case Instruction::FAdd:
  case Instruction::FMul:
  case Instruction::FSub:
  case Instruction::FDiv:
  case Instruction::FRem:
  case Instruction::FPExt:
    return FLOATNESS;
    break;

  case Instruction::FCmp:
    if (IsContentInst)
      return BOOLNESS;
    else
      return FLOATNESS;
    break;

  case Instruction::ICmp:
    if (IsContentInst)
      return BOOLNESS;
    if (cast<ICmpInst>(I)->isSigned())
      return SIGNEDNESS;
    if (cast<ICmpInst>(I)->isUnsigned())
      return UNSIGNEDNESS | POINTERNESS;
    break;

  case Instruction::SDiv:
  case Instruction::SRem:
    return SIGNEDNESS | NUMBERNESS;
    break;

  case Instruction::UDiv:
  case Instruction::URem:
    return UNSIGNEDNESS | NUMBERNESS;
    break;

  case Instruction::Alloca:
    if (IsContentInst)
      return POINTERNESS;
    if (getUse(Content)->get() == cast<AllocaInst>(I)->getArraySize())
      return UNSIGNEDNESS;
    break;

  case Instruction::Load:
    if (isUse(Content)
        && getOpNo(Content) == cast<LoadInst>(I)->getPointerOperandIndex())
      return POINTERNESS;
    break;

  case Instruction::Store:
    if (isUse(Content)
        && getOpNo(Content) == cast<StoreInst>(I)->getPointerOperandIndex())
      return POINTERNESS;
    break;

  case Instruction::AShr:
    if (IsContentInst or getOpNo(Content) == 0)
      return SIGNEDNESS | NUMBERNESS;
    if (getOpNo(Content) == 1)
      return ~(FLOATNESS | POINTERNESS);
    break;

  case Instruction::LShr:
    if (IsContentInst or getOpNo(Content) == 0)
      return UNSIGNEDNESS | NUMBERNESS;
    if (getOpNo(Content) == 1)
      return ~(FLOATNESS | POINTERNESS);
    break;

  case Instruction::Shl:
    return (SIGNEDNESS | UNSIGNEDNESS | BOOLNESS | NUMBERNESS);
    break;

  case Instruction::Mul:
    return ~(FLOATNESS | POINTERNESS);
    break;

  case Instruction::Br:
    if (isUse(Content) && cast<BranchInst>(I)->isConditional()
        && getUse(Content)->get() == cast<BranchInst>(I)->getCondition())
      return BOOLNESS;
    break;

  case Instruction::Select:
    if (isUse(Content)
        && getUse(Content)->get() == cast<SelectInst>(I)->getCondition())
      return BOOLNESS;
    break;

  case Instruction::And:
  case Instruction::Or:
  case Instruction::Xor:
    return NUMBERNESS | UNSIGNEDNESS | SIGNEDNESS | BOOLNESS;
    break;

  case Instruction::GetElementPtr:
    revng_abort("Didn't expect to find a GEP here");
    break;
  }

  return ~NUMBERNESS;
}

TypeFlowNode *TypeFlowGraph::addNodeContaining(FunctionMetadataCache &Cache,
                                               const UseOrValue &NC) {
  revng_assert(not ContentToNodeMap.count(NC));

  NodeColorProperty InitialColors = { NO_COLOR,
                                      getAcceptedColors(Cache, NC, Model) };
  auto *N = this->addNode(NC, InitialColors);
  ContentToNodeMap[NC] = N;

  return N;
}

// --------------- TypeFlowGraph

TypeFlowNode *
TypeFlowGraph::getNodeContaining(const UseOrValue &Content) const {
  const auto &It = ContentToNodeMap.find(Content);
  revng_assert(It != ContentToNodeMap.end());

  return It->second;
}

void TypeFlowGraph::dump(const llvm::Twine &Title, std::string FileName) {
  llvm::WriteGraph(this,
                   this->Func->getName() + Title,
                   false,
                   this->Func->getName() + Title,
                   FileName);
}

void TypeFlowGraph::print(llvm::raw_ostream &OS) {
  llvm::WriteGraph(OS, this);
}

void TypeFlowGraph::view() {
  llvm::ViewGraph(this, this->Func->getName(), false, this->Func->getName());
}

// --------------- TypeFlowGraph manipulation

/// Check if two nodes are already connected before adding the successor
static bool
addSuccessorIfAbsent(TypeFlowNode *N1, TypeFlowNode *N2, const EdgeLabel &E) {
  if (llvm::is_contained(N1->successors(), N2))
    return false;

  N1->addSuccessor(N2, E);
  return true;
}

/// Add edge (possibly both ways) between two nodes, based on the content
static bool connect(TypeFlowNode *N1, TypeFlowNode *N2) {

  // Value -> Value: no connection
  if (N1->isValue() and N2->isValue())
    return false;

  // Use -> Use: check that they have the same user before connecting
  if (N1->isUse() and N2->isUse()) {
    auto *N1User = N1->getUse()->getUser();
    auto *N2User = N2->getUse()->getUser();

    if (N1User != N2User)
      return false;

    const Instruction *I = cast<Instruction>(N1User);

    switch (I->getOpcode()) {
      // Currently the only instructions for which there is a typeflow between
      // the operands are comparisons
    case Instruction::ICmp: {
      bool Connected = false;

      Connected |= addSuccessorIfAbsent(N1, N2, ALL_COLORS);
      Connected |= addSuccessorIfAbsent(N2, N1, ALL_COLORS);

      return Connected;
    }

    default:
      return false;
    }
  }

  // If it's not Use->Use or Value->Value, one of the two is a Value and the
  // other one is a Use
  revng_assert((N1->isValue() and N2->isUse())
               or (N1->isUse() and N2->isValue()));

  TypeFlowNode *ValNode = N1->isValue() ? N1 : N2;
  TypeFlowNode *UseNode = N1->isUse() ? N1 : N2;

  revng_assert(ValNode->isValue() and UseNode->isUse());

  // Use -> Value: connect according to the accepted colors
  if (UseNode->getUse()->get() == ValNode->getValue()) {
    bool Connected = false;

    Connected |= addSuccessorIfAbsent(ValNode, UseNode, UseNode->getAccepted());
    Connected |= addSuccessorIfAbsent(UseNode, ValNode, ValNode->getAccepted());

    return Connected;
  }

  // Use -> User: connect according to user opcode
  if (UseNode->getUse()->getUser() == ValNode->getValue()) {
    const Instruction *I = cast<Instruction>(UseNode->getUse()->getUser());
    const size_t OpNo = UseNode->getUse()->getOperandNo();

    const auto AddBidirectionalEdge =
      [](TypeFlowNode *N1, TypeFlowNode *N2, const ColorSet C) {
        bool Connected = false;
        Connected |= addSuccessorIfAbsent(N1, N2, C);
        Connected |= addSuccessorIfAbsent(N2, N1, C);
        return Connected;
      };

    switch (I->getOpcode()) {
    case Instruction::LShr:
      // TODO: What to do when shifting booleans and pointers is still a
      // matter of discussion. For now the decision is that if a pointer or a
      // boolean gets shifted, it is not a pointer/boolean anymore, so we
      // don't add propagation edges forward for these colors. Also, if the
      // result of a shift is used in a boolean/pointer sense, this doesn't
      // give us enough information to color the operands, so don't propagate
      // backwards either.
      if (OpNo == 0)
        return AddBidirectionalEdge(UseNode, ValNode, UNSIGNEDNESS);
      break;

    case Instruction::AShr:
      if (OpNo == 0)
        return AddBidirectionalEdge(UseNode, ValNode, SIGNEDNESS);
      break;

    case Instruction::Shl:
      if (OpNo == 0)
        return AddBidirectionalEdge(UseNode,
                                    ValNode,
                                    (SIGNEDNESS | UNSIGNEDNESS));
      break;

    case Instruction::Add: {
      bool Connected = false;
      Connected |= addSuccessorIfAbsent(UseNode, ValNode, ~FLOATNESS);
      Connected |= addSuccessorIfAbsent(ValNode,
                                        UseNode,
                                        ~(FLOATNESS | POINTERNESS
                                          | NUMBERNESS));
      return Connected;
      break;
    }

    case Instruction::Sub:
      return AddBidirectionalEdge(UseNode,
                                  ValNode,
                                  ~(FLOATNESS | POINTERNESS | NUMBERNESS));
      break;

    case Instruction::PHI:
    case Instruction::FPToUI:
    case Instruction::IntToPtr:
    case Instruction::PtrToInt:
    case Instruction::FPToSI:
    case Instruction::UIToFP:
    case Instruction::SIToFP:
    case Instruction::BitCast:
    case Instruction::SExt:
    case Instruction::ZExt:
    case Instruction::Trunc:
      return AddBidirectionalEdge(UseNode, ValNode, ~NUMBERNESS);
      break;

    case Instruction::Mul:
    case Instruction::And:
    case Instruction::Or:
    case Instruction::Xor: {
      auto Colors = ~(FLOATNESS | POINTERNESS | NUMBERNESS);
      if (not I->getType()->isIntegerTy(1))
        Colors &= ~BOOLNESS;
      return AddBidirectionalEdge(UseNode, ValNode, Colors);
    } break;

    // Freeze is transparent
    case Instruction::Freeze:
      return AddBidirectionalEdge(UseNode, ValNode, ALL_COLORS);
    }

    if (auto *Call = llvm::dyn_cast<CallInst>(I)) {
      if (isCallToIsolatedFunction(Call)) {
        // No type flows between arguments and return value
        return false;

      } else {
        auto *Callee = Call->getCalledFunction();
        revng_assert(Callee);

        // These opcodes are transparent
        if (FunctionTags::Parentheses.isTagOf(Callee)
            or FunctionTags::Copy.isTagOf(Callee))
          return AddBidirectionalEdge(UseNode, ValNode, ALL_COLORS);
      }
    }
  }

  return false;
}

TypeFlowGraph vma::makeTypeFlowGraphFromFunction(FunctionMetadataCache &Cache,
                                                 const llvm::Function *F,
                                                 const model::Binary *Model) {
  TypeFlowGraph TG;
  TG.Func = F;
  TG.Model = Model;

  // Visit values before users (a part from phis)
  for (const BasicBlock *BB : ReversePostOrderTraversal(F)) {
    for (const Instruction &I : *BB) {

      auto IsPhiInstr = [](const llvm::User *Inst) {
        return cast<Instruction>(Inst)->getOpcode() == Instruction::PHI;
      };

      const auto ShouldValueBeAdded = [](const llvm::Value *V) {
        return isa<Instruction>(V) or isa<Argument>(V);
      };

      const auto ShouldUseBeAdded = [&ShouldValueBeAdded](const llvm::Use *U) {
        return ShouldValueBeAdded(U->get());
      };

      if (TGLog.isEnabled()) {
        std::string S;
        llvm::raw_string_ostream OS(S);
        I.printAsOperand(OS);
        revng_log(TGLog, "VISITING: [" << &I << "]  " << S);
      }

      // Add the Value node
      TypeFlowNode *InstNode;
      if (TG.ContentToNodeMap.count(&I)) {
        // If the instruction has already been added, it must be because of a
        // phi
        revng_assert(any_of(I.users(), IsPhiInstr));
        InstNode = TG.ContentToNodeMap[&I];
      } else if (ShouldValueBeAdded(&I)) {
        InstNode = TG.addNodeContaining(Cache, &I);
      } else {
        // Skip values that should not be added to the TypeFlowGraph
        continue;
      }

      revng_log(TGLog, "INST: " << InstNode);

      // Add a Use node for each operand
      SmallVector<TypeFlowNode *, 2> PrevOperands;
      for (const Use &Op : I.operands()) {
        if (not ShouldUseBeAdded(&Op))
          continue;

        TypeFlowNode *UseNode = TG.addNodeContaining(Cache, &Op);
        connect(UseNode, InstNode);
        revng_log(TGLog, "USE: " << UseNode);

        // Connect Uses to Uses
        for (auto *Prev : PrevOperands)
          connect(UseNode, Prev);

        PrevOperands.push_back(UseNode);

        // The operand value should have already been visited, unless the
        // instruction is a phi or the operand is a non-instruction
        // (constant, global, arg).
        if (not TG.ContentToNodeMap.count(Op.get())) {
          revng_assert(I.getOpcode() == Instruction::PHI
                       or not isa<Instruction>(Op.get()));
          TG.addNodeContaining(Cache, Op.get());
        }

        auto *OpValNode = TG.ContentToNodeMap[Op.get()];
        revng_log(TGLog, "OP_VAL: " << OpValNode);

        // Connect Operand Use to the Instruction's Value
        connect(UseNode, OpValNode);
      }
    }
  }

  return TG;
}

void vma::propagateColors(TypeFlowGraph &TG) {
  propagateColor<POINTERNESS>(TG);
  propagateColor<SIGNEDNESS>(TG);
  propagateColor<UNSIGNEDNESS>(TG);
  propagateColor<BOOLNESS>(TG);
  propagateColor<FLOATNESS>(TG);
  // Numberness is propagated separately by propagateNumberness(), since it
  // has to be propagated through pattern matching
}

template<unsigned Filter>
void vma::propagateColor(TypeFlowGraph &TG) {
  // Check that only one color at a time is being propagated
  revng_assert(ColorSet(Filter).countValid() == 1);

  llvm::df_iterator_default_set<TypeFlowNode *> Visited;

  for (auto *Node : TG.nodes()) {
    bool AlreadyVisited = (Visited.find(Node) != Visited.end());
    // Start from nodes that have only the desired color
    if (AlreadyVisited or not Node->getCandidates().contains(ColorSet(Filter)))
      continue;

    // Explore only the edges that have the desired color
    for (TypeFlowNode *Reachable :
         llvm::depth_first_ext(EdgeFilteredTG<Filter>(Node), Visited)) {
      // If a node is reachable from a source with a certain color through edges
      // that all have that color, by construction it should also accept that
      // color
      revng_assert(Reachable->getAccepted().contains(Filter));

      auto InitialColor = Reachable->getCandidates();
      InitialColor.addColor(Filter);
      Reachable->setCandidates(InitialColor);
    }
  }
}

bool vma::propagateNumberness(TypeFlowGraph &TG) {
  // TODO: backward propagate pointerness for known patterns (e.g. value + const
  // = ptr)
  // TODO: forward propagate numberness for known patterns (e.g. n+n = n )

  // For now, just reset all numberness flags
  for (auto *N : TG.nodes()) {
    auto InitialColor = N->getCandidates();
    InitialColor.Bits.reset(NUMBERNESS_INDEX);
    N->setCandidates(InitialColor);
  }

  return false;
}

void vma::makeBidirectional(TypeFlowGraph &TG) {
  // Add each node as a successor of its successors
  for (TypeFlowNode *N : TG.nodes()) {
    llvm::SmallVector<TypeFlowNode *, 2> ToModify;

    for (auto *Succ : N->successors())
      if (not llvm::is_contained(Succ->successors(), N))
        ToModify.push_back(Succ);

    for (auto *M : ToModify)
      M->addSuccessor(N);
  }

  // Verify that predecessors and successors are the same in all nodes
  if (VerifyLog.isEnabled()) {
    auto AllSuccessorsArePredecessors = [](TypeFlowNode *N) {
      const auto IsPredecessorOfN = [N](TypeFlowNode *N2) {
        return llvm::is_contained(N->predecessors(), N2);
      };

      return llvm::all_of(N->successors(), IsPredecessorOfN);
    };

    revng_assert(llvm::all_of(TG.nodes(), AllSuccessorsArePredecessors));
  }
}

unsigned vma::countCasts(const TypeFlowGraph &TG) {
  llvm::SmallSet<const TypeFlowNode *, 16> Visited;
  unsigned Cost = 0;

  for (const TypeFlowNode *TGNode : TG.nodes()) {
    auto NodeBits = TGNode->getCandidates().Bits;
    // Ignore nodes with no candidates
    if (NodeBits.count() == 0 or NodeBits == NUMBERNESS)
      continue;

    for (const TypeFlowNode *Succ : TGNode->successors()) {
      // Ignore already visited nodes
      if (Visited.count(Succ))
        continue;

      auto SuccBits = Succ->getCandidates().Bits;

      // If they have no common candidates it means there's a cast
      if ((SuccBits & NodeBits) == 0)
        Cost += 1;
    }

    Visited.insert(TGNode);
  }

  return Cost;
}

/// If the majority of the neighbors agree on a color, return it
///
/// Under certain conditions, we can color a node only looking at its decided
/// neighbors, i.e. those neighbors that are colored with exactly one color.
/// In particular, if the majority of the decided neighbors agree on a color,
/// and the node can be colored with that color, we are sure that the node
/// should be colored with that color.
/// The voting works as follows: if the number of decided neighbors who agree on
/// a color is such that it would remain the most popular color even if all the
/// undecided nodes were to be assigned to any other color, then the color wind
/// the majority voting.
static std::optional<ColorSet> majorityVote(const TypeFlowNode *Node) {
  // Don't try to assign a color to an already decided or uncolored node
  revng_assert(Node->isUndecided());

  /// Holds a counter for a given color
  struct ColorFrequency {
    ColorSet Color;
    unsigned Frequency;

    ColorFrequency() = delete;
    ColorFrequency(ColorIndex Idx) : Color(1 << Idx), Frequency(0) {}
  };

  // Create a counter for each color
  llvm::SmallVector<ColorFrequency, MAX_COLORS> ColorCounters;

  for (size_t I = 0; I < MAX_COLORS; ++I)
    ColorCounters.push_back(ColorIndex(I));

  // For each color, count the number of decided neighbors with that color
  int NUndecided = 0;
  for (const TypeFlowNode *Succ : Node->successors()) {
    const ColorSet SuccColor = Succ->getCandidates();

    if (Succ->isDecided() and Node->getCandidates().contains(SuccColor)) {
      // Since the node is decided, its color has exactly one set bit
      ColorIndex I = SuccColor.firstSetBit();
      // The index of the set bit corresponds to the color
      ColorCounters[I].Frequency += 1;
    } else if (Succ->isUndecided()) {
      NUndecided++;
    }
  }

  auto SortByFrequency = [](ColorFrequency &CF1, ColorFrequency &CF2) {
    return CF1.Frequency > CF2.Frequency;
  };

  llvm::sort(ColorCounters, SortByFrequency);

  ColorFrequency Max = ColorCounters[0];
  ColorFrequency SecondMax = ColorCounters[1];

  // If the most common color among the decided neighbors is still the most
  // common even if all the undecided nodes are colored with the second most
  // common color, then we have a winner.
  if (Max.Frequency > (SecondMax.Frequency + NUndecided)) {
    revng_assert(Node->getCandidates().contains(Max.Color));
    return Max.Color;
  }

  return {};
}

bool vma::applyMajorityVoting(TypeFlowGraph &TG) {
  bool Modified = false;

  do {
    Modified = false;

    for (TypeFlowNode *N : TG.nodes()) {
      if (N->isUndecided()) {
        // Check if the neighbors of a node agree on a certain color
        if (auto VotedColor = majorityVote(N)) {
          N->setCandidates(*VotedColor);
          Modified = true;
        }
      }
    }
  } while (Modified);

  return Modified;
}
