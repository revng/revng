//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

#include <map>

#include "llvm/IR/Argument.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"

#include "revng/Model/Binary.h"
#include "revng/Model/PrimitiveType.h"
#include "revng/Model/QualifiedType.h"
#include "revng/Model/TypeKind.h"
#include "revng/Support/Assert.h"

#include "revng-c/Support/ModelHelpers.h"
#include "revng-c/ValueManipulationAnalysis/TypeColors.h"
#include "revng-c/ValueManipulationAnalysis/VMAPipeline.h"

#include "Mincut.h"
#include "TypeFlowGraph.h"
#include "TypeFlowNode.h"

using namespace vma;
using llvm::AllocaInst;
using llvm::Argument;
using llvm::BranchInst;
using llvm::Constant;
using llvm::GlobalValue;
using llvm::ICmpInst;
using llvm::Instruction;
using llvm::LoadInst;
using llvm::SelectInst;
using llvm::StoreInst;
using llvm::Value;

using model::QualifiedType;
using model::Qualifier;

// --------------- Initializers

void TypeMapInitializer::initializeColors(vma::TypeFlowGraph *TFG) {
  for (TypeFlowNode *N : TFG->nodes()) {
    // Initialize only value nodes that don't have a color yet
    if (N->isValue() and N->getCandidates() == NO_COLOR
        and N->getAccepted() != NO_COLOR) {
      if (auto It = TypeMap.find(N->getValue()); It != TypeMap.end()) {
        ColorSet InitialColor = QTToColor(It->second);

        // Add the color corresponding to the type in the TypeMap only if it's
        // decided - i.e. it correspond to a single color of the VMA (which is
        // turn corresponds to a specific type in C). This is done because, in
        // case we don't have strong type information in the model, e.g. we
        // have a generic64_t type, we want the VMA to be able to find out the
        // best candidates. On the other hand, this information will be
        // available in the Accepted colors, which list all the possible
        // candidates that can be assigned toa given node.
        if (InitialColor.countValid() == 1)
          N->setCandidates(InitialColor);
      }
    }
  }
}

/// Returns the default candidates associated with an LLVM value or use, which
/// we can infer directly by only looking at the value (or user in case of
/// uses).
static ColorSet getInitialCandidates(const UseOrValue &Content) {
  // Arguments, constants, globals etc.
  if (isValue(Content)) {
    const Value *V = getValue(Content);

    if (isa<Argument>(V))
      return NO_COLOR;

    // Constants and globals should not be infected, since they don't belong to
    // a single function
    if (isa<Constant>(V) or isa<GlobalValue>(V))
      return NO_COLOR;

    if (not isa<Instruction>(V))
      return NO_COLOR;
  }

  // Instructions and operand uses should be the only thing remaining
  bool IsContentInst = isInst(Content);
  revng_assert(IsContentInst or isUse(Content));

  // If the content of the node is an Instruction's Value, assign colors
  // based on the instruction's opcode. Otherwise, if we are creating a node for
  // one of the operands, find which the user of the operand and check its
  // opcode.
  const Instruction *I = IsContentInst ?
                           cast<Instruction>(getValue(Content)) :
                           cast<Instruction>(getUse(Content)->getUser());

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
      return UNSIGNEDNESS;
    break;

  case Instruction::SDiv:
  case Instruction::SRem:
    if (IsContentInst)
      return SIGNEDNESS | NUMBERNESS;
    else
      return SIGNEDNESS;
    break;

  case Instruction::UDiv:
  case Instruction::URem:
    if (IsContentInst)
      return UNSIGNEDNESS | NUMBERNESS;
    else
      return UNSIGNEDNESS;
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
    if (IsContentInst)
      return SIGNEDNESS | NUMBERNESS;
    if (getOpNo(Content) == 0)
      return SIGNEDNESS;
    if (getOpNo(Content) == 1)
      return UNSIGNEDNESS;
    break;

  case Instruction::LShr:
    if (IsContentInst)
      return UNSIGNEDNESS | NUMBERNESS;
    else
      return UNSIGNEDNESS;
    break;

  case Instruction::Shl:
    if (IsContentInst or getOpNo(Content) == 0)
      return NO_COLOR;
    if (getOpNo(Content) == 1)
      return UNSIGNEDNESS;
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

  case Instruction::Trunc:
    return NO_COLOR;
    break;

  case Instruction::And:
  case Instruction::Or:
  case Instruction::Xor:
  case Instruction::Mul:
    if (IsContentInst)
      return NUMBERNESS;
    else
      return NO_COLOR;
    break;

  case Instruction::GetElementPtr:
    revng_abort("Didn't expect to find a GEP here");
    break;
  }

  return NO_COLOR;
}

void LLVMInitializer::initializeColors(vma::TypeFlowGraph *TFG) {
  for (TypeFlowNode *N : TFG->nodes())
    if (N->getCandidates() == NO_COLOR and N->getAccepted() != NO_COLOR)
      N->setCandidates(getInitialCandidates(N->getContent()));
}

// --------------- Updaters

static model::QualifiedType
upgradePrimitiveType(const model::QualifiedType &OldQT,
                     ColorSet VMAResult,
                     const model::Binary *Model) {
  // Upgrade only primitive types
  if (not OldQT.is(model::TypeKind::PrimitiveType))
    return OldQT;

  // The primitive type could be nested into an arbitrary chain of typedefs.
  // Since we passed the `Qualifier::is` function, we are sure that there are
  // only const qualifiers, if any. Now we need to check if there's actually at
  // least one or not.
  model::QualifiedType ActualQT = OldQT;
  revng_assert(llvm::all_of(ActualQT.Qualifiers(), Qualifier::isConst));

  bool IsConstant = false;

  auto *UnqualT = ActualQT.UnqualifiedType().getConst();
  while (auto *TypedefT = llvm::dyn_cast<model::TypedefType>(UnqualT)) {
    ActualQT = TypedefT->UnderlyingType();

    if (not ActualQT.Qualifiers().empty()) {
      revng_assert(llvm::all_of(ActualQT.Qualifiers(), Qualifier::isConst));
      IsConstant = true;
    }
    UnqualT = ActualQT.UnqualifiedType().getConst();
  }

  revng_assert(UnqualT->Kind() == model::TypeKind::PrimitiveType);

  // Transform original kind into a canonical color
  ColorSet OldColor = QTToColor(ActualQT);
  // Transform the color recovered by the VMA into a canonical color
  ColorSet FinalColor = getNearestModelColor(VMAResult);

  // Check if the VMA results are compatible with the original type
  if (OldColor.contains(FinalColor) and not(FinalColor == NO_COLOR)) {
    auto &FinalKind = ColorToPrimitiveType.at(FinalColor);
    auto OriginalSize = *UnqualT->size();
    QualifiedType FinalQT(Model->getPrimitiveType(FinalKind, OriginalSize), {});

    if (IsConstant)
      FinalQT.Qualifiers().push_back(model::Qualifier::createConst());
  }

  return OldQT;
}

void TypeMapUpdater::updateWithResults(const vma::TypeFlowGraph *TFG) {
  // Update type only for values for which we have recovered a rich type
  for (const TypeFlowNode *N : TFG->nodes()) {
    if (N->isValue() and N->getCandidates() != NO_COLOR) {
      if (auto It = TypeMap.find(N->getValue()); It != TypeMap.end()) {
        const Value *Key = It->first;
        const auto UpgradedQT = upgradePrimitiveType(It->second,
                                                     N->getCandidates(),
                                                     Model);
        It = TypeMap.erase(It);
        TypeMap.emplace_hint(It, Key, std::move(UpgradedQT));
      }
    }
  }
}

// --------------- Pipeline

void VMAPipeline::runSolver() {
  // Mincut preprocessing
  makeBidirectional(*TFG);
  applyMajorityVoting(*TFG);

  // Assign grey nodes using Karger
  if (llvm::any_of(TFG->nodes(),
                   [](vma::TypeFlowNode *N) { return N->isUndecided(); }))
    minCut(*TFG);
}

void VMAPipeline::run(FunctionMetadataCache &Cache, const llvm::Function *F) {
  // Nodes and edges initialization
  auto TypeGraph = vma::makeTypeFlowGraphFromFunction(Cache, F, &Model);
  TFG = &TypeGraph;

  // Color initialization
  for (auto &Init : Initializers)
    Init->initializeColors(TFG);

  // Propagation
  vma::propagateColors(*TFG);

  // Constraint solving
  if (UseSolver)
    runSolver();

  // Finalization
  if (Updater)
    Updater->updateWithResults(TFG);
}
