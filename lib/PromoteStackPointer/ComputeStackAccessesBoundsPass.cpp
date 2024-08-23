//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <optional>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/LazyValueInfo.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/ConstantRange.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instructions.h"

#include "revng/Model/IRHelpers.h"
#include "revng/Support/IRHelpers.h"

#include "revng-c/PromoteStackPointer/ComputeStackAccessesBoundsPass.h"
#include "revng-c/PromoteStackPointer/InstrumentStackAccessesPass.h"
#include "revng-c/Support/FunctionTags.h"

using namespace llvm;

static Logger<> Log("compute-stack-accesses-bounds");

template<typename T, std::ranges::range R>
bool areAll(const R &Range) {
  return llvm::all_of(Range, [](auto *Object) { return isa<T>(Object); });
}

static ConstantRange
getSCEVBoundaries(ScalarEvolution &SE, PostDominatorTree &PDT, Instruction *I) {
  const auto *SCEV = SE.getSCEV(I);
  auto FullSet = ConstantRange::getFull(SCEV->getType()->getIntegerBitWidth());

  revng_log(Log, "getSCEVBoundaries on " << dumpToString(*SCEV));
  LoggerIndent<> Indent(Log);

  auto *AddRec = dyn_cast<SCEVAddRecExpr>(SCEV);
  if (AddRec == nullptr)
    return FullSet;

  // Only consider loop in loop simplify form
  auto *Loop = AddRec->getLoop();
  if (not Loop->isLoopSimplifyForm())
    return FullSet;

  // Only provide a result if the block post-dominates the header, which
  // implies the block sees one more value than the backedge taken count
  BasicBlock *Location = I->getParent();
  if (not PDT.dominates(Location, Loop->getHeader()))
    return FullSet;

  // Collect operands
  // TODO: handle multiple AddRecs
  const auto *Start = AddRec->getStart();
  const class SCEV *End = nullptr;
  auto *BackedgeTakenCount = SE.getBackedgeTakenCount(Loop);
  const auto *Stride = AddRec->getStepRecurrence(SE);

  if (not areAll<SCEVConstant>(std::vector{
        Start, BackedgeTakenCount, Stride })) {
    return FullSet;
  }
  bool Increasing = cast<SCEVConstant>(Stride)->getAPInt().isStrictlyPositive();
  revng_log(Log, "Increasing: " << Increasing);

  if (Increasing) {
    // Compute end value
    llvm::SmallVector<const llvm::SCEV *, 4> Addends = {
      Start,
      SE.getMulExpr(Stride, BackedgeTakenCount),
      SE.getOne(Start->getType())
    };
    End = SE.getAddExpr(Addends);
  } else {
    End = SE.getAddExpr(Start, SE.getOne(Start->getType()));
    Start = SE.getAddExpr(Start, SE.getMulExpr(Stride, BackedgeTakenCount));
  }

  revng_log(Log,
            "Start:" << dumpToString(*Start) << " End: " << dumpToString(*End));

  // Ensure bounds could be computed and are constant
  if (not areAll<SCEVConstant>(std::vector{ Start, End }))
    return FullSet;

  APInt StartValue = cast<SCEVConstant>(Start)->getAPInt();
  APInt EndValue = cast<SCEVConstant>(End)->getAPInt();

  ConstantRange Result(StartValue, EndValue);

  if (EndValue.slt(StartValue))
    return FullSet;

  revng_log(Log, "Result: " << dumpToString(Result));

  return Result;
}

bool ComputeStackAccessesBoundsPass::runOnFunction(Function &F) {
  revng_log(Log, "Running on " << F.getName().str());
  LoggerIndent<> Indent(Log);

  ScalarEvolution &SE = getAnalysis<ScalarEvolutionWrapperPass>().getSE();
  PostDominatorTree &PDT = getAnalysis<PostDominatorTreeWrapperPass>()
                             .getPostDomTree();
  LazyValueInfo &LVI = getAnalysis<LazyValueInfoWrapperPass>().getLVI();

  if (F.isDeclaration())
    return false;

  for (Function &StackOffsetFunction :
       FunctionTags::StackOffsetMarker.functions(F.getParent())) {

    auto *FunctionType = StackOffsetFunction.getFunctionType();
    auto *DifferenceType = cast<IntegerType>(FunctionType->getParamType(1));

    for (CallBase *Call : callersIn(&StackOffsetFunction, &F)) {
      auto *Undef = UndefValue::get(DifferenceType);

      auto ReplaceUseWithBound = [&](Use &Operand, bool Lower) {
        revng_log(Log,
                  "ReplaceUseWithBound on: " << getName(Operand.get())
                                             << " (Lower: <<" << Lower << ")");

        auto *User = Operand.getUser();
        auto *V = Operand.get();
        const APInt *Bound = nullptr;

        // Get range from LVI
        // TODO: using LVI is inaccurate, it offers an upper (lower) bound not
        //       an exact bound.
        //       Note that this is unlike using SCEV, since we use it in a way
        //       that ensures the upper bound is actually used.
        //       Eventually we'll need to deal with this incoherence and either
        //       restrict the usage of LVI or drop it entirely.
        const auto &BoundRange = LVI.getConstantRange(V, cast<CallInst>(User));
        revng_log(Log, "LazyValueInfo says " << dumpToString(BoundRange));

        if (not BoundRange.isFullSet() and not BoundRange.isWrappedSet()) {
          Bound = Lower ? &BoundRange.getLower() : &BoundRange.getUpper();
        }

        if (auto *I = dyn_cast<Instruction>(V)) {
          // See what SCEV says
          auto SCEVRange = getSCEVBoundaries(SE, PDT, I);

          APInt SCEVBound;
          if (Lower) {
            SCEVBound = SCEVRange.getLower();
            if (not SCEVBound.isMinSignedValue()
                and (Bound == nullptr or SCEVBound.sgt(*Bound))) {
              Bound = &SCEVBound;
            }
          } else {
            SCEVBound = SCEVRange.getUpper();
            if (not SCEVBound.isMaxSignedValue()
                and (Bound == nullptr or SCEVBound.slt(*Bound))) {
              Bound = &SCEVBound;
            }
          }
        }

        Operand.set(Bound != nullptr ?
                      ConstantInt::get(DifferenceType, *Bound) :
                      Undef);
      };

      ReplaceUseWithBound(Call->getArgOperandUse(1), true);
      ReplaceUseWithBound(Call->getArgOperandUse(2), false);
    }
  }

  return true;
}

void ComputeStackAccessesBoundsPass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<LazyValueInfoWrapperPass>();
  AU.addRequired<ScalarEvolutionWrapperPass>();
  AU.addRequired<PostDominatorTreeWrapperPass>();
  AU.setPreservesCFG();
}

char ComputeStackAccessesBoundsPass::ID = 0;

using RegisterCSAB = RegisterPass<ComputeStackAccessesBoundsPass>;
static RegisterCSAB R("compute-stack-accesses-bounds",
                      "Compute Stack Accesses Bounds Pass");
