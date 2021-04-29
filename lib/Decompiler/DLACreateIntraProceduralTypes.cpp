//
// Copyright (c) rev.ng Srls. See LICENSE.md for details.
//

#include <optional>
#include <utility>

#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"

#include "revng/Support/Debug.h"
#include "revng/Support/FunctionTags.h"
#include "revng/Support/IRHelpers.h"

#include "DLAHelpers.h"
#include "DLAStep.h"
#include "DLATypeSystem.h"
#include "SCEVBaseAddressExplorer.h"

using namespace dla;
using namespace llvm;

using StepT = CreateIntraproceduralTypes;

// Returns true if an Instruction must forcibly be serialized.
//
// This is not implemented yet now, but it could be that an Instruction is
// forcibly serialized if it has more than one use, or if an end user of the
// decompiler decides that it must be serialized and have its own dedicated
// local variable.
//
// In general, in order to serialize an Instruction we need a type for it, so we
// will add a node in the LayoutTypeSystem for it.
static bool mustBeSerialized(const Instruction &) {
  return false;
}

using LayoutTypeSystemNode = dla::LayoutTypeSystemNode;
using SCEVTypeMekerMap = std::map<const SCEV *, uint64_t>;
using SCEVTypeMap = SCEVBaseAddressExplorer::SCEVTypeMap;

static int64_t getSCEVConstantSExtVal(const SCEV *S) {
  return cast<SCEVConstant>(S)->getAPInt().getSExtValue();
}

class InstanceLinkAdder {
  Function *F;
  ScalarEvolution *SE;
  llvm::DominatorTree DT;
  llvm::PostDominatorTree PDT;

  SCEVTypeMap SCEVToLayoutType;

protected:
  bool addInstanceLink(LayoutTypeSystem &TS,
                       Value *PointerVal,
                       const SCEV *BaseAddrSCEV,
                       const BasicBlock &B) {
    revng_assert(PointerVal != nullptr);
    revng_assert(isa<IntegerType>(PointerVal->getType())
                 or isa<PointerType>(PointerVal->getType()));
    revng_assert(B.getParent() == F);
    bool Created = false; // Created LayoutTypeSystemNode, or Link

    LayoutTypeSystemNode *Src = nullptr;
    {
      // Check if the SCEV associated to the base address already has an
      // associated LayoutTypeSystemNode.
      // If it has, we want don't need to create a new node in TS for the source
      // of the instance link, and we can add the instance link directly from
      // the type of the base address link.
      auto It = SCEVToLayoutType.lower_bound(BaseAddrSCEV);
      if (It != SCEVToLayoutType.end()
          and not SCEVToLayoutType.key_comp()(BaseAddrSCEV, It->first)) {
        Src = &*It->second;
      } else if (auto *U = dyn_cast<SCEVUnknown>(BaseAddrSCEV)) {
        // If the BaseAddrSCEV doesn't have an associated type, we want to
        // create it and add it.
        Value *BaseAddr = U->getValue();
        revng_assert(nullptr != BaseAddr);
        const auto &[Layout, NewType] = TS.getOrCreateLayoutType(BaseAddr);
        Created |= NewType;
        auto P = std::make_pair(BaseAddrSCEV, Layout);
        Src = SCEVToLayoutType.emplace_hint(It, std::move(P))->second;
      } else {
        // If BaseAddrSCEV is not typed and it does not refer to a global
        // variable we cannot go on.
        return Created;
      }
    }
    revng_assert(Src != nullptr);

    const auto &[Tgt, IsNewType] = TS.getOrCreateLayoutType(PointerVal);
    Created |= IsNewType;
    revng_assert(Tgt != nullptr);

    revng_assert(Src != Tgt or BaseAddrSCEV == SE->getSCEV(PointerVal));
    if (Src == Tgt)
      return Created;

    const SCEV *PointerValSCEV = SE->getSCEV(PointerVal);
    Type *PointerType = PointerValSCEV->getType();
    if (BaseAddrSCEV->getType() != PointerType) {
      BaseAddrSCEV = SE->getZeroExtendExpr(BaseAddrSCEV, PointerType);
    }

    const SCEV *NegBaseAddrSCEV = SE->getNegativeSCEV(BaseAddrSCEV);
    const SCEV *OffsetSCEV = SE->getAddExpr(NegBaseAddrSCEV, PointerValSCEV);

    // For now we only support constant offsets and recurring expressions
    // representing arrays
    if (not isa<SCEVConstant>(OffsetSCEV)
        and not isa<SCEVAddRecExpr>(OffsetSCEV))
      return Created;

    OffsetExpression OE{};
    while (isa<SCEVAddRecExpr>(OffsetSCEV)) {
      const auto *Rec = cast<SCEVAddRecExpr>(OffsetSCEV);
      const SCEV *StrideExpr = Rec->getStepRecurrence(*SE);
      auto StrideValue = getSCEVConstantSExtVal(StrideExpr);

      // Don't add links for recurring expressions with non-positive strides.
      if (StrideValue <= 0LL)
        return Created;

      OE.Strides.push_back(StrideValue);
      const Loop *L = Rec->getLoop();
      revng_assert(L != nullptr);
      std::optional<int64_t> TripCount;
      if (L->isLoopSimplifyForm()) {
        // If the loop is simplified, use getBackedgeTakenCount to infer the
        // trip count.
        const SCEV *SCEVBackedgeCount = SE->getBackedgeTakenCount(L);
        if (auto *Count = dyn_cast<SCEVConstant>(SCEVBackedgeCount)) {
          SmallVector<BasicBlock *, 4> ExitBlocks;
          L->getUniqueExitBlocks(ExitBlocks);
          const auto IsDominatedByB = [&DT = this->DT,
                                       &B](const BasicBlock *OtherB) {
            return DT.dominates(&B, OtherB);
          };
          if (std::all_of(ExitBlocks.begin(),
                          ExitBlocks.end(),
                          IsDominatedByB)) {
            // If B (where the memory access is) dominates all the exit
            // blocks, then B is executed the same number of times as the
            // loop header.
            // This number is the trip count of the loop, which in
            // loop-simplified form is SCEVBackedgeCount + 1, because in
            // loop-simplified form we only have one back edge.
            TripCount = Count->getAPInt().getSExtValue() + 1;
          } else if (PDT.dominates(L->getHeader(), &B)) {
            // If the loop header postdominates B, B is executed the same
            // number of times as the only backedge
            TripCount = Count->getAPInt().getSExtValue();
          } // In all the other cases we know nothing
        }
      } else {
        // If the loop is not simplified, getBackedgeTakenCount may give some
        // results, but not enough to reliably infer the trip count.
        // Just set it as missing and keep going.
      }

      // Don't add links for recurring expressions with negative trip counts.
      if (TripCount.has_value() and TripCount.value() < 0LL)
        return Created;

      OE.TripCounts.push_back(std::move(TripCount));
      OffsetSCEV = Rec->getStart();
    }

    // For now we do not support offsets that are not constant.
    if (not isa<SCEVConstant>(OffsetSCEV))
      return Created;

    // Don't add links for instances at negative offsets.
    OE.Offset = getSCEVConstantSExtVal(OffsetSCEV);
    if (OE.Offset < 0LL)
      return Created;

    Created |= TS.addInstanceLink(Src, Tgt, std::move(OE)).second;
    return Created;
  }

public:
  void setupForProcessingFunction(ModulePass *MP, Function *TheF) {
    SE = &MP->getAnalysis<llvm::ScalarEvolutionWrapperPass>(*TheF).getSE();
    F = TheF;
    DT.recalculate(*F);
    PDT.recalculate(*F);
    SCEVToLayoutType.clear();
  }

  bool getOrCreateSCEVTypes(LayoutTypeSystem &TS) {
    bool Changed = false;

    // Add entry in SCEVToLayoutType map for arguments. We always add these
    // because here F is always an isolated function.
    for (Argument &A : F->args()) {
      revng_assert(isa<IntegerType>(A.getType())
                   or isa<PointerType>(A.getType()));
      LayoutTypeSystemNode *ArgLayout = TS.getLayoutType(&A);
      const SCEV *S = SE->getSCEV(&A);
      SCEVToLayoutType.insert(std::make_pair(S, ArgLayout));
    }

    for (BasicBlock &B : *F) {
      for (auto &I : B) {
        // Add entry in SCEVToLayoutType map for values returned by F
        if (auto *RetI = dyn_cast<ReturnInst>(&I)) {
          if (Value *RetVal = RetI->getReturnValue()) {
            revng_assert(isa<StructType>(RetVal->getType())
                         or isa<IntegerType>(RetVal->getType())
                         or isa<PointerType>(RetVal->getType()));
            if (isa<StructType>(RetVal->getType())) {
              auto RetTys = TS.getLayoutTypes(*RetVal);
              auto NRetTypes = RetTys.size();
              revng_assert(NRetTypes > 1ULL);

              // If RetVal is a ConstantAggregate we cannot infer anything about
              // type layouts right now. We need to handle layout pointed to by
              // constant addresses first. This might be useful to infer types
              // in data sections of binaries be we don't handle it now. When we
              // do, it will become necessary to handle this case.
              if (isa<ConstantAggregate>(RetVal)
                  or isa<ConstantAggregateZero>(RetVal))
                continue;

              if (isa<UndefValue>(RetVal))
                continue;

              if (auto *Call = dyn_cast<CallInst>(RetVal)) {
                const Function *Callee = getCallee(Call);
                auto CTags = FunctionTags::TagsSet::from(Callee);
                revng_assert(CTags.contains(FunctionTags::StructInitializer));

                revng_assert(not Callee->isVarArg());

                auto *RetTy = cast<StructType>(Callee->getReturnType());
                revng_assert(RetTy->getNumElements() == Callee->arg_size());
                revng_assert(RetTy == F->getReturnType());

                auto StructTypeNodes = TS.getOrCreateLayoutTypes(*Call);
                revng_assert(StructTypeNodes.size() == Callee->arg_size());

                for (const auto &[RetNodeNew, Arg] :
                     llvm::zip_first(StructTypeNodes, Call->arg_operands())) {
                  const auto &[ArgNode, New] = TS.getOrCreateLayoutType(Arg);
                  Changed |= New;
                  const auto &[RetNode, NewNode] = RetNodeNew;
                  Changed |= NewNode;
                  Changed |= TS.addEqualityLink(RetNode, ArgNode).second;
                }

                continue;
              }

              auto *InsertVal = cast<InsertValueInst>(RetVal);
              auto RetOps = getInsertValueLeafOperands(InsertVal);
              revng_assert(RetOps.size() == NRetTypes);
              decltype(NRetTypes) N = 0ULL;
              for (; N < NRetTypes; ++N) {
                if (RetOps[N] == nullptr)
                  continue;
                const SCEV *S = SE->getSCEV(RetOps[N]);
                SCEVToLayoutType.insert(std::make_pair(S, RetTys[N]));
              }
            } else {
              LayoutTypeSystemNode *RetTy = TS.getLayoutType(RetVal);
              const SCEV *S = SE->getSCEV(RetVal);
              SCEVToLayoutType.insert(std::make_pair(S, RetTy));
            }
          }
        } else if (auto *PHI = dyn_cast<PHINode>(&I)) {
          // Booleans can not be addresses, so we can skip them.
          if (PHI->getType()->isIntegerTy(1))
            continue;

          revng_assert(isa<IntegerType>(PHI->getType())
                       or isa<PointerType>(PHI->getType()));
          {
            LayoutTypeSystemNode *PHIType = TS.getLayoutType(PHI);
            const SCEV *PHISCEV = SE->getSCEV(PHI);
            SCEVToLayoutType.insert(std::make_pair(PHISCEV, PHIType));
          }

          // PHI Incoming values
          for (Value *In : PHI->incoming_values()) {
            revng_assert(isa<IntegerType>(In->getType())
                         or isa<PointerType>(In->getType()));
            LayoutTypeSystemNode *InTy = TS.getLayoutType(In);
            const SCEV *InSCEV = SE->getSCEV(In);
            SCEVToLayoutType.insert(std::make_pair(InSCEV, InTy));
          }
        } else if (auto *Sel = dyn_cast<SelectInst>(&I)) {
          // Booleans can not be addresses, so we can skip them.
          if (Sel->getType()->isIntegerTy(1))
            continue;

          revng_assert(isa<IntegerType>(Sel->getType())
                       or isa<PointerType>(Sel->getType()));

          // Selects are very much like PHIs.
          const auto &[SelType, New] = TS.getOrCreateLayoutType(Sel);
          Changed |= New;
          const SCEV *SelSCEV = SE->getSCEV(Sel);
          SCEVToLayoutType.insert(std::make_pair(SelSCEV, SelType));

          // True incoming value
          {
            Value *TrueV = Sel->getTrueValue();
            revng_assert(isa<IntegerType>(TrueV->getType())
                         or isa<PointerType>(TrueV->getType()));
            const auto &[TrueTy, NewT] = TS.getOrCreateLayoutType(TrueV);
            Changed |= NewT;
            const SCEV *TrueSCEV = SE->getSCEV(TrueV);
            SCEVToLayoutType.insert(std::make_pair(TrueSCEV, TrueTy));
            Changed |= TS.addInheritanceLink(TrueTy, SelType).second;
          }

          // False incoming value
          {
            Value *FalseV = Sel->getFalseValue();
            revng_assert(isa<IntegerType>(FalseV->getType())
                         or isa<PointerType>(FalseV->getType()));
            const auto &[FalseTy, NewT] = TS.getOrCreateLayoutType(FalseV);
            Changed |= NewT;
            const SCEV *FalseSCEV = SE->getSCEV(FalseV);
            SCEVToLayoutType.insert(std::make_pair(FalseSCEV, FalseTy));
            Changed |= TS.addInheritanceLink(FalseTy, SelType).second;
          }

        } else if (auto *C = dyn_cast<CallInst>(&I)) {
          const Function *Callee = getCallee(C);

          // Skip llvm intrinsics
          if (Callee->isIntrinsic())
            continue;

          if (Callee->hasName()
              and Callee->getName() == "revng_init_local_sp") {
            const auto &[StackLayout, New] = TS.getOrCreateLayoutType(C);
            Changed |= New;
            const SCEV *CallSCEV = SE->getSCEV(C);
            SCEVToLayoutType.insert(std::make_pair(CallSCEV, StackLayout));
            continue;
          }

          auto CTags = FunctionTags::TagsSet::from(Callee);
          if (CTags.contains(FunctionTags::StructInitializer)) {

            revng_assert(not Callee->isVarArg());

            auto *RetTy = cast<StructType>(Callee->getReturnType());
            revng_assert(RetTy->getNumElements() == Callee->arg_size());

            auto StructTypeNodes = TS.getOrCreateLayoutTypes(*C);
            revng_assert(StructTypeNodes.size() == Callee->arg_size());

            for (const auto &[RetTypeNodeNew, Arg] :
                 llvm::zip_first(StructTypeNodes, C->arg_operands())) {
              const auto &[ArgTypeNode, New] = TS.getOrCreateLayoutType(Arg);
              Changed |= New;
              const auto &[RetTypeNode, NewNode] = RetTypeNodeNew;
              Changed |= NewNode;
              Changed |= TS.addEqualityLink(RetTypeNode, ArgTypeNode).second;
            }

            continue;
          }

          // Consider only isolated functions. We don't want to create types for
          // QEMU helpers or other nasty functions.
          // In particular, QEMU helpers are used to implement specific CPU
          // instructions, and typically take as input argument either @env or
          // CPU State Variables representing registers. This is bad for two
          // main reasons:
          //  1. They tend to collapse different computations, possibly on stuff
          //     with different types, on the same LayoutTypeSystemNode.
          //  2. @env is not a construct coming from the original program being
          //     decompiled, rather a QEMU artifact that represents the CPU
          //     state. Hence it has no really meaningful type in the program.
          if (not CTags.contains(FunctionTags::Lifted))
            continue;

          revng_assert(not Callee->isVarArg());
          revng_assert(isa<PointerType>(Callee->getType()));
          const auto *PointerT = cast<PointerType>(Callee->getType());
          const auto *FunctionT = PointerT->getPointerElementType();
          revng_assert(isa<FunctionType>(FunctionT));

          // Add entry in SCEVToLayoutType map for return values of CallInst
          if (C->getNumUses()) {
            // Return values
            revng_assert(isa<StructType>(C->getType())
                         or isa<IntegerType>(C->getType())
                         or isa<PointerType>(C->getType()));

            if (isa<StructType>(C->getType())) {
              // Types representing the return type
              auto FormalRetTys = TS.getLayoutTypes(*Callee);
              auto Size = FormalRetTys.size();
              auto ExtractedVals = getExtractedValuesFromCall(C);
              revng_assert(Size == ExtractedVals.size());
              for (const auto &[Ext, RetTy] :
                   llvm::zip(ExtractedVals, FormalRetTys)) {

                if (Ext.empty())
                  continue;

                for (llvm::ExtractValueInst *E : Ext) {
                  revng_assert(E);
                  llvm::Type *ExtTy = E->getType();
                  revng_assert(isa<IntegerType>(ExtTy)
                               or isa<PointerType>(ExtTy));

                  const auto &[ExtLayout, New] = TS.getOrCreateLayoutType(E);
                  Changed |= New;
                  Changed |= TS.addEqualityLink(RetTy, ExtLayout).second;
                  const SCEV *S = SE->getSCEV(E);
                  SCEVToLayoutType.insert(std::make_pair(S, ExtLayout));
                }
              }
            } else {
              // Type representing the return type
              revng_assert(not C->getType()->isIntegerTy(1));
              LayoutTypeSystemNode *RetTy = TS.getLayoutType(Callee);
              const auto &[CType, NewC] = TS.getOrCreateLayoutType(C);
              Changed |= NewC;
              Changed |= TS.addEqualityLink(RetTy, CType).second;
              const SCEV *RetS = SE->getSCEV(C);
              SCEVToLayoutType.insert(std::make_pair(RetS, CType));
            }
          }

          // Add entry in SCEVToLayoutType map for actual arguments of CallInst.
          for (Use &ArgU : C->arg_operands()) {
            revng_assert(isa<IntegerType>(ArgU->getType())
                         or isa<PointerType>(ArgU->getType()));
            const auto &[ArgTy, Created] = TS.getOrCreateLayoutType(ArgU);
            Changed |= Created;
            const SCEV *ArgS = SE->getSCEV(ArgU);
            SCEVToLayoutType.insert(std::make_pair(ArgS, ArgTy));
          }
        } else if (isa<LoadInst>(I) or isa<StoreInst>(I)) {
          Value *PointerOp(nullptr);
          if (auto *Load = dyn_cast<LoadInst>(&I))
            PointerOp = Load->getPointerOperand();
          else if (auto *Store = dyn_cast<StoreInst>(&I))
            PointerOp = Store->getPointerOperand();

          if (auto *CExpr = dyn_cast<ConstantExpr>(PointerOp)) {
            if (CExpr->isCast()) {
              bool IsIntToPtr = false;
              bool IsPtrToInt = false;
              bool IsBitCast = false;
              {
                auto *Cast = CExpr->getAsInstruction();
                IsIntToPtr = isa<PtrToIntInst>(Cast);
                IsPtrToInt = isa<IntToPtrInst>(Cast);
                IsBitCast = isa<BitCastInst>(Cast);
                // Cleanup, because getAsInstruction actually creates an
                // instruction not linked to any basic block.
                Cast->deleteValue();
              }
              if (IsIntToPtr or IsPtrToInt or IsBitCast) {
                Value *Op = CExpr->getOperand(0);
                if (isa<ConstantInt>(Op)) {

                  bool New = false;
                  LayoutTypeSystemNode *SrcLayout = nullptr;
                  LayoutTypeSystemNode *TgtLayout = nullptr;

                  std::tie(SrcLayout, New) = TS.getOrCreateLayoutType(Op);
                  Changed |= New;
                  std::tie(TgtLayout, New) = TS.getOrCreateLayoutType(CExpr);
                  Changed |= New;

                  Changed |= TS.addEqualityLink(SrcLayout, TgtLayout).second;

                  const SCEV *LoadSCEV = SE->getSCEV(CExpr);
                  SCEVToLayoutType.insert(std::make_pair(LoadSCEV, TgtLayout));
                }
              }
            }
          }

          if (auto *L = dyn_cast<LoadInst>(&I)) {
            revng_assert(isa<IntegerType>(L->getType())
                         or isa<PointerType>(L->getType()));

            revng_assert(not L->getType()->isIntegerTy(1));
            const auto &[LoadedTy, Created] = TS.getOrCreateLayoutType(L);
            Changed |= Created;
            const SCEV *LoadSCEV = SE->getSCEV(L);
            SCEVToLayoutType.insert(std::make_pair(LoadSCEV, LoadedTy));
          }
        } else if (auto *A = dyn_cast<AllocaInst>(&I)) {
          revng_assert(isa<IntegerType>(A->getType()->getElementType())
                       or isa<PointerType>(A->getType()->getElementType()));
          const auto &[LoadedTy, Created] = TS.getOrCreateLayoutType(A);
          Changed |= Created;
          const SCEV *LoadSCEV = SE->getSCEV(A);
          SCEVToLayoutType.insert(std::make_pair(LoadSCEV, LoadedTy));
        } else if (isa<IntToPtrInst>(&I) or isa<PtrToIntInst>(&I)
                   or isa<BitCastInst>(&I)) {
          Value *Op = I.getOperand(0);

          bool New = false;
          LayoutTypeSystemNode *SrcLayout = nullptr;
          LayoutTypeSystemNode *TgtLayout = nullptr;

          std::tie(SrcLayout, New) = TS.getOrCreateLayoutType(Op);
          Changed |= New;
          std::tie(TgtLayout, New) = TS.getOrCreateLayoutType(&I);
          Changed |= New;

          Changed |= TS.addEqualityLink(SrcLayout, TgtLayout).second;
          const SCEV *LoadSCEV = SE->getSCEV(&I);
          SCEVToLayoutType.insert(std::make_pair(LoadSCEV, TgtLayout));
        }
      }
    }
    return Changed;
  }

  bool createBaseAddrWithInstanceLink(LayoutTypeSystem &TS,
                                      Value *PointerVal,
                                      const BasicBlock &B) {
    revng_assert(nullptr != PointerVal);

    bool AddedSomething = false;

    // If PointerVal points to an undef, do nothing
    if (isa<UndefValue>(PointerVal))
      return AddedSomething;

    const SCEV *PtrSCEV = SE->getSCEV(PointerVal);
    using Explorer = SCEVBaseAddressExplorer;
    auto PossibleBaseAddresses = Explorer().findBases(SE,
                                                      PtrSCEV,
                                                      SCEVToLayoutType);
    for (const SCEV *BaseAddrSCEV : PossibleBaseAddresses)
      AddedSomething |= addInstanceLink(TS, PointerVal, BaseAddrSCEV, B);

    return AddedSomething;
  }
};

bool StepT::runOnTypeSystem(LayoutTypeSystem &TS) {
  bool Changed = false;
  InstanceLinkAdder ILA;
  Module &M = TS.getModule();
  for (Function &F : M.functions()) {
    auto FTags = FunctionTags::TagsSet::from(&F);
    if (F.isIntrinsic() or not FTags.contains(FunctionTags::Lifted))
      continue;
    revng_assert(not F.isVarArg());

    ILA.setupForProcessingFunction(ModPass, &F);
    Changed |= ILA.getOrCreateSCEVTypes(TS);

    llvm::ReversePostOrderTraversal RPOT(&F.getEntryBlock());
    for (BasicBlock *B : RPOT) {
      for (Instruction &I : *B) {
        // If I has no operands we've nothing to do.
        if (not I.getNumOperands())
          continue;

        if (mustBeSerialized(I)) {
          revng_unreachable(); // This is not handled yet.
          continue;
        }

        // InsertValue and ExtractValue are special because their operands have
        // struct type, so we don't handle them explictly.
        // Both will be analyzed only as operands of their respective uses.
        if (isa<ExtractValueInst>(I) or isa<InsertValueInst>(I))
          continue;

        // Load and Store are handled separately, because we look into their
        // pointer operands, and we have to add accesses to the generated
        // LayoutTypeSystemNodes.
        if (isa<LoadInst>(I) or isa<StoreInst>(I)) {
          // Regular memory accesses. For now these Instructions are the only
          // one that give us information to identify which Values are pointers
          // to types, because they are used in Load and Stores as
          // PointerOperands.
          Use *PtrUse(nullptr);
          if (auto *Load = dyn_cast<LoadInst>(&I))
            PtrUse = &Load->getOperandUse(Load->getPointerOperandIndex());
          else if (auto *Store = dyn_cast<StoreInst>(&I))
            PtrUse = &Store->getOperandUse(Store->getPointerOperandIndex());
          else
            continue;
          revng_assert(PtrUse != nullptr);

          // Find the possible base addresses of the PointerOperand
          Value *PointerVal = PtrUse->get();

          // But if the pointer operand is a global variable we have nothing to
          // do, because loading from it means reading from a register which has
          // no good information to propagate about types.
          if (isa<GlobalVariable>(PointerVal)) {
            // We should think about adding a special case for the register used
            // as stack pointer, but we need to find a nice way to do it that is
            // architecture independent.
            continue;
          }

          // If the pointer operand is null or undef we have nothing to do.
          if (isa<ConstantPointerNull>(PointerVal)
              or isa<UndefValue>(PointerVal)) {
            continue;
          }

          Changed |= ILA.createBaseAddrWithInstanceLink(TS, PointerVal, *B);
          auto *AddrLayout = TS.getLayoutType(PointerVal);
          AddrLayout->L.Accesses.insert(PtrUse);
          continue;
        }

        SmallVector<Value *, 8> Pointers;

        // Handle all the other instructions, looking if we can find the base
        // address from which is calculated each Instruction, if it can
        // represent an address.
        if (auto *Ret = dyn_cast<ReturnInst>(&I)) {

          if (not Ret->getNumOperands())
            continue;

          revng_assert(Ret->getNumOperands() == 1U);
          auto *RetVal = Ret->getOperand(0);

          if (isa<UndefValue>(RetVal))
            continue;

          if (RetVal->getType()->isStructTy()) {
            // If RetVal is a ConstantAggregate we cannot infer anything about
            // type layouts right now. We need to handle layout pointed to by
            // constant addresses first. This might be useful to infer types in
            // data sections of binaries be we don't handle it now. When we do,
            // it will become necessary to handle this case.
            if (isa<ConstantAggregate>(RetVal)
                or isa<ConstantAggregateZero>(RetVal))
              continue;

            if (auto *Call = dyn_cast<CallInst>(RetVal)) {

              const Function *Callee = getCallee(Call);
              auto CTags = FunctionTags::TagsSet::from(Callee);
              revng_assert(CTags.contains(FunctionTags::StructInitializer));

              revng_assert(not Callee->isVarArg());
              auto *RetTy = cast<StructType>(Callee->getReturnType());
              revng_assert(RetTy == F.getReturnType());
              revng_assert(RetTy->getNumElements() == Callee->arg_size());

              Pointers.append(Call->arg_begin(), Call->arg_end());

            } else {

              auto *InsVal = cast<InsertValueInst>(RetVal);
              Pointers = getInsertValueLeafOperands(InsVal);
            }

          } else {
            revng_assert(isa<IntegerType>(RetVal->getType())
                         or isa<PointerType>(RetVal->getType()));
            Pointers.push_back(RetVal);
          }
        } else if (auto *Call = dyn_cast<CallInst>(&I)) {
          // For calls we actually look at their parameters.
          for (Value *PointerVal : Call->arg_operands())
            Pointers.push_back(PointerVal);
        } else if (isa<PtrToIntInst>(&I) or isa<IntToPtrInst>(&I)
                   or isa<BitCastInst>(&I)) {
          Pointers.push_back(I.getOperand(0));
        } else {

          // Ignore Instructions that, depending on their type, cannot represent
          // an address. Among these types that cannot represent pointers are
          // for now void and bool (which is just a 1-bit wide integer in llvm)
          llvm::Type *InstrType = I.getType();
          if (InstrType->isVoidTy() or InstrType->isIntegerTy(1))
            continue;

          switch (I.getOpcode()) {
          case Instruction::Mul:
          case Instruction::SDiv:
          case Instruction::UDiv:
          case Instruction::SRem:
          case Instruction::URem:
          case Instruction::AShr:
          case Instruction::LShr:
          case Instruction::Shl:
          case Instruction::And:
          case Instruction::Xor:
          case Instruction::Or:
            continue;
          default: // do nothing
                   ;
          }

          // Consider other Instructions themselves as pointers.
          Pointers.push_back(&I);
        }

        for (Value *PointerVal : Pointers) {
          if (nullptr != PointerVal)
            Changed |= ILA.createBaseAddrWithInstanceLink(TS, PointerVal, *B);
        }
      }
    }
  }
  if (VerifyLog.isEnabled()) {
    revng_assert(TS.verifyConsistency());
    revng_assert(TS.verifyInstanceDAG());
  }
  return Changed;
}
