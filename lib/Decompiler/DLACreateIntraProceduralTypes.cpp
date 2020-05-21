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
#include "revng/Support/IRHelpers.h"

#include "DLAHelpers.h"
#include "DLAStep.h"
#include "DLATypeSystem.h"

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

static bool isConstantAddress(const ConstantInt *C) {
  // TODO: insert some logic to properly detect when this constant represents
  // a pointer pointing into some segment
  return false;
}

using LayoutTypeSystemNode = dla::LayoutTypeSystemNode;
using SCEVTypeMap = std::map<const SCEV *, LayoutTypeSystemNode *>;
using SCEVTypeMekerMap = std::map<const SCEV *, uint64_t>;

static int64_t getSCEVConstantSExtVal(const SCEV *S) {
  return cast<SCEVConstant>(S)->getAPInt().getSExtValue();
}

static bool isAlwaysAddress(const Value &V) {
  if (auto *Call = dyn_cast<CallInst>(&V))
    if (auto *Callee = Call->getCalledFunction())
      if (Callee->hasName() and Callee->getName() == "revng_init_local_sp")
        return true;
  return false;
}

static bool isNeverAddress(const Value &I) {

  if (isAlwaysAddress(I))
    return false;

  const auto IsCallToNotIsolated = [](const Value *V) {
    if (auto *C = dyn_cast<CallInst>(V))
      return nullptr == C->getCalledFunction()->getMetadata("revng.func.entry");
    return false;
  };

  if (auto *E = dyn_cast<ExtractValueInst>(&I))
    if (IsCallToNotIsolated(E->getAggregateOperand()))
      return true;

  return IsCallToNotIsolated(&I);
}

class SCEVBaseAddressExplorer {
private:
  SmallVector<const SCEV *, 4> Worklist;

public:
  SCEVBaseAddressExplorer() = default;

  std::set<const SCEV *>
  findBases(ScalarEvolution *SE, const SCEV *Root, const SCEVTypeMap &M) {
    std::set<const SCEV *> Result;
    Worklist.clear();
    Worklist.push_back(Root);

    while (not Worklist.empty()) {
      const SCEV *AddressCandidate = Worklist.pop_back_val();
      const SCEV *AddrSCEV = nullptr;
      if (const auto *C = dyn_cast<SCEVConstant>(AddressCandidate)) {
        // Constants are considered addresses only in case they point to some
        // segment. They are never traversed.
        auto *ConstantVal = C->getValue();
        if (isConstantAddress(ConstantVal))
          AddrSCEV = AddressCandidate;
      } else {
        auto NTraversed = checkAddressOrTraverse(SE, AddressCandidate);
        if (NTraversed) {
          // If we have traversed AddressCandidate, it means that it doesn't
          // look like an address SCEV, so we want to keep looking in is
          // operands to find a base address.
          // However, it might be a typed SCEVs, so we also have to check if
          // AddressCandidate is in M. If it is, we consider it to be an address
          // in any case. In this case, given that we have traversed
          // AddressCandidate, checkAddressOrTraverse has pushed something on
          // the Worlist, and we have to clear it to prevent the SCEV
          // exploration to continue past the address.
          if (AddressCandidate != Root and M.contains(AddressCandidate)) {
            // We have to stop the search in this direction. Pop back all the
            // nodes that have been pushed by checkAddressOrTraverse.
            for (decltype(NTraversed) I = 0ULL; I < NTraversed; ++I)
              Worklist.pop_back();
            AddrSCEV = AddressCandidate;
          }

        } else {

          // If we have not traversed AddressCandidate, but IsAddr is true, it
          // means that AddressCandidate looks like an address SCEV.
          AddrSCEV = AddressCandidate;
          if (auto *U = dyn_cast<SCEVUnknown>(AddressCandidate))
            if (isNeverAddress(*U->getValue()))
              AddrSCEV = nullptr;
        }
      }

      // Add AddrSCEV to the results if we have found one.
      if (AddrSCEV != nullptr)
        Result.insert(AddrSCEV);
    }

    return Result;
  }

private:
  size_t checkAddressOrTraverse(ScalarEvolution *SE, const SCEV *S) {
    auto OldSize = Worklist.size();
    switch (S->getSCEVType()) {
    case scConstant: {
      // Constants should be checked with isConstantAddress.
      revng_unreachable();
    } break;
    case scUnknown: {
      // Unknowns are always considered addresses
      [[maybe_unused]] Value *UVal = cast<SCEVUnknown>(S)->getValue();
      revng_assert(isa<UndefValue>(UVal) or isa<Argument>(UVal)
                   or isa<Instruction>(UVal) or isa<GlobalVariable>(UVal)
                   or isa<ConstantExpr>(UVal));
    } break;
    case scZeroExtend: {
      // Zero extension never changes the value of pointers, so we can safely
      // traversi it.
      const SCEVZeroExtendExpr *ZE = cast<SCEVZeroExtendExpr>(S);
      Worklist.push_back(ZE->getOperand());
    } break;
    case scTruncate:
    case scSignExtend: {
      // Truncate and Extend are basically casts, so in the first implementation
      // we did not consider them addresses per-se, and we traversed them.
      // So we initially had this code here:
      //    Worklist.push_back(cast<SCEVCastExpr>(S)->getOperand());
      // However, it turned out that tend to show up in nasty situations, so we
      // temporarily disabled their traversal.
      // For now they are simply considered addresses, but we might need to
      // reconsider this behavior in the future, and re-enable their traversal
      // with the two commented lines of code above.
    } break;
    case scSMaxExpr:
    case scUMaxExpr:
    case scSMinExpr:
    case scUMinExpr: {
    } break;
    case scAddExpr: {
      const SCEVNAryExpr *Add = cast<SCEVNAryExpr>(S);
      revng_assert(Add->getNumOperands() > 1U);
      // Assume that we only have at most one constant operand.
      bool FoundConstOp = false;
      // Push all the operands on the worlist, to see which of them may be an
      // address.
      for (const SCEV *Op : Add->operands()) {
        if (auto *C = dyn_cast<SCEVConstant>(Op)) {
          revng_assert(not FoundConstOp);
          FoundConstOp = true;
          if (not isConstantAddress(C->getValue()))
            continue;
        }
        Worklist.push_back(Op);
      }
    } break;
    case scAddRecExpr: {
      // We only accept AddRecExpr in certain forms:
      // - the stride must be known
      // - the start must be either a constant, an unknown, a normal add, or
      //   another addrec
      const auto *AddRec = cast<SCEVAddRecExpr>(S);
      const auto *Start = AddRec->getStart();
      const auto *Stride = AddRec->getStepRecurrence(*SE);
      if (not isa<SCEVConstant>(Stride)) {
        break;
      }
      // The AddRec is never an address, but we traverse its start expression
      // because it could be an address.
      Worklist.push_back(Start);
    } break;
    case scMulExpr: {
      // In general scMulExpr are not addresses, so they cannot be traversed.
      // There is only one exception. SCEV do not have the expressive power for
      // representing bit masks. Hence, for expressions of the form
      //   A = B & 0xff00
      // we have a SCEV(A) of the form (B / 8) * 8
      // The SCEV AST tree has the form
      //
      //   B            8
      //    \          /
      //     scUDivExpr      8
      //          \         /
      //           scMulExpr
      //
      // So, we have a MulExpr with exactly two operands, whose second operand
      // is a constant (APConst) that is a power of two, and whose first operand
      // is a UDivExpr (UDiv).
      // (UDiv) also has exactly two operands, and the second is another
      // constant (ADPconstUDiv) which is equal to (APConst). Given that this
      // pattern happens in real code, we pattern match it.
      const SCEVMulExpr *M = cast<SCEVMulExpr>(S);
      if (M->getNumOperands() != 2) {
        break;
      }
      const SCEV *LHS = M->getOperand(0);
      const SCEV *RHS = M->getOperand(1);
      const auto *LHSConst = dyn_cast<SCEVConstant>(LHS);
      const auto *RHSConst = dyn_cast<SCEVConstant>(RHS);
      const auto *LHSUDiv = dyn_cast<SCEVUDivExpr>(LHS);
      const auto *RHSUDiv = dyn_cast<SCEVUDivExpr>(RHS);
      const SCEVConstant *Const = LHSConst ? LHSConst : RHSConst;
      const SCEVUDivExpr *UDiv = LHSUDiv ? LHSUDiv : RHSUDiv;
      if (Const and UDiv) {
        const SCEV *UDivRHS = UDiv->getRHS();
        if (const auto *ConstUDivRHS = dyn_cast<SCEVConstant>(UDivRHS)) {
          const APInt &APConst = Const->getValue()->getValue();
          const APInt &APConstUDiv = ConstUDivRHS->getValue()->getValue();
          if (APConst == APConstUDiv and APConst.isPowerOf2()
              and APConstUDiv.isPowerOf2()) {
            // Here we have pattern matched the pattern described above, so we
            // traverse the composite expression A = B & 0xff00 and keep
            // exploring B, without marking A as address.
            const SCEV *UDivLHS = UDiv->getLHS();
            Worklist.push_back(UDivLHS);
            break;
          }
        }
      }
    } break;
    case scUDivExpr: {
      // UDivExpr are never addresses unless they are matched as alignment
      // operations along with a MulExpr (see the scMulExpr case).
      // Given that we don't traverse them, if we find a UDivExpr we always
      // consider it an address.
    } break;
    default:
      revng_unreachable("Unknown SCEV kind!");
    }
    auto NewSize = Worklist.size();
    revng_assert(NewSize >= OldSize);
    return NewSize - OldSize;
  }
};

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
      OE.Strides.push_back(getSCEVConstantSExtVal(StrideExpr));
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
      OE.TripCounts.push_back(std::move(TripCount));
      OffsetSCEV = Rec->getStart();
    }

    // For now we do not support offsets that are not constant.
    if (not isa<SCEVConstant>(OffsetSCEV))
      return Created;

    OE.Offset = getSCEVConstantSExtVal(OffsetSCEV);
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
          if (not Callee->getMetadata("revng.func.entry"))
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

                if (nullptr == Ext)
                  continue;

                revng_assert(isa<ExtractValueInst>(Ext));

                llvm::Type *ExtTy = Ext->getType();
                revng_assert(isa<IntegerType>(ExtTy)
                             or isa<PointerType>(ExtTy));

                const auto &[ExtLayout, New] = TS.getOrCreateLayoutType(Ext);
                Changed |= New;
                Changed |= TS.addEqualityLink(RetTy, ExtLayout).second;
                const SCEV *S = SE->getSCEV(Ext);
                SCEVToLayoutType.insert(std::make_pair(S, ExtLayout));
              }
            } else {
              // Type representing the return type
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
                revng_assert(isa<ConstantInt>(Op));

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

          if (auto *L = dyn_cast<LoadInst>(&I)) {
            revng_assert(isa<IntegerType>(L->getType())
                         or isa<PointerType>(L->getType()));
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
    if (F.isIntrinsic() or not F.getMetadata("revng.func.entry"))
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

        // Branch operands are Basic Blocks, and we don't want to infer types
        // for those.
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
            auto *InsVal = cast<InsertValueInst>(RetVal);
            Pointers = getInsertValueLeafOperands(InsVal);
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
