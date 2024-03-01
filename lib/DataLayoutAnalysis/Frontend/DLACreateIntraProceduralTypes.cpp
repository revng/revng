//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <optional>
#include <utility>

#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Analysis/LoopInfo.h"
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

#include "revng/EarlyFunctionAnalysis/FunctionMetadataCache.h"
#include "revng/Model/Architecture.h"
#include "revng/Support/Assert.h"
#include "revng/Support/Debug.h"
#include "revng/Support/FunctionTags.h"
#include "revng/Support/IRHelpers.h"

#include "revng-c/DataLayoutAnalysis/DLATypeSystem.h"
#include "revng-c/Support/FunctionTags.h"
#include "revng-c/Support/IRHelpers.h"

#include "../FuncOrCallInst.h"
#include "DLATypeSystemBuilder.h"
#include "SCEVBaseAddressExplorer.h"

using namespace dla;
using namespace llvm;
using namespace model::Architecture;

using LayoutTypeSystemNode = dla::LayoutTypeSystemNode;
using SCEVTypeMap = SCEVBaseAddressExplorer::SCEVTypeMap;

static int64_t getSCEVConstantSExtVal(const SCEV *S) {
  return cast<SCEVConstant>(S)->getAPInt().getSExtValue();
}

class DLATypeSystemLLVMBuilder::InstanceLinkAdder {
public:
  InstanceLinkAdder(const model::Binary &M, FunctionMetadataCache &Cache) :
    Model(M), Cache(&Cache) {}

private:
  const model::Binary &Model;
  Function *F;
  ScalarEvolution *SE;
  llvm::DominatorTree DT;
  llvm::PostDominatorTree PDT;

  SCEVTypeMap SCEVToLayoutType;
  FunctionMetadataCache *Cache;

protected:
  bool addInstanceLink(DLATypeSystemLLVMBuilder &Builder,
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
        const auto &[Layout, NewType] = Builder.getOrCreateLayoutType(BaseAddr);
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

    const auto &[Tgt, IsNewType] = Builder.getOrCreateLayoutType(PointerVal);
    Created |= IsNewType;
    revng_assert(Tgt != nullptr);

    revng_assert(Src != Tgt or BaseAddrSCEV == SE->getSCEV(PointerVal));
    if (Src == Tgt)
      return Created;

    const SCEV *PointerValSCEV = SE->getSCEV(PointerVal);
    if (PointerValSCEV->getType()->isPointerTy()) {
      auto PointerByteSize = getPointerSize(Model.Architecture());
      llvm::Type *IntType = llvm::IntegerType::getIntNTy(PointerVal
                                                           ->getContext(),
                                                         PointerByteSize * 8);
      PointerValSCEV = SE->getPtrToIntExpr(PointerValSCEV, IntType);
    }

    if (BaseAddrSCEV->getType()->isPointerTy()) {
      auto PointerByteSize = getPointerSize(Model.Architecture());
      llvm::Type *IntType = llvm::IntegerType::getIntNTy(PointerVal
                                                           ->getContext(),
                                                         PointerByteSize * 8);
      BaseAddrSCEV = SE->getPtrToIntExpr(BaseAddrSCEV, IntType);
    }

    revng_assert(isa<IntegerType>(BaseAddrSCEV->getType())
                 and isa<IntegerType>(PointerValSCEV->getType()));

    auto BaseAddrBitWidth = BaseAddrSCEV->getType()->getIntegerBitWidth();
    auto PointerValBitWidth = PointerValSCEV->getType()->getIntegerBitWidth();
    if (BaseAddrBitWidth < PointerValBitWidth) {
      BaseAddrSCEV = SE->getZeroExtendExpr(BaseAddrSCEV,
                                           PointerValSCEV->getType());
    } else if (BaseAddrBitWidth > PointerValBitWidth) {
      PointerValSCEV = SE->getZeroExtendExpr(PointerValSCEV,
                                             BaseAddrSCEV->getType());
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

      // If the stride is not a constant we cannot handle it, so we bail out.
      if (not isa<SCEVConstant>(StrideExpr))
        return Created;

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
        auto *Count = dyn_cast<SCEVConstant>(SCEVBackedgeCount);
        if (Count != nullptr and not Count->isZero()) {
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

    if (OE.Strides.size()) {
      // Nested SCEVAddRecs are not guaranteed to have strides that go from
      // larger to smaller.
      // This code re-orders them so that the larger strides always come first.
      // TODO: this dance of copies is only necessary because Strides and
      // TripCounts are held as 2 separate vectors in OffsetExpression.
      // Eventually we should change the code so that its a vector of structs
      // instead of two separate vectors.
      llvm::SmallVector<std::pair<int64_t, std::optional<int64_t>>> Tmp;
      for (const auto &[S, TC] : llvm::zip(OE.Strides, OE.TripCounts))
        Tmp.push_back({ S, TC });
      llvm::stable_sort(Tmp, llvm::on_first<std::greater<int64_t>>());
      for (const auto &Group : llvm::enumerate(Tmp)) {
        OE.Strides[Group.index()] = Group.value().first;
        OE.TripCounts[Group.index()] = Group.value().second;
      }
    }

    // For now we do not support offsets that are not constant.
    if (not isa<SCEVConstant>(OffsetSCEV))
      return Created;

    // Don't add links for instances at negative offsets.
    OE.Offset = getSCEVConstantSExtVal(OffsetSCEV);
    if (OE.Offset < 0LL)
      return Created;

    Created |= Builder.TS.addInstanceLink(Src, Tgt, std::move(OE)).second;
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

  bool getOrCreateSCEVTypes(DLATypeSystemLLVMBuilder &Builder) {
    bool Changed = false;

    // Add entry in SCEVToLayoutType map for arguments. We always add these
    // because here F is always an isolated function.
    for (Argument &A : F->args()) {
      revng_assert(isa<IntegerType>(A.getType())
                   or isa<PointerType>(A.getType()));
      LayoutTypeSystemNode *ArgLayout = Builder.getLayoutType(&A);
      const SCEV *S = SE->getSCEV(&A);
      SCEVToLayoutType.insert(std::make_pair(S, ArgLayout));
    }

    auto &TS = Builder.TS;

    for (BasicBlock &B : *F) {
      for (auto &I : B) {
        // Add entry in SCEVToLayoutType map for values returned by F
        if (auto *RetI = dyn_cast<ReturnInst>(&I)) {
          if (Value *RetVal = RetI->getReturnValue()) {
            revng_assert(isa<StructType>(RetVal->getType())
                         or isa<IntegerType>(RetVal->getType())
                         or isa<PointerType>(RetVal->getType()));
            if (isa<StructType>(RetVal->getType())) {
              auto RetTys = Builder.getLayoutTypes(*RetVal);
              auto NRetTypes = RetTys.size();
              revng_assert(NRetTypes > 1ULL);

              for (Value *Leaf : findPhiTreeLeaves(RetVal)) {
                // If Leaf is a ConstantAggregate we cannot infer anything
                // about type layouts right now. We need to handle layout
                // pointed to by constant addresses first. This might be useful
                // to infer types in data sections of binaries be we don't
                // handle it now. When we do, it will become necessary to handle
                // this case.
                if (isa<ConstantAggregate>(Leaf)
                    or isa<ConstantAggregateZero>(Leaf))
                  continue;

                if (isa<UndefValue>(Leaf))
                  continue;

                if (auto *Call = dyn_cast<CallInst>(Leaf)) {
                  const Function *Callee = getCallee(Call);
                  revng_assert(Callee != nullptr);

                  auto CTags = FunctionTags::TagsSet::from(Callee);
                  revng_assert(CTags.contains(FunctionTags::StructInitializer));

                  revng_assert(not Callee->isVarArg());

                  auto *RetTy = cast<StructType>(Callee->getReturnType());
                  revng_assert(RetTy->getNumElements() == Callee->arg_size());
                  revng_assert(RetTy == F->getReturnType());

                  auto StructTypeNodes = Builder.getOrCreateLayoutTypes(*Call);
                  revng_assert(StructTypeNodes.size() == Callee->arg_size());

                  for (const auto &[RetNodeNew, Arg] :
                       llvm::zip_first(StructTypeNodes, Call->args())) {
                    const auto &[ArgNode,
                                 New] = Builder.getOrCreateLayoutType(Arg);
                    Changed |= New;
                    const auto &[RetNode, NewNode] = RetNodeNew;
                    Changed |= NewNode;
                    Changed |= TS.addEqualityLink(RetNode, ArgNode).second;
                  }
                }
              }

            } else {
              LayoutTypeSystemNode *RetTy = Builder.getLayoutType(RetVal);
              const SCEV *S = SE->getSCEV(RetVal);
              SCEVToLayoutType.insert(std::make_pair(S, RetTy));
            }
          }
        } else if (auto *PHI = dyn_cast<PHINode>(&I)) {

          // Booleans can not be addresses, so we can skip them.
          if (PHI->getType()->isIntegerTy(1))
            continue;

          revng_assert(isa<IntegerType>(PHI->getType())
                       or isa<PointerType>(PHI->getType())
                       or isa<StructType>(PHI->getType()));
          if (not isa<StructType>(PHI->getType())) {

            LayoutTypeSystemNode *PHIType = Builder.getLayoutType(PHI);
            const SCEV *PHISCEV = SE->getSCEV(PHI);
            SCEVToLayoutType.insert(std::make_pair(PHISCEV, PHIType));

            // PHI Incoming values
            for (Value *In : PHI->incoming_values()) {
              revng_assert(isa<IntegerType>(In->getType())
                           or isa<PointerType>(In->getType()));
              LayoutTypeSystemNode *InTy = Builder.getLayoutType(In);
              const SCEV *InSCEV = SE->getSCEV(In);
              SCEVToLayoutType.insert(std::make_pair(InSCEV, InTy));
            }
          } else {
            // If it's a struct is not SCEVable, so there's no point in trying
            // to get SCEVs for this.
          }

        } else if (auto *Sel = dyn_cast<SelectInst>(&I)) {
          // Booleans can not be addresses, so we can skip them.
          if (Sel->getType()->isIntegerTy(1))
            continue;

          revng_assert(isa<IntegerType>(Sel->getType())
                       or isa<PointerType>(Sel->getType()));

          // Selects are very much like PHIs.
          const auto &[SelType, New] = Builder.getOrCreateLayoutType(Sel);
          Changed |= New;
          const SCEV *SelSCEV = SE->getSCEV(Sel);
          SCEVToLayoutType.insert(std::make_pair(SelSCEV, SelType));

          // True incoming value
          {
            Value *TrueV = Sel->getTrueValue();
            revng_assert(isa<IntegerType>(TrueV->getType())
                         or isa<PointerType>(TrueV->getType()));
            const auto &[TrueTy, NewT] = Builder.getOrCreateLayoutType(TrueV);
            Changed |= NewT;
            const SCEV *TrueSCEV = SE->getSCEV(TrueV);
            SCEVToLayoutType.insert(std::make_pair(TrueSCEV, TrueTy));
            Changed |= Builder.TS
                         .addInstanceLink(TrueTy, SelType, OffsetExpression{})
                         .second;
          }

          // False incoming value
          {
            Value *FalseV = Sel->getFalseValue();
            revng_assert(isa<IntegerType>(FalseV->getType())
                         or isa<PointerType>(FalseV->getType()));
            const auto &[FalseTy, NewT] = Builder.getOrCreateLayoutType(FalseV);
            Changed |= NewT;
            const SCEV *FalseSCEV = SE->getSCEV(FalseV);
            SCEVToLayoutType.insert(std::make_pair(FalseSCEV, FalseTy));
            Changed |= Builder.TS
                         .addInstanceLink(FalseTy, SelType, OffsetExpression{})
                         .second;
          }

        } else if (auto *C = dyn_cast<CallInst>(&I)) {

          if (isCallToTagged(C, FunctionTags::MallocLike)) {

            const auto &[StackLayout, New] = Builder.getOrCreateLayoutType(C);
            Changed |= New;
            const SCEV *CallSCEV = SE->getSCEV(C);
            SCEVToLayoutType.insert(std::make_pair(CallSCEV, StackLayout));

            auto *Placeholder = TS.createArtificialLayoutType();
            Placeholder->Size = getPointerSize(Model.Architecture());
            TS.addPointerLink(Placeholder, StackLayout);
            Changed = true;
            continue;
          }

          if (isCallToTagged(C, FunctionTags::AddressOf)) {
            // AddressOf always generates a Layout node
            const auto &[AddrLayout, New] = Builder.getOrCreateLayoutType(C);
            Changed |= New;
            const SCEV *CallSCEV = SE->getSCEV(C);
            SCEVToLayoutType.insert(std::make_pair(CallSCEV, AddrLayout));

            // Add an equality edge between the `AddressOf` node and it's
            // pointee node
            auto *Arg = C->getArgOperand(1);
            auto [PointedLayout, ArgIsNew] = Builder.getOrCreateLayoutType(Arg);
            Changed |= ArgIsNew;
            auto [_, NewLink] = TS.addEqualityLink(PointedLayout, AddrLayout);
            Changed |= NewLink;
            continue;
          }

          if (isCallToTagged(C, FunctionTags::StructInitializer)) {

            const Function *Callee = getCallee(C);
            revng_assert(not Callee->isVarArg());

            auto *RetTy = cast<StructType>(Callee->getReturnType());
            revng_assert(RetTy->getNumElements() == Callee->arg_size());

            auto StructTypeNodes = Builder.getOrCreateLayoutTypes(*C);
            revng_assert(StructTypeNodes.size() == Callee->arg_size());

            for (const auto &[RetTypeNodeNew, Arg] :
                 llvm::zip_first(StructTypeNodes, C->args())) {
              const auto &[ArgTypeNode,
                           New] = Builder.getOrCreateLayoutType(Arg);
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
          if (not isCallToIsolatedFunction(C))
            continue;

          const auto PrototypeRef = Cache->getCallSitePrototype(Model, C);
          if (PrototypeRef.empty())
            continue;

          const model::Type *Prototype = PrototypeRef.getConst();
          revng_assert(Prototype);

          const Function *Callee = getCallee(C);
          revng_assert(not Callee or not Callee->isVarArg());

          // Add entry in SCEVToLayoutType map for return values of CallInst
          if (not C->getType()->isVoidTy()) {
            // Return values
            revng_assert(isa<StructType>(C->getType())
                         or isa<IntegerType>(C->getType())
                         or isa<PointerType>(C->getType()));

            if (isa<StructType>(C->getType())) {
              // Types representing the return type
              auto ExtractedVals = getExtractedValuesFromInstruction(C);
              auto Size = ExtractedVals.size();
              auto FormalRetTys = Callee ? Builder.getLayoutTypes(*Callee) :
                                           TS.createArtificialLayoutTypes(Size);
              revng_assert(Size == FormalRetTys.size());
              for (const auto &[Ext, RetTy] :
                   llvm::zip(ExtractedVals, FormalRetTys)) {

                if (Ext.empty())
                  continue;

                for (llvm::CallInst *E : Ext) {
                  using FunctionTags::OpaqueExtractValue;
                  revng_assert(isCallToTagged(E, OpaqueExtractValue));
                  llvm::Type *ExtTy = E->getType();
                  revng_assert(isa<IntegerType>(ExtTy)
                               or isa<PointerType>(ExtTy));

                  const auto &[ExtLayout,
                               New] = Builder.getOrCreateLayoutType(E);
                  Changed |= New;
                  Changed |= TS.addEqualityLink(RetTy, ExtLayout).second;
                  const SCEV *S = SE->getSCEV(E);
                  SCEVToLayoutType.insert(std::make_pair(S, ExtLayout));
                }
              }
            } else {
              // Type representing the return type
              revng_assert(not C->getType()->isIntegerTy(1));
              LayoutTypeSystemNode *RetTy = Callee ?
                                              Builder.getLayoutType(Callee) :
                                              TS.createArtificialLayoutType();
              const auto &[CType, NewC] = Builder.getOrCreateLayoutType(C);
              Changed |= NewC;
              Changed |= Builder.TS.addEqualityLink(RetTy, CType).second;
              const SCEV *RetS = SE->getSCEV(C);
              SCEVToLayoutType.insert(std::make_pair(RetS, CType));
            }
          }

          // Add entry in SCEVToLayoutType map for actual arguments of CallInst.
          for (Use &ArgU : C->args()) {
            revng_assert(isa<IntegerType>(ArgU->getType())
                         or isa<PointerType>(ArgU->getType()));
            const auto &[ArgTy, Created] = Builder.getOrCreateLayoutType(ArgU);
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

                  std::tie(SrcLayout, New) = Builder.getOrCreateLayoutType(Op);
                  Changed |= New;
                  std::tie(TgtLayout,
                           New) = Builder.getOrCreateLayoutType(CExpr);
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
            const auto &[LoadedTy, Created] = Builder.getOrCreateLayoutType(L);
            Changed |= Created;
            const SCEV *LoadSCEV = SE->getSCEV(L);
            SCEVToLayoutType.insert(std::make_pair(LoadSCEV, LoadedTy));
          }
        } else if (auto *A = dyn_cast<AllocaInst>(&I)) {
          revng_assert(isa<IntegerType>(A->getAllocatedType())
                       or isa<PointerType>(A->getAllocatedType()));
          const auto &[LoadedTy, Created] = Builder.getOrCreateLayoutType(A);
          Changed |= Created;
          const SCEV *LoadSCEV = SE->getSCEV(A);
          SCEVToLayoutType.insert(std::make_pair(LoadSCEV, LoadedTy));
        } else if (isa<IntToPtrInst>(&I) or isa<PtrToIntInst>(&I)
                   or isa<BitCastInst>(&I) or isa<ZExtInst>(&I)) {
          Value *Op = I.getOperand(0);

          bool New = false;
          LayoutTypeSystemNode *SrcLayout = nullptr;
          LayoutTypeSystemNode *TgtLayout = nullptr;

          std::tie(SrcLayout, New) = Builder.getOrCreateLayoutType(Op);
          Changed |= New;
          std::tie(TgtLayout, New) = Builder.getOrCreateLayoutType(&I);
          Changed |= New;

          Changed |= Builder.TS.addEqualityLink(SrcLayout, TgtLayout).second;
          const SCEV *LoadSCEV = SE->getSCEV(&I);
          SCEVToLayoutType.insert(std::make_pair(LoadSCEV, TgtLayout));
        }
      }
    }
    return Changed;
  }

  bool createBaseAddrWithInstanceLink(DLATypeSystemLLVMBuilder &Builder,
                                      Value *PointerVal,
                                      const BasicBlock &B) {
    revng_assert(PointerVal);

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
      AddedSomething |= addInstanceLink(Builder, PointerVal, BaseAddrSCEV, B);

    return AddedSomething;
  }
};

using Builder = DLATypeSystemLLVMBuilder;

bool Builder::connectToFuncsWithSamePrototype(const llvm::CallInst *Call,
                                              const model::Binary &Model) {
  revng_assert(Call);
  // This procedure for regular functions (i.e. those that are used in direct
  // calls) is done separately by `CreateInterProceduralTypes`
  revng_assert(Call->isIndirectCall());
  bool Changed = false;

  auto Prototype = Cache->getCallSitePrototype(Model, Call);
  if (Prototype.empty())
    return false;

  auto It = VisitedPrototypes.find(Prototype.get());
  if (It == VisitedPrototypes.end()) {
    VisitedPrototypes.insert({ Prototype.get(), Call });
  } else {
    FuncOrCallInst OtherCall = It->second;
    revng_assert(not OtherCall.isNull());

    if (Call->getType()->isVoidTy()) {
      revng_assert(OtherCall.getRetType()->isVoidTy());
    } else {
      revng_assert(not OtherCall.getRetType()->isVoidTy());
      // Connect return values
      auto OtherRetVals = getLayoutTypes(*OtherCall.getVal());
      auto RetVals = getLayoutTypes(*Call);
      revng_assert(RetVals.size() == OtherRetVals.size());
      for (auto [N1, N2] : llvm::zip(OtherRetVals, RetVals)) {
        Changed = true;
        TS.addEqualityLink(N1, N2);
      }
    }

    // Connect arguments
    for (const auto &ArgIt : llvm::enumerate(Call->args())) {
      // Arguments can only be integers and pointers
      const Value *Arg1 = ArgIt.value();
      const Value *Arg2 = OtherCall.getArg(ArgIt.index());
      revng_assert(Arg1->getType()->isIntOrPtrTy()
                   and Arg2->getType()->isIntOrPtrTy());

      auto *Arg1Node = getLayoutType(Arg1);
      auto *Arg2Node = getLayoutType(Arg2);

      Changed = true;
      TS.addEqualityLink(Arg1Node, Arg2Node);
    }
  }

  return Changed;
}

bool Builder::createIntraproceduralTypes(llvm::Module &M,
                                         llvm::ModulePass *MP,
                                         const model::Binary &Model) {
  bool Changed = false;
  InstanceLinkAdder ILA(Model, *Cache);

  for (Function &F : M.functions()) {
    auto FTags = FunctionTags::TagsSet::from(&F);
    if (F.isIntrinsic() or not FTags.contains(FunctionTags::Isolated))
      continue;
    revng_assert(not F.isVarArg());

    ILA.setupForProcessingFunction(MP, &F);
    Changed |= ILA.getOrCreateSCEVTypes(*this);

    llvm::ReversePostOrderTraversal RPOT(&F.getEntryBlock());
    for (BasicBlock *B : RPOT) {
      for (Instruction &I : *B) {
        // If I has no operands we've nothing to do.
        if (not I.getNumOperands())
          continue;

        // ExtractValue is special since its operand has struct type, so we
        // don't handle them explicitly.  It will be analyzed only as operands
        // of its uses.
        revng_assert(not isa<ExtractValueInst>(I));
        if (isCallToTagged(&I, FunctionTags::OpaqueExtractValue))
          continue;

        // Load and Store are handled separately, because we look into their
        // pointer operands, and we have to add accesses to the generated
        // LayoutTypeSystemNodes.
        if (isa<LoadInst>(I) or isa<StoreInst>(I)) {
          // Regular memory accesses. For now these Instructions are the only
          // one that give us information to identify which Values are pointers
          // to types, because they are used in Load and Stores as
          // PointerOperands.
          Value *PointerVal = nullptr;
          Value *Val = nullptr;
          if (auto *Load = dyn_cast<LoadInst>(&I)) {
            PointerVal = Load->getPointerOperand();
            Val = &I;
          } else if (auto *Store = dyn_cast<StoreInst>(&I)) {
            PointerVal = Store->getPointerOperand();
            Val = Store->getValueOperand();
          } else {
            continue;
          }
          revng_assert(PointerVal);

          // But if the pointer operand is a global variable we have nothing to
          // do, because loading from it means reading from a register which has
          // no good information to propagate about types.
          if (isa<GlobalVariable>(PointerVal))
            continue;

          // If the pointer operand is null or undef we have nothing to do.
          if (isa<ConstantPointerNull>(PointerVal)
              or isa<UndefValue>(PointerVal)) {
            continue;
          }

          // Create Base node
          Changed |= ILA.createBaseAddrWithInstanceLink(*this, PointerVal, *B);

          // Create Access node
          auto AccessSize = Val->getType()->getScalarSizeInBits() / 8;
          auto *AccessNode = TS.createArtificialLayoutType();
          AccessNode->Size = AccessSize;
          AccessNode->InterferingInfo = AllChildrenAreNonInterfering;

          // Add link between pointer node and access node
          auto *PointerNode = getLayoutType(PointerVal);
          revng_assert(PointerNode);
          TS.addInstanceLink(PointerNode, AccessNode, OffsetExpression{});

          // Create pointer edge between the access node and the pointee node.
          const auto &[PointeeNode, HasChanged] = getOrCreateLayoutType(Val);
          revng_assert(PointeeNode);
          revng_assert(not HasChanged or not isa<LoadInst>(I));

          // Add a pointer link from the AccessNode to the PointeeNode. Note
          // that at this stage we don't know yet if PointeeNode is going to be
          // the root of another chunk of the type hierarchy, or just a plain
          // scalar. But because of graph initialization works, we need to
          // assume optimistically that it will NOT be a plain scalar. This has
          // 2 consequences:
          //  - the pointee node becomes a place where a different part of the
          //    type hierarchy will attach when it will be materialized later
          //    during a new iteration of this loop on a new instruction
          // - it will provide the information that many Load/Store instructions
          //   have read/written a given value. This is used later to add
          //   equality links between things that have pointer edges towards the
          //   same PointeeNode.

          // All this being said, if the AccessSize is different from the
          // pointer size in the model, we already know for sure that the
          // pointee is not going to be a pointer, and we can bail out early.
          if (AccessSize != getPointerSize(Model.Architecture()))
            continue;

          PointeeNode->InterferingInfo = Unknown;
          TS.addPointerLink(AccessNode, PointeeNode);

          // If the pointee already has nodes that point to it, their type must
          // be the same as the type of PointerNode, so add an equality link.
          using PointerGraph = EdgeFilteredGraph<LayoutTypeSystemNode *,
                                                 isPointerEdge>;
          using InversePointerGraph = llvm::Inverse<PointerGraph>;
          for (LayoutTypeSystemNode *PointerToPointee :
               llvm::children<InversePointerGraph>(PointeeNode)) {
            revng_assert(AccessSize == getPointerSize(Model.Architecture()));
            revng_assert(AccessSize == PointerToPointee->Size);
            if (PointerToPointee != AccessNode)
              TS.addEqualityLink(PointerToPointee, AccessNode);
          }

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
            for (Value *Leaf : findPhiTreeLeaves(RetVal)) {
              // If Leaf is a ConstantAggregate we cannot infer anything about
              // type layouts right now. We need to handle layout pointed to by
              // constant addresses first. This might be useful to infer types
              // in data sections of binaries be we don't handle it now. When we
              // do, it will become necessary to handle this case.
              if (isa<ConstantAggregate>(Leaf)
                  or isa<ConstantAggregateZero>(Leaf))
                continue;

              if (isa<UndefValue>(Leaf))
                continue;

              auto *Call = cast<CallInst>(Leaf);

              const Function *Callee = getCallee(Call);
              auto CTags = FunctionTags::TagsSet::from(Callee);
              revng_assert(CTags.contains(FunctionTags::StructInitializer));

              revng_assert(not Callee->isVarArg());
              auto *RetTy = cast<StructType>(Callee->getReturnType());
              revng_assert(RetTy == F.getReturnType());
              revng_assert(RetTy->getNumElements() == Callee->arg_size());

              Pointers.append(Call->arg_begin(), Call->arg_end());
            }
          } else {
            revng_assert(isa<IntegerType>(RetVal->getType())
                         or isa<PointerType>(RetVal->getType()));
            Pointers.push_back(RetVal);
          }
        } else if (auto *Call = dyn_cast<CallInst>(&I)) {
          // For calls we actually look at their parameters.
          for (Value *PointerVal : Call->args())
            Pointers.push_back(PointerVal);

        } else if (isa<PtrToIntInst>(&I) or isa<IntToPtrInst>(&I)
                   or isa<BitCastInst>(&I) or isa<ZExtInst>(&I)) {
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
          default: {
            // do nothing
          } break;
          }

          // Consider other Instructions themselves as pointers.
          Pointers.push_back(&I);
        }

        for (Value *PointerVal : Pointers) {
          if (PointerVal and not isa<StructType>(PointerVal->getType()))
            Changed |= ILA.createBaseAddrWithInstanceLink(*this,
                                                          PointerVal,
                                                          *B);
        }

        // For indirect calls, we want to enforce the following: if this call
        // shares the prototype with another function in the model, their return
        // values and arguments must have the same layout. This is done by
        // adding equality links between these nodes.
        if (auto *Call = dyn_cast<CallInst>(&I))
          if (Call->isIndirectCall())
            Changed |= connectToFuncsWithSamePrototype(Call, Model);
      }
    }
  }

  if (VerifyLog.isEnabled())
    revng_assert(TS.verifyConsistency());

  return Changed;
}
