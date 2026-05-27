//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Passes/PassBuilder.h"

#include "revng/FunctionIsolation/StructInitializers.h"
#include "revng/Model/FunctionTags.h"
#include "revng/Support/Debug.h"
#include "revng/Support/IRBuilder.h"
#include "revng/Support/IRHelpers.h"
#include "revng/Support/OpaqueFunctionsPool.h"

using namespace llvm;

/// This pass eliminates phi nodes of `StructType` by splitting them into one
/// phi per struct field.
///
/// For each phi `%p : {T0, ..., Tn}`, it inserts an `OpaqueExtractvalue` for
/// each field at the end of every predecessor block, then creates `n+1`
/// scalar phis, one per field. Uses of the original phi that are themselves
/// `OpaqueExtractvalue` calls get rewired to the matching scalar phi; chained
/// struct phis are handled by reusing the per-field phis of the producer
/// directly (so no extraction is needed across the chain). Any other kind of
/// use causes the pass to abort.
///
/// It turns:
///
/// a:
///   %x1 = call <{i64, i64}> @f(...)
///   br c
/// b:
///   %x2 = call <{i64, i64}> @f(...)
///   br c
/// c:
///   %x3 = phi <{i64, i64}> [(a, %x1), (b, %x2)]
///   %y0 = call i64 @OpaqueExtractvalue(%x3, i64 0)
///   %y1 = call i64 @OpaqueExtractvalue(%x3, i64 1)
///
/// Into:
///
/// a:
///   %x1   = call <{i64, i64}> @f(...)
///   %x1.0 = call i64 @OpaqueExtractvalue(%x1, i64 0)
///   %x1.1 = call i64 @OpaqueExtractvalue(%x1, i64 1)
///   br c
/// b:
///   %x2   = call <{i64, i64}> @f(...)
///   %x2.0 = call i64 @OpaqueExtractvalue(%x2, i64 0)
///   %x2.1 = call i64 @OpaqueExtractvalue(%x2, i64 1)
///   br c
/// c:
///   %y0 = phi i64 [(a, %x1.0), (b, %x2.0)]
///   %y1 = phi i64 [(a, %x1.1), (b, %x2.1)]
class SplitStructPhis : public llvm::FunctionPass {
public:
  static char ID;
  SplitStructPhis() : llvm::FunctionPass(ID) {}

  bool runOnFunction(llvm::Function &F) override {
    SmallVector<PHINode *, 16> SingleIncoming;
    SmallVector<PHINode *, 16> MultiIncoming;

    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        auto *Phi = dyn_cast<PHINode>(&I);
        if (Phi == nullptr or not isa<StructType>(Phi->getType()))
          continue;
        if (Phi->getNumIncomingValues() == 1)
          SingleIncoming.push_back(Phi);
        else
          MultiIncoming.push_back(Phi);
      }
    }

    bool Changed = not SingleIncoming.empty() or not MultiIncoming.empty();

    // Trivially eliminate single-incoming struct phis.
    for (PHINode *Phi : SingleIncoming) {
      Phi->replaceAllUsesWith(Phi->getIncomingValue(0));
      Phi->eraseFromParent();
    }

    if (MultiIncoming.empty())
      return Changed;

    auto OpaqueEVPool = FunctionTags::OpaqueExtractValue
                          .getPool(*F.getParent());
    LLVMContext &Ctx = F.getContext();
    Type *Int64Ty = IntegerType::getInt64Ty(Ctx);

    // Step 1: pre-create empty per-field phis for every struct phi, so we can
    // resolve cross-references (including cycles) between chained struct phis.
    DenseMap<PHINode *, SmallVector<PHINode *, 2>> PerFieldPhis;
    for (PHINode *Phi : MultiIncoming) {
      auto *ST = cast<StructType>(Phi->getType());
      unsigned NumIncoming = Phi->getNumIncomingValues();
      SmallVector<PHINode *, 2> Fields;
      for (Type *FieldType : ST->elements()) {
        auto *NewPhi = PHINode::Create(FieldType, NumIncoming, "", Phi);
        NewPhi->setDebugLoc(Phi->getDebugLoc());
        Fields.push_back(NewPhi);
      }
      PerFieldPhis[Phi] = std::move(Fields);
    }

    // Step 2: fill in the incoming values for each per-field phi.
    for (PHINode *Phi : MultiIncoming) {
      auto *ST = cast<StructType>(Phi->getType());
      auto &Fields = PerFieldPhis[Phi];

      for (unsigned I = 0, N = Phi->getNumIncomingValues(); I < N; ++I) {
        Value *Incoming = Phi->getIncomingValue(I);
        BasicBlock *Pred = Phi->getIncomingBlock(I);

        // Undef / poison: propagate the same to every per-field phi.
        if (isa<UndefValue>(Incoming) or isa<PoisonValue>(Incoming)) {
          for (auto &&[FieldType, NewPhi] : zip(ST->elements(), Fields)) {
            Value *V = isa<PoisonValue>(Incoming) ?
                         cast<Value>(PoisonValue::get(FieldType)) :
                         UndefValue::get(FieldType);
            NewPhi->addIncoming(V, Pred);
          }
          continue;
        }

        // Chained struct phi: reuse the producer's per-field phis directly.
        if (auto *IncomingPhi = dyn_cast<PHINode>(Incoming)) {
          auto It = PerFieldPhis.find(IncomingPhi);
          if (It != PerFieldPhis.end()) {
            for (auto &&[Src, Dst] : zip(It->second, Fields))
              Dst->addIncoming(Src, Pred);
            continue;
          }
        }

        // If the incoming is a `struct_initializer` call, use its arguments
        // directly instead of materializing `OpaqueExtractvalue` calls that
        // would just undo the packing. The `struct_initializer` call becomes
        // dead afterwards and DCE will clean it up.
        if (auto *Call = dyn_cast<CallInst>(Incoming);
            Call != nullptr
            and isCallToTagged(Call, FunctionTags::StructInitializer)) {
          revng_assert(Call->arg_size() == Fields.size());
          for (auto &&[Arg, FieldNewPhi] : zip(Call->args(), Fields))
            FieldNewPhi->addIncoming(Arg.get(), Pred);
          continue;
        }

        // General case: materialize one OpaqueExtractvalue per field at the
        // end of the predecessor block.
        Instruction *InsertBefore = Pred->getTerminator();
        IRBuilder<> Builder(InsertBefore);
        for (auto &&[Idx, FieldNewPhi] : llvm::enumerate(Fields)) {
          Type *FieldType = ST->getElementType(Idx);
          auto *FT = FunctionType::get(FieldType,
                                       { Incoming->getType(), Int64Ty },
                                       false);
          FunctionTags::TypePair Key = { FieldType, Incoming->getType() };
          auto *EVFn = OpaqueEVPool.get(Key, FT, "OpaqueExtractvalue");
          auto *Index = ConstantInt::get(Int64Ty, Idx);
          CallInst *Extract = Builder.CreateCall(EVFn, { Incoming, Index });
          Extract->setDebugLoc(InsertBefore->getDebugLoc());
          FieldNewPhi->addIncoming(Extract, Pred);
        }
      }
    }

    // Step 3: rewrite uses of the original struct phis.
    //   - `OpaqueExtractvalue` users get rewired to the matching per-field phi.
    //   - Chained struct phis already point at our per-field phis (Step 2), so
    //     we leave their use of the original phi alone (it'll be dropped in
    //     Step 4 when the chained phi itself is erased).
    //   - Anything else: reconstruct the struct via `struct_initializer` at the
    //     join block and redirect the use to the reconstruction.
    // remove-lifting-artifacts has already purged the bodies of all
    // non-isolated functions before this point, so don't put one back.
    StructInitializers Initializers(F.getParent(), /* EmitBody */ false);

    for (PHINode *Phi : MultiIncoming) {
      auto &Fields = PerFieldPhis[Phi];

      for (User *U : llvm::make_early_inc_range(Phi->users())) {
        auto *Call = dyn_cast<CallInst>(U);
        if (Call != nullptr
            and isCallToTagged(Call, FunctionTags::OpaqueExtractValue)) {
          auto *IndexConst = cast<ConstantInt>(Call->getArgOperand(1));
          uint64_t Index = IndexConst->getZExtValue();
          revng_assert(Index < Fields.size());
          Call->replaceAllUsesWith(Fields[Index]);
          Call->eraseFromParent();
        }
      }

      // After OpaqueExtractvalue rewriting, any remaining non-chained use
      // needs the original struct value. Reconstruct it via struct_initializer.
      bool NeedsReconstruction = false;
      for (User *U : Phi->users()) {
        if (auto *OtherPhi = dyn_cast<PHINode>(U))
          if (PerFieldPhis.find(OtherPhi) != PerFieldPhis.end())
            continue;
        NeedsReconstruction = true;
        break;
      }

      if (NeedsReconstruction) {
        Instruction *InsertionPoint = Phi->getParent()->getFirstNonPHI();
        revng::IRBuilder ReconstructBuilder(Ctx);
        ReconstructBuilder.SetInsertPoint(InsertionPoint, Phi->getDebugLoc());

        SmallVector<Value *, 4> FieldValues(Fields.begin(), Fields.end());
        auto *ST = cast<StructType>(Phi->getType());
        CallInst *Reconstructed = Initializers.createCall(ReconstructBuilder,
                                                          ST,
                                                          FieldValues);

        for (User *U : llvm::make_early_inc_range(Phi->users())) {
          if (auto *OtherPhi = dyn_cast<PHINode>(U))
            if (PerFieldPhis.find(OtherPhi) != PerFieldPhis.end())
              continue;
          U->replaceUsesOfWith(Phi, Reconstructed);
        }
      }
    }

    // Step 4: erase the original struct phis. Any remaining cross-references
    // between them (in chained-phi operand lists) are dropped via undef so
    // that the erases can happen in any order.
    for (PHINode *Phi : MultiIncoming)
      Phi->replaceAllUsesWith(UndefValue::get(Phi->getType()));
    for (PHINode *Phi : MultiIncoming)
      Phi->eraseFromParent();

    return Changed;
  }
};

char SplitStructPhis::ID;
static RegisterPass<SplitStructPhis> R("split-struct-phis", "", false, false);
