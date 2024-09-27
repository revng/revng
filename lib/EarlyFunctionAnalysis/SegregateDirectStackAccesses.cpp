/// \file SegregateDirectStackAccesses.cpp
/// Segregate direct stack accesses from all other memory accesses through alias
/// information.
///
/// This provides a way to say that stack accesses do not interfere with any
/// other memory access.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/PatternMatch.h"

#include "revng/BasicAnalyses/GeneratedCodeBasicInfo.h"
#include "revng/EarlyFunctionAnalysis/SegregateDirectStackAccesses.h"
#include "revng/Support/Assert.h"

using namespace llvm;
class SegregateDirectStackAccessesPassImpl;

using SDSAP = SegregateDirectStackAccessesPass;
using SDSAPI = SegregateDirectStackAccessesPassImpl;

class SegregateDirectStackAccessesPassImpl {
  LLVMContext *Context = nullptr;
  GeneratedCodeBasicInfo *GCBI = nullptr;
  std::vector<Instruction *> DirectStackAccesses;
  std::vector<Instruction *> NotDirectStackAccesses;

public:
  void run(Function &, FunctionAnalysisManager &);

private:
  void segregateAccesses(Function &F);
  void decorateStackAccesses();
};

PreservedAnalyses SDSAP::run(Function &F, FunctionAnalysisManager &FAM) {
  SegregateDirectStackAccessesPassImpl SDASP;
  SDASP.run(F, FAM);
  return PreservedAnalyses::none();
}

void SDSAPI::run(Function &F, FunctionAnalysisManager &FAM) {
  Context = &(F.getContext());

  // Get the result of the GCBI analysis
  GCBI = &(FAM.getResult<GeneratedCodeBasicInfoAnalysis>(F));
  revng_assert(GCBI != nullptr);

  // Populate the two buckets with all load and store instruction of the
  // function, properly segregated.
  segregateAccesses(F);

  // Adorn IR with the alias information collected before.
  decorateStackAccesses();
}

void SDSAPI::segregateAccesses(Function &F) {
  using namespace PatternMatch;

  Value *LoadSP = nullptr;
  bool Found = false;

  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      if (auto *LI = dyn_cast<LoadInst>(&I)) {
        if (GCBI->isSPReg(skipCasts(LI->getPointerOperand()))) {
          revng_assert(!Found);
          LoadSP = LI;
          Found = true;
          break;
        }
      }
    }

    if (Found)
      break;
  }

  // Modify the IR after alloca instructions, if they exist.
  auto It = F.getEntryBlock().begin();
  while (It->getOpcode() == Instruction::Alloca)
    It++;
  IRBuilder<> Builder(&(*It));

  // Context: inttoptr instructions basically inhibits all optimizations. In
  // particular, when an integer is inttoptr'd twice with different destination
  // type, alias analysis messes up. Hence, we need to ensure that no inttoptr
  // exists when operating on a instruction that directly accesses the stack.
  // Note that this problem will be addressed by opaque pointers in the future.
  auto *I8PtrTy = Builder.getInt8PtrTy();
  auto *CE = ConstantExpr::getBitCast(GCBI->spReg(), I8PtrTy->getPointerTo());
  Value *SPI8Ptr = Builder.CreateLoad(I8PtrTy, CE);

  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      Value *Pointer = nullptr;

      // Differentiate accesses and add them onto their respective bucket.
      // Everything that is not a direct access on the stack is put onto the
      // bucket `NotDirectStackAccesses`. Load/store that access the CSVs will
      // have their alias info added later as well.
      if (auto *LI = dyn_cast<LoadInst>(&I))
        Pointer = skipCasts(LI->getPointerOperand());
      else if (auto *SI = dyn_cast<StoreInst>(&I))
        Pointer = skipCasts(SI->getPointerOperand());

      if (Pointer != nullptr) {
        Value *LoadPtr = nullptr;
        Value *LHS = nullptr;
        Value *BitCast = nullptr;
        ConstantInt *Offset = nullptr;
        Builder.SetInsertPoint(&I);

        // Do we have a inttoptr to a load i64, i64* LoadPtr as pointer operand
        // of the current instruction, where LoadPtr is SP? Change it with the
        // newly-created bitcasted load in order to prevent from using inttoptr.
        if (Pointer == LoadSP) {
          Type *DestTy = nullptr;
          if (isa<LoadInst>(&I))
            DestTy = I.getOperand(0)->getType();
          else
            DestTy = I.getOperand(0)->getType()->getPointerTo();

          BitCast = Builder.CreateBitCast(SPI8Ptr, DestTy);
          I.setOperand(isa<LoadInst>(&I) ? 0 : 1, BitCast);

          DirectStackAccesses.emplace_back(&I);
        } else if (match(Pointer, m_c_Add(m_Value(LHS), m_ConstantInt(Offset)))
                   && LHS == LoadSP) {
          // Do we have a inttoptr whose pointer operand is an instruction `add
          // i64 LHS, X` with X negative constant and LHS the stack pointer? If
          // so, canonicalize the i2p + add into a gep whose value is bitcasted
          // to the original type of SP.
          auto *GEP = Builder.CreateGEP(Builder.getInt8Ty(), SPI8Ptr, Offset);

          Type *DestTy = nullptr;
          if (isa<LoadInst>(&I))
            DestTy = I.getOperand(0)->getType();
          else
            DestTy = I.getOperand(0)->getType()->getPointerTo();

          BitCast = Builder.CreateBitCast(GEP, DestTy);
          I.setOperand(isa<LoadInst>(&I) ? 0 : 1, BitCast);

          DirectStackAccesses.emplace_back(&I);
        } else {
          NotDirectStackAccesses.emplace_back(&I);
        }
      }
    }
  }
}

void SDSAPI::decorateStackAccesses() {
  MDBuilder MDB(*Context);

  MDNode *AliasDomain = MDB.createAliasScopeDomain("CSVAliasDomain");

  // Create two different domains, one will be populated with memory operations
  // that directly access the stack; the other one, with all the remaining kind
  // of accesses (to the CSVs, to the heap, etc.).
  auto *DSAScope = MDB.createAliasScope("DirectStackAccessScope", AliasDomain);
  auto *NDSAScope = MDB.createAliasScope("Not(DirectStackAccessScope)",
                                         AliasDomain);

  auto *DSASet = MDNode::get(*Context, ArrayRef<Metadata *>({ DSAScope }));
  auto *NDSASet = MDNode::get(*Context, ArrayRef<Metadata *>({ NDSAScope }));

  for (auto *I : DirectStackAccesses) {
    I->setMetadata(LLVMContext::MD_alias_scope, DSASet);
    I->setMetadata(LLVMContext::MD_noalias, NDSASet);
  }

  for (auto *I : NotDirectStackAccesses) {
    I->setMetadata(LLVMContext::MD_alias_scope, NDSASet);
    I->setMetadata(LLVMContext::MD_noalias, DSASet);
  }
}
