//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Pass.h"

#include "revng/Model/Binary.h"
#include "revng/Model/LoadModelPass.h"
#include "revng/Support/MetaAddress.h"
#include "revng/Support/OpaqueFunctionsPool.h"

#include "revng-c/Support/FunctionTags.h"
#include "revng-c/Support/IRHelpers.h"
#include "revng-c/Support/ModelHelpers.h"

using namespace llvm;

struct MakeSegmentRefPass : public FunctionPass {
public:
  static char ID;

  MakeSegmentRefPass() : FunctionPass(ID) {}

  bool runOnFunction(Function &F) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<LoadModelWrapperPass>();
  }
};

static std::optional<std::pair<MetaAddress, uint64_t>>
findLiteralInSegments(const model::Binary &Binary, uint64_t Literal) {
  std::optional<std::pair<MetaAddress, uint64_t>> Result = std::nullopt;
  auto Arch = toLLVMArchitecture(Binary.Architecture());

  for (const auto &Segment : Binary.Segments()) {
    if (Segment.contains(MetaAddress::fromGeneric(Arch, Literal))) {
      revng_assert(not Result.has_value());
      Result = { { Segment.StartAddress(), Segment.VirtualSize() } };
    }
  }

  return Result;
}

bool MakeSegmentRefPass::runOnFunction(Function &F) {
  using CI = llvm::ConstantInt;

  llvm::LLVMContext &Context = F.getContext();
  Module *M = F.getParent();

  OpaqueFunctionsPool<SegmentRefPoolKey> SegmentRefPool(M, false);
  initSegmentRefPool(SegmentRefPool, M);
  OpaqueFunctionsPool<TypePair> AddressOfPool(M, false);
  initAddressOfPool(AddressOfPool, M);

  auto &ModelWrapper = getAnalysis<LoadModelWrapperPass>().get();
  const TupleTree<model::Binary> &Model = ModelWrapper.getReadOnlyModel();

  bool Changed = false;
  IRBuilder<> IRB(Context);
  llvm::Type *PtrSizedInteger = getPointerSizedInteger(Context, *Model);

  for (Instruction &I : instructions(F)) {
    for (Use &Op : I.operands()) {
      if (isa<SwitchInst>(&I) && (cast<SwitchInst>(&I)->getCondition() != Op))
        continue;

      if (auto *BI = dyn_cast<BranchInst>(&I))
        if (BI->isConditional())
          if (BI->getCondition() != Op)
            continue;

      ConstantInt *ConstOp = dyn_cast<CI>(skipCasts(Op));
      if (ConstOp != nullptr and ConstOp->getBitWidth() <= 64) {
        uint64_t Literal = ConstOp->getZExtValue();

        if (auto Segment = findLiteralInSegments(*Model, Literal); Segment) {
          const auto &[StartAddress, VirtualSize] = *Segment;

          if (isa<PHINode>(&I)) {
            auto *BB = cast<PHINode>(&I)->getIncomingBlock(Op);
            IRB.SetInsertPoint(BB->getTerminator());
          } else {
            IRB.SetInsertPoint(&I);
          }

          IntegerType *OperandType = ConstOp->getType();
          SegmentRefPoolKey Key = { StartAddress, VirtualSize, OperandType };
          auto *SegmentRefFunction = SegmentRefPool.get(Key,
                                                        OperandType,
                                                        {},
                                                        "segmentRef");

          if (not hasSegmentKeyMetadata(*SegmentRefFunction)) {
            setSegmentKeyMetadata(*SegmentRefFunction,
                                  StartAddress,
                                  VirtualSize);
          }

          // Inject call to SegmentRef
          auto *SegmentRefCall = IRB.CreateCall(SegmentRefFunction);

          // Inject call to AddressOf
          auto SegmentRefType = SegmentRefCall->getType();
          auto *AddressOfFunctionType = getAddressOfType(PtrSizedInteger,
                                                         SegmentRefType);
          auto *AddressOfFunction = AddressOfPool.get({ PtrSizedInteger,
                                                        SegmentRefType },
                                                      AddressOfFunctionType,
                                                      "AddressOf");
          auto SegmentType = Model->Segments()
                               .at({ StartAddress, VirtualSize })
                               .Type();
          Constant *ModelTypeString = serializeToLLVMString(SegmentType, *M);
          Value *AddressOfCall = IRB.CreateCall(AddressOfFunction,
                                                { ModelTypeString,
                                                  SegmentRefCall });
          AddressOfCall = IRB.CreateZExtOrTrunc(AddressOfCall, OperandType);

          // Inject LLVM add
          auto Delta = Literal - StartAddress.address();
          if (Delta > 0) {
            auto *Offset = ConstantInt::get(OperandType, Delta);
            AddressOfCall = IRB.CreateAdd(AddressOfCall, Offset);
          }

          if (auto *CE = dyn_cast<ConstantExpr>(Op))
            if (CE->getType()->isPointerTy())
              AddressOfCall = IRB.CreateIntToPtr(AddressOfCall, CE->getType());

          I.setOperand(Op.getOperandNo(), AddressOfCall);

          Changed = true;
        }
      }
    }
  }

  return Changed;
}

char MakeSegmentRefPass::ID = 0;

static constexpr const char *Flag = "make-segment-ref";

using Reg = RegisterPass<MakeSegmentRefPass>;
static Reg X(Flag,
             "Replace literals and global constants with an opaque call using "
             "segments");
