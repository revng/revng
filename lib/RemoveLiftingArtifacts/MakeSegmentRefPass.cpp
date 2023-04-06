//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

#include <functional>
#include <optional>

#include "llvm/ADT/StringExtras.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FormatVariadic.h"

#include "revng/Model/Binary.h"
#include "revng/Model/RawBinaryView.h"
#include "revng/Support/Debug.h"
#include "revng/Support/MetaAddress.h"
#include "revng/Support/OpaqueFunctionsPool.h"

#include "revng-c/Support/FunctionTags.h"
#include "revng-c/Support/IRHelpers.h"
#include "revng-c/Support/ModelHelpers.h"

#include "MakeSegmentRefPass.h"

using namespace llvm;

void MakeSegmentRefPass::getAnalysisUsage(llvm::AnalysisUsage &AU) const {
  AU.addRequired<LoadBinaryWrapperPass>();
  AU.addRequired<LoadModelWrapperPass>();
}

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

static std::optional<llvm::StringRef>
getStringLiteral(RawBinaryView &BinaryView,
                 MetaAddress SegmentAddress,
                 uint64_t SegmentVirtualSize,
                 uint64_t StringOffsetInSegment) {
  // If the segment is not read only it's not a string literal
  const bool IsReadOnly = BinaryView.isReadOnly(SegmentAddress,
                                                SegmentVirtualSize);
  if (not IsReadOnly)
    return std::nullopt;

  MetaAddress StringAddress = SegmentAddress + StringOffsetInSegment;
  uint64_t SegmentSizeFromString = SegmentVirtualSize - StringOffsetInSegment;

  // If it's not there, it's not a string literal
  auto DataOrNone = BinaryView.getStringByAddress(StringAddress,
                                                  SegmentSizeFromString);
  if (not DataOrNone.has_value())
    return std::nullopt;

  llvm::StringRef StringView = DataOrNone.value();

  // If it doesn't end with \0 it's not a string literal
  auto NullTerminatorPos = StringView.find('\0');
  if (NullTerminatorPos == llvm::StringLiteral::npos)
    return std::nullopt;

  StringView = StringView.take_front(NullTerminatorPos);

  // If some of the characters are not printable, is not a string literal
  const auto IsPrintableInCString = [](const char C) {
    return not llvm::isPrint(C) and not llvm::isSpace(C);
  };
  if (llvm::any_of(StringView, IsPrintableInCString))
    return std::nullopt;

  return StringView;
}

bool MakeSegmentRefPass::runOnModule(Module &M) {
  llvm::LLVMContext &Context = M.getContext();

  OpaqueFunctionsPool<SegmentRefPoolKey> SegmentRefPool(&M, false);
  initSegmentRefPool(SegmentRefPool, &M);

  OpaqueFunctionsPool<TypePair> AddressOfPool(&M, false);
  initAddressOfPool(AddressOfPool, &M);

  OpaqueFunctionsPool<StringLiteralPoolKey> StringLiteralPool(&M, false);
  initStringLiteralPool(StringLiteralPool, &M);

  auto &ModelWrapper = getAnalysis<LoadModelWrapperPass>().get();
  const TupleTree<model::Binary> &Model = ModelWrapper.getReadOnlyModel();

  RawBinaryView &BinaryView = getAnalysis<LoadBinaryWrapperPass>().get();

  bool Changed = false;
  IRBuilder<> IRB(Context);
  llvm::Type *PtrSizedInteger = getPointerSizedInteger(Context, *Model);

  for (Function &F : M) {
    for (Instruction &I : instructions(F)) {

      if (isCallToTagged(&I, FunctionTags::AllocatesLocalVariable))
        continue;

      for (Use &Op : I.operands()) {
        if (isa<SwitchInst>(&I) && (cast<SwitchInst>(&I)->getCondition() != Op))
          continue;

        if (auto *BI = dyn_cast<BranchInst>(&I))
          if (BI->isConditional())
            if (BI->getCondition() != Op)
              continue;

        ConstantInt *ConstOp = dyn_cast<ConstantInt>(skipCasts(Op));
        if (ConstOp != nullptr and ConstOp->getBitWidth() <= 64) {
          uint64_t ConstantAddress = ConstOp->getZExtValue();

          if (auto Segment = findLiteralInSegments(*Model, ConstantAddress);
              Segment) {
            const auto &[StartAddress, VirtualSize] = *Segment;
            auto OffsetInSegment = ConstantAddress - StartAddress.address();

            if (isa<PHINode>(&I)) {
              auto *BB = cast<PHINode>(&I)->getIncomingBlock(Op);
              IRB.SetInsertPoint(BB->getTerminator());
            } else {
              IRB.SetInsertPoint(&I);
            }

            IntegerType *OperandType = ConstOp->getType();

            // Check if the use of this constant is a icmp. If it is we cannot
            // replace it with a string literal, because comparisons between
            // string literals are undefined behavior in C.
            bool UseIsComparison = llvm::isa<llvm::CmpInst>(Op.getUser());
            // Check if the Op is large as a pointer. If it isn't it can't be a
            // string literal.
            // See if we can find a string literal there.
            auto OptString = getStringLiteral(BinaryView,
                                              StartAddress,
                                              VirtualSize,
                                              OffsetInSegment);

            if (not UseIsComparison and OptString.has_value()) {
              auto Str = OptString.value();
              Constant *ConstExpr = getUniqueString(&M, Str);

              auto RealOpType = Op->getType();
              StringLiteralPoolKey Key = {
                StartAddress, VirtualSize, OffsetInSegment, RealOpType
              };

              auto LiteralFunction = StringLiteralPool
                                       .get(Key,
                                            RealOpType,
                                            { ConstExpr->getType() },
                                            "cstringLiteral");

              if (not hasStringLiteralMetadata(*LiteralFunction)) {
                setStringLiteralMetadata(*LiteralFunction,
                                         StartAddress,
                                         VirtualSize,
                                         OffsetInSegment,
                                         Str.size());
              }

              Value *Call = IRB.CreateCall(LiteralFunction, { ConstExpr });
              I.setOperand(Op.getOperandNo(), Call);

              revng_assert(llvm::verifyModule(M, &llvm::dbgs()) == 0);
            } else {
              SegmentRefPoolKey Key = { StartAddress,
                                        VirtualSize,
                                        OperandType };
              auto *SegmentRefFunction = SegmentRefPool.get(Key,
                                                            OperandType,
                                                            {},
                                                            "segmentRef");

              if (SegmentRefFunction->getMetadata(SegmentRefMDName) == nullptr)
                setSegmentKeyMetadata(*SegmentRefFunction,
                                      StartAddress,
                                      VirtualSize);

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
              Constant *ModelTypeString = serializeToLLVMString(SegmentType, M);
              Value *AddressOfCall = IRB.CreateCall(AddressOfFunction,
                                                    { ModelTypeString,
                                                      SegmentRefCall });
              AddressOfCall = IRB.CreateZExtOrTrunc(AddressOfCall, OperandType);

              // Inject LLVM add
              if (OffsetInSegment > 0) {
                auto *Offset = ConstantInt::get(OperandType, OffsetInSegment);
                AddressOfCall = IRB.CreateAdd(AddressOfCall, Offset);
              }

              if (auto *CE = dyn_cast<ConstantExpr>(Op))
                if (CE->getType()->isPointerTy())
                  AddressOfCall = IRB.CreateIntToPtr(AddressOfCall,
                                                     CE->getType());

              I.setOperand(Op.getOperandNo(), AddressOfCall);
            }

            Changed = true;
          }
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
