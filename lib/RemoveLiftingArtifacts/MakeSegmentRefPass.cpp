//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <functional>
#include <optional>

#include "llvm/ADT/StringExtras.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FormatVariadic.h"

#include "revng/Model/Architecture.h"
#include "revng/Model/Binary.h"
#include "revng/Model/RawBinaryView.h"
#include "revng/Pipeline/Context.h"
#include "revng/Pipeline/Contract.h"
#include "revng/Pipeline/Kind.h"
#include "revng/Pipeline/LLVMContainer.h"
#include "revng/Pipeline/RegisterPipe.h"
#include "revng/Pipes/FileContainer.h"
#include "revng/Pipes/FunctionPass.h"
#include "revng/Support/Debug.h"
#include "revng/Support/IRHelpers.h"
#include "revng/Support/MetaAddress.h"
#include "revng/Support/OpaqueFunctionsPool.h"

#include "revng-c/Pipes/Kinds.h"
#include "revng-c/Support/FunctionTags.h"
#include "revng-c/Support/ModelHelpers.h"

#include "MakeSegmentRefPass.h"

using namespace llvm;

struct MakeSegmentRefPassImpl : public pipeline::FunctionPassImpl {
private:
  const model::Binary &Binary;
  llvm::Module &M;
  llvm::LLVMContext &Context;
  OpaqueFunctionsPool<SegmentRefPoolKey> SegmentRefPool;
  OpaqueFunctionsPool<TypePair> AddressOfPool;
  OpaqueFunctionsPool<StringLiteralPoolKey> StringLiteralPool;

public:
  MakeSegmentRefPassImpl(llvm::ModulePass &Pass,
                         const model::Binary &Binary,
                         llvm::Module &M) :
    pipeline::FunctionPassImpl(Pass),
    Binary(Binary),
    M(M),
    Context(M.getContext()),
    SegmentRefPool(&M, false),
    AddressOfPool(&M, false),
    StringLiteralPool(&M, false) {
    initSegmentRefPool(SegmentRefPool, &M);
    initAddressOfPool(AddressOfPool, &M);
    initStringLiteralPool(StringLiteralPool, &M);
  }

  bool runOnFunction(const model::Function &ModelFunction,
                     llvm::Function &Function) override;

public:
  static void getAnalysisUsage(llvm::AnalysisUsage &AU);
};

void MakeSegmentRefPassImpl::getAnalysisUsage(llvm::AnalysisUsage &AU) {
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

bool MakeSegmentRefPassImpl::runOnFunction(const model::Function &ModelFunction,
                                           llvm::Function &F) {
  RawBinaryView &BinaryView = getAnalysis<LoadBinaryWrapperPass>().get();

  bool Changed = false;
  IRBuilder<> IRB(Context);
  llvm::Type *PtrSizedInteger = getPointerSizedInteger(Context, Binary);

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
      using namespace model::Architecture;
      auto PointerSize = getPointerSize(Binary.Architecture());
      if (ConstOp != nullptr
          and (ConstOp->getBitWidth() == (8 * PointerSize))) {
        uint64_t ConstantAddress = ConstOp->getZExtValue();

        if (auto Segment = findLiteralInSegments(Binary, ConstantAddress);
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
                                       Str.size(),
                                       RealOpType);
            }

            Value *Call = IRB.CreateCall(LiteralFunction, { ConstExpr });
            I.setOperand(Op.getOperandNo(), Call);

            revng::verify(&M);
          } else {
            SegmentRefPoolKey Key = { StartAddress, VirtualSize, OperandType };
            auto *SegmentRefFunction = SegmentRefPool.get(Key,
                                                          OperandType,
                                                          {},
                                                          "segmentRef");

            auto UniqueIDMDName = FunctionTags::UniqueIDMDName;
            if (SegmentRefFunction->getMetadata(UniqueIDMDName) == nullptr) {
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
            Constant *ModelTypeString = nullptr;
            if (const auto &SegmentType = Binary.Segments()
                                            .at({ StartAddress, VirtualSize })
                                            .Type()) {
              ModelTypeString = toLLVMString(SegmentType, M);
            } else {
              auto Byte = model::PrimitiveType::makeGeneric(1);
              auto Arr = model::ArrayType::make(std::move(Byte), VirtualSize);
              ModelTypeString = toLLVMString(Arr, M);
            }
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

  return Changed;
}

template<>
char pipeline::FunctionPass<MakeSegmentRefPassImpl>::ID = 0;

namespace revng::pipes {

class MakeSegmentRef {
public:
  static constexpr auto Name = "make-segment-ref";

  std::array<pipeline::ContractGroup, 1> getContract() const {
    pipeline::Contract BinaryPart(kinds::Binary, 0, kinds::Binary, 0);
    pipeline::Contract FunctionsPart(kinds::StackAccessesSegregated,
                                     1,
                                     kinds::StackAccessesSegregated,
                                     1);
    return { pipeline::ContractGroup({ BinaryPart, FunctionsPart }) };
  }

  void run(pipeline::ExecutionContext &EC,
           const BinaryFileContainer &SourceBinary,
           pipeline::LLVMContainer &Output);
};

void MakeSegmentRef::run(pipeline::ExecutionContext &EC,
                         const BinaryFileContainer &SourceBinary,
                         pipeline::LLVMContainer &ModuleContainer) {
  if (not SourceBinary.exists())
    return;

  const TupleTree<model::Binary> &Model = getModelFromContext(EC);
  auto BufferOrError = llvm::MemoryBuffer::getFileOrSTDIN(*SourceBinary.path());
  auto Buffer = cantFail(errorOrToExpected(std::move(BufferOrError)));
  RawBinaryView RawBinary(*Model, Buffer->getBuffer());

  llvm::legacy::PassManager PM;
  PM.add(new pipeline::LoadExecutionContextPass(&EC, ModuleContainer.name()));
  PM.add(new LoadModelWrapperPass(Model));
  PM.add(new LoadBinaryWrapperPass(Buffer->getBuffer()));
  PM.add(new pipeline::FunctionPass<MakeSegmentRefPassImpl>);
  PM.run(ModuleContainer.getModule());
}

} // namespace revng::pipes

static pipeline::RegisterPipe<revng::pipes::MakeSegmentRef> RegMSRPipe;
