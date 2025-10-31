/// \file FixHelpers.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <cstdint>
#include <cstdlib>
#include <limits>
#include <optional>
#include <stdexcept>

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/NativeFormatting.h"

#include "revng/HelperArgumentsAnalysis/Annotation.h"
#include "revng/Lift/LibTcg.h"
#include "revng/Lift/VariableManager.h"
#include "revng/Support/CommandLine.h"
#include "revng/Support/Debug.h"
#include "revng/Support/IRBuilder.h"
#include "revng/Support/IRHelpers.h"

#include "revng/Model/Generated/Early/Architecture.h"

using namespace llvm;

using std::string;

static Logger Log("fix-helpers");

static cl::opt<string> ArchitectureName("fix-helpers-architecture",
                                        cl::desc("architecture of the helper "
                                                 "of the input file for "
                                                 "fix-helpers."),
                                        cl::cat(MainCategory));

static void setMetadata(VariableManager &Variables,
                        Function &F,
                        StringRef MetadataName,
                        const aua::Annotation::OffsetAndSizeSet &Offsets) {
  revng_log(Log, "Parsing " << MetadataName << ":");
  LoggerIndent Indent(Log);

  QuickMetadata QMD(getContext(&F));
  SmallVector<Metadata *, 2> CSVList;

  for (auto [Offset, Size] : Offsets) {
    int32_t RemainingSize = Size;
    while (RemainingSize != 0) {
      auto [CSV,
            OffsetInField] = Variables.getByCPUStateOffsetWithRemainder(Offset);

      if (CSV == nullptr) {
        revng_log(Log, "Couldn't find CSV for offset " << Offset);
        break;
      }

      revng_log(Log, CSV->getName());

      revng_assert(CSV->getParent() == F.getParent());

      auto CSVSize = CSV->getValueType()->getIntegerBitWidth() / 8;
      CSVList.push_back(QMD.get(CSV->getName()));

      // If necessary, move on to the next field
      int64_t AvailableSize = CSVSize - OffsetInField;
      RemainingSize = std::max(0L, RemainingSize - AvailableSize);
      Offset = 0;
    }
  }

  if (Offsets.size() > 0) {
    revng_log(Log,
              "Identified " << CSVList.size() << " CSVs out of "
                            << Offsets.size() << " offsets");
  }

  MDNode *MDAnnotation = QMD.tuple({ QMD.get(0), QMD.tuple(CSVList) });
  F.setMetadata(MetadataName, MDAnnotation);
}

static void convertToCSVAnnotation(VariableManager &Variables,
                                   Function &F,
                                   const aua::Annotation &Annotation) {
  revng_log(Log, "Converting CSV to annotations");
  LoggerIndent Indent(Log);
  setMetadata(Variables, F, "revng.csvaccess.offsets.load", Annotation.Reads);
  setMetadata(Variables, F, "revng.csvaccess.offsets.store", Annotation.Writes);
}

class AccessFixer {
public:
  enum AccessType {
    Read,
    Write
  };

private:
  VariableManager &Variables;
  Type &IntPtrType;
  revng::NonDebugInfoCheckingIRBuilder Builder;
  Function &Abort;

public:
  AccessFixer(Module &M, VariableManager &Variables, Type &IntPtrType) :
    Variables(Variables),
    IntPtrType(IntPtrType),
    Builder(IntPtrType.getContext()),
    Abort(notNull(M.getFunction("abort"))) {}

public:
  void fixMemoryAccess(Instruction &I, const aua::Annotation &Annotation);

  CallInst *emitAbort(Function &F) {
    Builder.SetInsertPointPastAllocas(&F);
    auto *AbortCall = Builder.CreateCall(&Abort);
    AbortCall->setMetadata("dbg", nullptr);
    return AbortCall;
  }

private:
  template<AccessType Type>
  void handle(Instruction &I, const aua::Annotation::OffsetAndSizeSet &Targets);

  template<AccessType Type>
  Value *emit(Instruction &I, uint64_t Offset, uint64_t Size);

  CallInst *emitAbort(revng::IRBuilder &Builder) {
    return Builder.CreateCall(&Abort);
  }

  CallInst *emitAbort(Instruction &Before) {
    Builder.SetInsertPoint(&Before);
    return emitAbort(Builder);
  }

  std::pair<CallInst &, CallInst &> decomposeMemcpy(llvm::Instruction &I);

  template<AccessType Type>
  std::pair<Value *, uint64_t> getOffsetAndSize(Instruction &I);
};

template<>
Value *AccessFixer::emit<AccessFixer::Write>(Instruction &I,
                                             uint64_t Offset,
                                             uint64_t Size) {
  revng_log(Log, "Handling a write to " << Offset << " of size " << Size);

  if (auto *Store = dyn_cast<StoreInst>(&I)) {
    Value *ToStore = Store->getValueOperand();

    auto MaybeStore = Variables.storeToCPUStateOffset(Builder,
                                                      Size,
                                                      Offset,
                                                      ToStore);
    if (MaybeStore) {
      return *MaybeStore;
    } else {
      // storeToCPUStateOffset can fail in case of padding and the like
      revng_log(Log, "Nothing emitted");
      return nullptr;
    }
  } else if (auto *Call = dyn_cast<CallInst>(&I)) {
    Variables.memOpAtCPUStateOffset(Builder, Call, Offset, false);
    return nullptr;
  } else {
    revng_abort();
  }
}

template<>
Value *AccessFixer::emit<AccessFixer::Read>(Instruction &I,
                                            uint64_t Offset,
                                            uint64_t Size) {
  revng_log(Log, "Handling a read from " << Offset << " of size " << Size);
  if (auto *Load = dyn_cast<LoadInst>(&I)) {
    if (auto *Result = Variables.loadFromCPUStateOffset(Builder,
                                                        Size,
                                                        Offset)) {
      return Result;
    } else {
      // loadFromCPUStateOffset can fail in case of padding and the like
      revng_log(Log, "Nothing emitted");
      return llvm::UndefValue::get(I.getType());
    }
  } else if (auto *Call = dyn_cast<CallInst>(&I)) {
    Variables.memOpAtCPUStateOffset(Builder, Call, Offset, true);
    return nullptr;

  } else {
    revng_abort();
  }
}

template<>
std::pair<Value *, uint64_t>
AccessFixer::getOffsetAndSize<AccessFixer::Write>(Instruction &I) {
  uint64_t Size = 0;
  Value *Address = nullptr;
  if (auto *Store = dyn_cast<StoreInst>(&I)) {
    Address = Store->getPointerOperand();
    Size = Store->getValueOperand()->getType()->getIntegerBitWidth() / 8;
  } else if (auto *Call = dyn_cast<CallInst>(&I)) {
    Address = Call->getArgOperand(0);
    Size = cast<ConstantInt>(Call->getOperand(2))->getLimitedValue();
  } else {
    revng_abort();
  }
  return { Builder.CreatePtrToInt(Address, &IntPtrType), Size };
}

template<>
std::pair<Value *, uint64_t>
AccessFixer::getOffsetAndSize<AccessFixer::Read>(Instruction &I) {
  uint64_t Size = 0;
  Value *Address = nullptr;
  if (auto *Load = dyn_cast<LoadInst>(&I)) {
    Address = Load->getPointerOperand();
    Size = Load->getType()->getIntegerBitWidth() / 8;
  } else if (auto *Call = dyn_cast<CallInst>(&I)) {
    Address = Call->getArgOperand(1);
    Size = cast<ConstantInt>(Call->getOperand(2))->getLimitedValue();
  } else {
    revng_abort();
  }
  return { Builder.CreatePtrToInt(Address, &IntPtrType), Size };
}

template<AccessFixer::AccessType Type>
void AccessFixer::handle(Instruction &I,
                         const aua::Annotation::OffsetAndSizeSet &Targets) {
  LLVMContext &Context = getContext(&I);
  Function *F = I.getFunction();

  Builder.SetInsertPoint(&I);
  auto [Address, Size] = getOffsetAndSize<Type>(I);
  auto *AddressType = cast<IntegerType>(Address->getType());

  revng_log(Log,
            "Handling " << getName(&I) << ", which has " << Targets.size()
                        << " targets");
  LoggerIndent Indent(Log);

  // Handle the single-offset situation
  if (Targets.size() == 1) {
    Value *Result = emit<Type>(I, Targets.begin()->first, Size);
    if (Result != nullptr)
      I.replaceAllUsesWith(Result);
    return;
  }

  // There are multiple offsets. We need to emit a switch.

  // Split the block
  BasicBlock *Before = I.getParent();
  BasicBlock *After = Before->splitBasicBlock(I.getIterator());
  Before->getTerminator()->eraseFromParent();

  // Create the default case
  BasicBlock *DefaultCase = BasicBlock::Create(Context, "", F, After);
  Builder.SetInsertPoint(DefaultCase);
  emitAbort(*Builder.CreateUnreachable());

  // Create the switch
  Builder.SetInsertPoint(Before);
  auto *Switch = Builder.CreateSwitch(Address, DefaultCase, Targets.size());
  PHINode *Phi = nullptr;

  // If the instruction provides a result, we also need to emit a phi
  bool HasResult = not I.getType()->isVoidTy();
  if (HasResult)
    Phi = PHINode::Create(I.getType(), Targets.size(), "", &*After->begin());

  // For each target, create a basic block, add a case to the switch and an
  // incoming to the phi
  for (auto &&[Offset, Size] : Targets) {
    BasicBlock *OffsetCase = BasicBlock::Create(Context, "", F, DefaultCase);
    Builder.SetInsertPoint(OffsetCase);
    Value *Result = emit<Type>(I, Offset, Size);
    Builder.CreateBr(After);
    revng_assert(not OffsetCase->empty());

    Switch->addCase(ConstantInt::get(AddressType, Offset), OffsetCase);

    if (HasResult)
      Phi->addIncoming(Result, OffsetCase);
  }

  if (HasResult)
    I.replaceAllUsesWith(Phi);
}

static void
setArgumentAlignment(CallBase &Call, unsigned ArgIdx, unsigned Align) {
  LLVMContext &Context = Call.getContext();
  AttributeList Attrs = Call.getAttributes();

  auto AlignAttribute = Attribute::get(Context, Attribute::Alignment, Align);
  AttributeList NewAttrs = Attrs.addParamAttribute(Context,
                                                   ArgIdx,
                                                   AlignAttribute);

  Call.setAttributes(NewAttrs);
}

std::pair<CallInst &, CallInst &>
AccessFixer::decomposeMemcpy(llvm::Instruction &I) {
  auto &Call = cast<CallInst>(I);
  revng_assert(Call.getIntrinsicID() == Intrinsic::memcpy);
  auto *Size = cast<ConstantInt>(Call.getArgOperand(2));

  BasicBlock &Entry = I.getFunction()->getEntryBlock();
  Builder.SetInsertPoint(&Entry, Entry.begin());
  auto *UInt8Type = IntegerType::getInt8Ty(Abort.getContext());
  auto *Storage = Builder.CreateAlloca(UInt8Type, 0, Size);
  uint64_t Alignment = Storage->getAlign().value();
  auto &Read = Call;
  auto &Write = *cast<CallInst>(Call.clone());
  Write.insertAfter(&Read);

  Read.setArgOperand(0, Storage);
  setArgumentAlignment(Read, 0, Alignment);

  Write.setArgOperand(1, Storage);
  setArgumentAlignment(Write, 1, Alignment);

  return { Read, Write };
}

void AccessFixer::fixMemoryAccess(Instruction &I,
                                  const aua::Annotation &Annotation) {
  revng_log(Log, "Handling memory access " << getName(&I));
  LoggerIndent Indent(Log);

  revng_assert(not I.isTerminator());
  auto *Load = dyn_cast<LoadInst>(&I);
  auto *Store = dyn_cast<StoreInst>(&I);
  if (Annotation.Escapes) {
    revng_log(Log, "It escapes, emitting abort.");
    emitAbort(I);
  } else if (Load != nullptr and Load->getType()->isPointerTy()) {
    revng_log(Log, "Loading a pointer, emitting abort.");
    emitAbort(I);
  } else if (Store != nullptr
             and Store->getValueOperand()->getType()->isPointerTy()) {
    revng_log(Log, "Storing a pointer, emitting abort.");
    emitAbort(I);
  } else {
    revng_log(Log, "It's a memcpy-like.");

    bool IsRead = Annotation.Reads.size() != 0;
    bool IsWrite = Annotation.Writes.size() != 0;

    if (IsRead and IsWrite) {
      revng_log(Log, "Decomposing memcpy.");
      const auto &[Read, Write] = decomposeMemcpy(I);
      handle<AccessFixer::Read>(Read, Annotation.Reads);
      handle<AccessFixer::Write>(Write, Annotation.Writes);

      // Read is I and will be erased later
      Write.eraseFromParent();
    } else if (IsWrite) {
      handle<AccessFixer::Write>(I, Annotation.Writes);
    } else if (IsRead) {
      handle<AccessFixer::Read>(I, Annotation.Reads);
    } else {
      revng_abort();
    }

    I.eraseFromParent();
  }
}

static void fixHelpers(VariableManager &Variables, Module &Module) {
  LLVMContext &Context = Module.getContext();
  auto &DL = Module.getDataLayout();
  AccessFixer Fixer(Module, Variables, *DL.getIntPtrType(Context));

  for (Function &F : Module) {
    revng_log(Log, "Handling " << F.getName());
    LoggerIndent Indent(Log);

    auto Annotation = aua::Annotation::deserialize(F)
                        .value_or(aua::Annotation());

    // Upgrade { Offset, Size } annotations to reference the CSV for
    // consumptions of other passes
    convertToCSVAnnotation(Variables, F, Annotation);

    if (Annotation.Escapes) {
      revng_log(Log, "CPU state escapes, emitting an abort");
      Fixer.emitAbort(F);
    }

    // Go through all instructions and replace memory accesses in env with
    // accesses to CSVs
    SmallVector<Instruction *> Annotated;
    for (Instruction &I : instructions(F))
      if (aua::Annotation::isAnnotated(I))
        Annotated.push_back(&I);

    for (Instruction *I : Annotated) {
      Fixer.fixMemoryAccess(*I, *aua::Annotation::deserialize(*I));
    }
  }

  revng::verify(&Module);
}

class FixHelpers : public llvm::ModulePass {
public:
  static char ID;

public:
  FixHelpers() : llvm::ModulePass(ID) {}

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {}

  bool runOnModule(llvm::Module &M) override {
    auto Architecture = model::Architecture::fromQEMUName(ArchitectureName);
    revng_assert(Architecture != model::Architecture::Invalid);
    auto TheLibTcg = LibTcg::get(Architecture);

    // TODO: fix TargetIsLittleEndian once we support non-x86-64 targets
    VariableManager Variables(M,
                              /* TargetIsLittleEndian */ true,
                              TheLibTcg.archInfo().env_offset,
                              TheLibTcg.envPointer(),
                              TheLibTcg.globalNames());

    fixHelpers(Variables, M);

    return true;
  }
};

char FixHelpers::ID = 0;
using Register = RegisterPass<FixHelpers>;
static Register X("fix-helpers", "Fix helper functions", true, true);
