#ifndef GENERATEDCODEBASICINFO_H
#define GENERATEDCODEBASICINFO_H

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <cstdint>
#include <map>
#include <utility>

// LLVM includes
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"

// Local libraries includes
#include "revng/Support/IRHelpers.h"
#include "revng/Support/revng.h"

// Forward declarations
namespace llvm {
class BasicBlock;
class GlobalVariable;
class Instruction;
class MDNode;
} // namespace llvm

static const char *BlockTypeMDName = "revng.block.type";
static const char *JTReasonMDName = "revng.jt.reasons";

namespace BlockType {

/// \brief Classification of the various basic blocks we are creating
enum Values {
  /// A basic block generated during translation representing a jump target
  JumpTargetBlock,

  // TODO: UntypedBlock is a bad name
  /// A basic block generated during translation that it's not a jump target
  UntypedBlock,

  /// Basic block representing the dispatcher
  DispatcherBlock,

  /// Basic block used to handle an expectedly unknown jump target
  AnyPCBlock,

  /// Basic block used to handle an unexpectedly unknown jump target
  UnexpectedPCBlock,

  /// Basic block representing the default case of the dispatcher switch
  DispatcherFailureBlock,

  /// Basic block to handle jumps to non-translated code
  ExternalJumpsHandlerBlock,

  /// The entry point of the root function
  EntryPoint
};

inline const char *getName(Values Reason) {
  switch (Reason) {
  case JumpTargetBlock:
    return "JumpTargetBlock";
  case UntypedBlock:
    return "UntypedBlock";
  case DispatcherBlock:
    return "DispatcherBlock";
  case AnyPCBlock:
    return "AnyPCBlock";
  case UnexpectedPCBlock:
    return "UnexpectedPCBlock";
  case DispatcherFailureBlock:
    return "DispatcherFailureBlock";
  case ExternalJumpsHandlerBlock:
    return "ExternalJumpsHandlerBlock";
  case EntryPoint:
    return "EntryPoint";
  }

  revng_abort();
}

inline Values fromName(llvm::StringRef ReasonName) {
  if (ReasonName == "JumpTargetBlock")
    return JumpTargetBlock;
  else if (ReasonName == "UntypedBlock")
    return UntypedBlock;
  else if (ReasonName == "DispatcherBlock")
    return DispatcherBlock;
  else if (ReasonName == "AnyPCBlock")
    return AnyPCBlock;
  else if (ReasonName == "UnexpectedPCBlock")
    return UnexpectedPCBlock;
  else if (ReasonName == "DispatcherFailureBlock")
    return DispatcherFailureBlock;
  else if (ReasonName == "ExternalJumpsHandlerBlock")
    return ExternalJumpsHandlerBlock;
  else if (ReasonName == "EntryPoint")
    return EntryPoint;
  else
    revng_abort();
}

} // namespace BlockType

inline void setBlockType(llvm::TerminatorInst *T, BlockType::Values Value) {
  QuickMetadata QMD(getContext(T));
  T->setMetadata(BlockTypeMDName, QMD.tuple(BlockType::getName(Value)));
}

inline llvm::BasicBlock *
findByBlockType(llvm::Function *F, BlockType::Values Value) {
  using namespace llvm;
  QuickMetadata QMD(getContext(F));
  for (BasicBlock &BB : *F) {
    if (auto *T = BB.getTerminator()) {
      auto *MD = T->getMetadata(BlockTypeMDName);
      if (auto *Node = cast_or_null<MDTuple>(MD))
        if (BlockType::fromName(QMD.extract<StringRef>(Node, 0)) == Value)
          return &BB;
    }
  }

  return nullptr;
}

/// \brief Pass to collect basic information about the generated code
///
/// This pass provides useful information for other passes by extracting them
/// from the generated IR, and possibly caching them.
///
/// It provides details about the input architecture such as the size of its
/// delay slot, the name of the program counter register and so on. It also
/// provides information about the generated basic blocks, distinguishing
/// between basic blocks generated due to translation and dispatcher-related
/// basic blocks.
class GeneratedCodeBasicInfo : public llvm::ModulePass {
public:
  static char ID;

public:
  GeneratedCodeBasicInfo() :
    llvm::ModulePass(ID),
    InstructionAlignment(0),
    DelaySlotSize(0),
    PC(nullptr),
    Dispatcher(nullptr),
    DispatcherFail(nullptr),
    AnyPC(nullptr),
    UnexpectedPC(nullptr),
    PCRegSize(0),
    RootFunction(nullptr) {}

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
    AU.setPreservesAll();
  }

  bool runOnModule(llvm::Module &M) override;

  /// \brief Return the type of basic block, see BlockType.
  BlockType::Values getType(llvm::BasicBlock *BB) const {
    return getType(BB->getTerminator());
  }

  BlockType::Values getType(llvm::TerminatorInst *T) const {
    using namespace llvm;

    revng_assert(T != nullptr);
    MDNode *MD = T->getMetadata(BlockTypeMDName);

    BasicBlock *BB = T->getParent();
    if (BB == &BB->getParent()->getEntryBlock())
      return BlockType::EntryPoint;

    if (MD == nullptr) {
      Instruction *First = &*T->getParent()->begin();
      if (CallInst *Call = getCallTo(First, "newpc"))
        if (getLimitedValue(Call->getArgOperand(2)) == 1)
          return BlockType::JumpTargetBlock;

      return BlockType::UntypedBlock;
    }

    auto *BlockTypeMD = cast<MDTuple>(MD);

    QuickMetadata QMD(getContext(T));
    return BlockType::fromName(QMD.extract<llvm::StringRef>(BlockTypeMD, 0));
  }

  uint32_t getJTReasons(llvm::BasicBlock *BB) const {
    return getJTReasons(BB->getTerminator());
  }

  uint32_t getJTReasons(llvm::TerminatorInst *T) const {
    using namespace llvm;
    uint32_t Result = 0;

    MDNode *Node = T->getMetadata(JTReasonMDName);
    auto *Tuple = cast_or_null<MDTuple>(Node);
    revng_assert(Tuple != nullptr);

    for (Metadata *ReasonMD : Tuple->operands()) {
      StringRef Text = cast<MDString>(ReasonMD)->getString();
      Result |= static_cast<uint32_t>(JTReason::fromName(Text));
    }

    return Result;
  }

  KillReason::Values getKillReason(llvm::BasicBlock *BB) const {
    return getKillReason(BB->getTerminator());
  }

  KillReason::Values getKillReason(llvm::TerminatorInst *T) const {
    using namespace llvm;

    auto *NoReturnMD = T->getMetadata("noreturn");
    if (auto *NoreturnTuple = dyn_cast_or_null<MDTuple>(NoReturnMD)) {
      QuickMetadata QMD(getContext(T));
      return KillReason::fromName(QMD.extract<StringRef>(NoreturnTuple, 0));
    }

    return KillReason::NonKiller;
  }

  bool isKiller(llvm::BasicBlock *BB) const {
    return isKiller(BB->getTerminator());
  }

  bool isKiller(llvm::TerminatorInst *T) const {
    return getKillReason(T) != KillReason::NonKiller;
  }

  /// \brief Return the value to which instructions must be aligned in the input
  ///        architecture
  unsigned instructionAlignment() const { return InstructionAlignment; }

  /// \brief Return the size of the delay slot for the input architecture
  unsigned delaySlotSize() const { return DelaySlotSize; }

  /// \brief Return the CSV representing the stack pointer
  llvm::GlobalVariable *spReg() const { return SP; }

  /// \brief Check if \p GV is the stack pointer CSV
  bool isSPReg(const llvm::GlobalVariable *GV) const {
    revng_assert(SP != nullptr);
    return GV == SP;
  }

  bool isSPReg(const llvm::Value *V) const {
    auto *GV = llvm::dyn_cast<const llvm::GlobalVariable>(V);
    if (GV != nullptr)
      return isSPReg(GV);
    return false;
  }

  /// \brief Return the CSV representing the program counter
  llvm::GlobalVariable *pcReg() const { return PC; }

  unsigned pcRegSize() const { return PCRegSize; }

  /// \brief Check if \p GV is the program counter CSV
  bool isPCReg(const llvm::GlobalVariable *GV) const {
    revng_assert(PC != nullptr);
    return GV == PC;
  }

  /// \brief Return the basic block associated to \p PC
  ///
  /// Returns nullptr if the PC doesn't have a basic block (yet)
  llvm::BasicBlock *getBlockAt(uint64_t PC) const {
    auto It = JumpTargets.find(PC);
    if (It == JumpTargets.end())
      return nullptr;

    return It->second;
  }

  /// \brief Return true if the basic block is a jump target
  bool isJumpTarget(llvm::BasicBlock *BB) const {
    return getType(BB->getTerminator()) == BlockType::JumpTargetBlock;
  }

  bool isJump(llvm::BasicBlock *BB) const {
    return isJump(BB->getTerminator());
  }

  /// \brief Return true if \p T represents a jump in the input assembly
  ///
  /// Return true if \p T targets include only dispatcher-related basic blocks
  /// and jump targets.
  bool isJump(llvm::TerminatorInst *T) const {
    revng_assert(T != nullptr);

    for (llvm::BasicBlock *Successor : T->successors()) {
      if (not(Successor->empty() or Successor == Dispatcher
              or Successor == DispatcherFail or Successor == AnyPC
              or Successor == UnexpectedPC or isJumpTarget(Successor)))
        return false;
    }

    return true;
  }

  /// \brief Return true if \p BB is the result of translating some code
  ///
  /// Return false if \p BB is a dispatcher-related basic block.
  bool isTranslated(llvm::BasicBlock *BB) const {
    BlockType::Values Type = getType(BB);
    return (Type == BlockType::UntypedBlock
            or Type == BlockType::JumpTargetBlock);
  }

  /// \brief Find the PC which lead to generated \p TheInstruction
  ///
  /// \return a pair of integers: the first element represents the PC and the
  ///         second the size of the instruction.
  std::pair<uint64_t, uint64_t> getPC(llvm::Instruction *TheInstruction) const;

  /// \brief Return the program counter of the next (i.e., fallthrough)
  ///        instruction of \p TheInstruction
  uint64_t getNextPC(llvm::Instruction *TheInstruction) const {
    auto Pair = getPC(TheInstruction);
    return Pair.first + Pair.second;
  }

  llvm::CallInst *getFunctionCall(llvm::BasicBlock *BB) const {
    return getFunctionCall(BB->getTerminator());
  }

  // TODO: is this a duplication of FunctionCallIdentification::isCall?
  // TODO: we could unpack the information too
  llvm::CallInst *getFunctionCall(llvm::TerminatorInst *T) const {
    auto It = T->getIterator();
    auto End = T->getParent()->begin();
    while (It != End) {
      It--;
      if (llvm::CallInst *Call = getCallTo(&*It, "function_call"))
        return Call;

      if (not isMarker(&*It))
        return nullptr;
    }

    return nullptr;
  }

  bool isFunctionCall(llvm::BasicBlock *BB) const {
    return isFunctionCall(BB->getTerminator());
  }

  bool isFunctionCall(llvm::TerminatorInst *T) const {
    return getFunctionCall(T) != nullptr;
  }

  llvm::BasicBlock *anyPC() { return AnyPC; }
  llvm::BasicBlock *unexpectedPC() { return UnexpectedPC; }

  const llvm::ArrayRef<llvm::GlobalVariable *> csvs() const { return CSVs; }

  class CSVsUsedByHelperCall {
  public:
    void sort() {
      std::sort(Read.begin(), Read.end());
      std::sort(Written.begin(), Written.end());
    }

  public:
    std::vector<llvm::GlobalVariable *> Read;
    std::vector<llvm::GlobalVariable *> Written;
  };

  static CSVsUsedByHelperCall getCSVUsedByHelperCall(llvm::CallInst *Call) {
    revng_assert(isCallToHelper(Call));
    CSVsUsedByHelperCall Result;
    Result.Read = extractCSVs(Call, "revng.csvaccess.offsets.load");
    Result.Written = extractCSVs(Call, "revng.csvaccess.offsets.store");
    return Result;
  }

private:
  static std::vector<llvm::GlobalVariable *>
  extractCSVs(llvm::CallInst *Call, const char *MetadataKind) {
    using namespace llvm;

    std::vector<GlobalVariable *> Result;
    auto *Tuple = cast_or_null<MDTuple>(Call->getMetadata(MetadataKind));
    if (Tuple == nullptr)
      return Result;

    QuickMetadata QMD(getContext(Call));

    auto OperandsRange = QMD.extract<MDTuple *>(Tuple, 1)->operands();
    for (const MDOperand &Operand : OperandsRange) {
      auto *CSV = QMD.extract<Constant *>(Operand.get());
      Result.push_back(cast<GlobalVariable>(CSV));
    }

    return Result;
  }

private:
  uint32_t InstructionAlignment;
  uint32_t DelaySlotSize;
  llvm::GlobalVariable *PC;
  llvm::GlobalVariable *SP;
  llvm::BasicBlock *Dispatcher;
  llvm::BasicBlock *DispatcherFail;
  llvm::BasicBlock *AnyPC;
  llvm::BasicBlock *UnexpectedPC;
  std::map<uint64_t, llvm::BasicBlock *> JumpTargets;
  unsigned PCRegSize;
  llvm::Function *RootFunction;
  std::vector<llvm::GlobalVariable *> CSVs;
};

template<>
struct BlackListTrait<const GeneratedCodeBasicInfo &, llvm::BasicBlock *>
  : BlackListTraitBase<const GeneratedCodeBasicInfo &> {
  using BlackListTraitBase<const GeneratedCodeBasicInfo &>::BlackListTraitBase;
  bool isBlacklisted(llvm::BasicBlock *Value) const {
    return !this->Obj.isTranslated(Value);
  }
};

#endif // GENERATEDCODEBASICINFO_H
