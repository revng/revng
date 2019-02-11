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
  BlockType getType(llvm::BasicBlock *BB) const {
    return getType(BB->getTerminator());
  }

  BlockType getType(llvm::TerminatorInst *T) const {
    using namespace llvm;

    revng_assert(T != nullptr);
    MDNode *MD = T->getMetadata(BlockTypeMDName);

    BasicBlock *BB = T->getParent();
    if (BB == &BB->getParent()->getEntryBlock())
      return EntryPoint;

    if (MD == nullptr) {
      Instruction *First = &*T->getParent()->begin();
      if (CallInst *Call = getCallTo(First, "newpc"))
        if (getLimitedValue(Call->getArgOperand(2)) == 1)
          return JumpTargetBlock;

      return UntypedBlock;
    }

    auto *BlockTypeMD = cast<MDTuple>(MD);

    QuickMetadata QMD(getContext(T));
    return BlockType(QMD.extract<uint32_t>(BlockTypeMD, 0));
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
    return getType(BB->getTerminator()) == JumpTargetBlock;
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
    BlockType Type = getType(BB);
    return Type == UntypedBlock or Type == JumpTargetBlock;
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
