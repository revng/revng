#ifndef _GENERATEDCODEBASICINFO_H
#define _GENERATEDCODEBASICINFO_H

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

// Local includes
#include "ir-helpers.h"
#include "revamb.h"

// Forward declarations
namespace llvm {
class BasicBlock;
class GlobalVariable;
class Instruction;
class MDNode;
}

static const char *BlockTypeMDName = "revamb.block.type";

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
class GeneratedCodeBasicInfo : public llvm::FunctionPass {
public:
  static char ID;

public:
  GeneratedCodeBasicInfo() : llvm::FunctionPass(ID), DelaySlotSize(0),
                             PC(nullptr), Dispatcher(nullptr),
                             AnyPC(nullptr), UnexpectedPC(nullptr) { }


  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
    AU.setPreservesAll();
  }

  bool runOnFunction(llvm::Function &F) override;

  /// \brief Return the type of basic block, see BlockType.
  BlockType getType(llvm::BasicBlock *BB) const {
    return getType(BB->getTerminator());
  }

  BlockType getType(llvm::TerminatorInst *T) const {
    assert(T != nullptr);
    llvm::MDNode *MD = T->getMetadata(BlockTypeMDName);

    if (MD == nullptr)
      return UntypedBlock;

    auto *BlockTypeMD = llvm::cast<llvm::MDTuple>(MD);

    QuickMetadata QMD(getContext(T));
    return BlockType(QMD.extract<uint32_t>(BlockTypeMD, 0));
  }

  /// \brief Return the size of the delay slot for the input architecture
  unsigned delaySlotSize() const { return DelaySlotSize; }

  /// \brief Return the CSV representing the program counter
  llvm::GlobalVariable *pcReg() const { return PC; }

  /// \brief Check if \p GV is the program counter CSV
  bool isPCReg(llvm::GlobalVariable *GV) const {
    assert(PC != nullptr);
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
    assert(T != nullptr);

    for (llvm::BasicBlock *Successor : T->successors()) {
      if (!(Successor == Dispatcher
            || Successor == AnyPC
            || Successor == UnexpectedPC
            || isJumpTarget(Successor)))
        return false;
    }

    return true;
  }

  /// \brief Return true if \p BB is the result of translating some code
  ///
  /// Return false if \p BB is a dispatcher-related basic block.
  bool isTranslated(llvm::BasicBlock *BB) const {
    return BB != Dispatcher && BB != AnyPC && BB != UnexpectedPC;
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

  /// \brief Calls \p Visitor for each instruction preceeding \p I
  ///
  /// See visitPredecessors in ir-helpers.h
  void visitPredecessors(llvm::Instruction *I, RVisitorFunction Visitor);

private:
  uint32_t DelaySlotSize;
  llvm::GlobalVariable *PC;
  llvm::BasicBlock *Dispatcher;
  llvm::BasicBlock *AnyPC;
  llvm::BasicBlock *UnexpectedPC;
  std::map<uint64_t, llvm::BasicBlock *> JumpTargets;
};

template<>
struct BlackListTrait<const GeneratedCodeBasicInfo &, llvm::BasicBlock *> :
  BlackListTraitBase<const GeneratedCodeBasicInfo &> {
  using BlackListTraitBase<const GeneratedCodeBasicInfo &>::BlackListTraitBase;
  bool isBlacklisted(llvm::BasicBlock *Value) {
    return !this->Obj.isTranslated(Value);
  }
};

inline
void GeneratedCodeBasicInfo::visitPredecessors(llvm::Instruction *I,
                                               RVisitorFunction Visitor) {
  using BLT = BlackListTrait<const GeneratedCodeBasicInfo &,
                             llvm::BasicBlock *>;
  ::visitPredecessors(I, Visitor, BLT(*this));
}

#endif // _GENERATEDCODEBASICINFO_H
