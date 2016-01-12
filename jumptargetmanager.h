#ifndef _JUMPTARGETMANAGER_H
#define _JUMPTARGETMANAGER_H

// Standard includes
#include <cstdint>
#include <map>
#include <set>

// Forward declarations
namespace llvm {
class BasicBlock;
class Function;
class Instruction;
class LLVMContext;
class Module;
class SwitchInst;
class Value;
}

class JumpTargetManager;

/// \brief Transform constant writes to the PC in jumps
/// This pass looks for all the calls to the ExitTB function calls, looks for
/// the last write to the PC before them, checks if the written value is
/// statically known, and, if so, replaces it with a jump to the corresponding
/// translated code. If the write to the PC is not constant, no action is
/// performed, and the call to ExitTB remains there for delayed handling.
class TranslateDirectBranchesPass : public llvm::FunctionPass {
public:
  static char ID;

  TranslateDirectBranchesPass() : llvm::FunctionPass(ID),
    JTM(nullptr) { }

  TranslateDirectBranchesPass(JumpTargetManager *JTM) :
    FunctionPass(ID),
    JTM(JTM) { }

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const;

  bool runOnFunction(llvm::Function &F) override;

private:
  /// Obtains the absolute address of the PC correspoding to the original
  /// assembly instruction coming after the specified LLVM instruction
  uint64_t getNextPC(llvm::Instruction *TheInstruction);

private:
  llvm::Value *PCReg;
  JumpTargetManager *JTM;
};

class JumpTargetsFromConstantsPass : public llvm::FunctionPass {
public:
  static char ID;

  JumpTargetsFromConstantsPass() : llvm::FunctionPass(ID), JTM(nullptr) { }

  JumpTargetsFromConstantsPass(JumpTargetManager *JTM) :
    llvm::FunctionPass(ID),
    JTM(JTM) { }

  bool runOnFunction(llvm::Function &F) override;

private:
  JumpTargetManager *JTM;
  std::set<llvm::BasicBlock *> Visited;
};

class JumpTargetManager {
public:
  using BlockWithAddress = std::pair<uint64_t, llvm::BasicBlock *>;
  static const BlockWithAddress NoMoreTargets;

public:
  using RangesVector = std::vector<std::pair<uint64_t, uint64_t>>;
  JumpTargetManager(llvm::Function *TheFunction,
                    llvm::Value *PCReg,
                    Architecture& SourceArchitecture,
                    std::vector<SegmentInfo>& Segments);

  /// Handle a new program counter. We might already have a basic block for that
  /// program counter, or we could even have a translation for it. Return one
  /// of these, if appropriate.
  ///
  /// \param PC the new program counter.
  /// \param ShouldContinue an out parameter indicating whether the returned
  ///        basic block was just a placeholder or actually contains a
  ///        translation.
  ///
  /// \return the basic block to use from now on, or null if the program counter
  ///         is not associated to a basic block.
  llvm::BasicBlock *newPC(uint64_t PC, bool& ShouldContinue);

  /// Save the PC-Instruction association for future use (jump target)
  void registerInstruction(uint64_t PC, llvm::Instruction *Instruction);

  /// Save the PC-BasicBlock association for futur use (jump target)
  void registerBlock(uint64_t PC, llvm::BasicBlock *Block);

  void translateIndirectJumps();

  /// \brief Return the most recent instruction writing the program counter
  /// Note that the search is performed only in the current basic block.
  /// The function will assert if the write instruction is not found.
  ///
  /// \return a pointer to the last StoreInst writing the program counter, or
  ///         nullptr if a call to an helper has been found before the write to
  ///         the PC.
  llvm::StoreInst *getPrevPCWrite(llvm::Instruction *TheInstruction);

  llvm::Function *exitTB() { return ExitTB; }

  /// Pop from the list of program counters to explore
  ///
  /// \return a pair containing the PC and the initial block to use, or
  ///         JumpTarget::NoMoreTargets if we're done.
  BlockWithAddress peek();

  /// Return true if there are unexplored jump targets
  bool empty() { return Unexplored.empty(); }

  /// Get or create a block for the given PC
  llvm::BasicBlock *getBlockAt(uint64_t PC);

  llvm::BasicBlock *dispatcher() { return Dispatcher; }

  bool isPCReg(llvm::Value *TheValue) { return TheValue == PCReg; }

private:
  // TODO: instead of a gigantic switch case we could map the original memory
  //       area and write the address of the translated basic block at the jump
  //       target
  void createDispatcher(llvm::Function *OutputFunction,
                        llvm::Value *SwitchOnPtr,
                        bool JumpDirectly);

  template<typename value_type, unsigned endian>
  void findCodePointers(const unsigned char *Start, const unsigned char *End);

  bool isExecutableAddress(uint64_t Address) {
    for (std::pair<uint64_t, uint64_t> Range : ExecutableRanges)
      if (Range.first <= Address && Address < Range.second)
        return true;
    return false;
  }

  void harvest();
private:
  using BlockMap = std::map<uint64_t, llvm::BasicBlock *>;
  using InstructionMap = std::map<uint64_t, llvm::Instruction *>;

  llvm::Module &TheModule;
  llvm::LLVMContext& Context;
  llvm::Function* TheFunction;
  /// Holds the association between a PC and the last generated instruction for
  /// the previous instruction.
  InstructionMap OriginalInstructionAddresses;
  /// Holds the association between a PC and a BasicBlock.
  BlockMap JumpTargets;
  /// Queue of program counters we still have to translate.
  std::vector<BlockWithAddress> Unexplored;
  llvm::Value *PCReg;
  llvm::Function *ExitTB;
  RangesVector ExecutableRanges;
  llvm::BasicBlock *Dispatcher;
  llvm::SwitchInst *DispatcherSwitch;
};

#endif // _JUMPTARGETMANAGER_H
