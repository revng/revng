#ifndef _JUMPTARGETMANAGER_H
#define _JUMPTARGETMANAGER_H

// Standard includes
#include <cstdint>
#include <map>
#include <set>
#include <vector>

// Forward declarations
namespace llvm {
class BasicBlock;
class Function;
class Instruction;
class LLVMContext;
class Module;
class SwitchInst;
class StoreInst;
class Value;
}

class JumpTargetManager;

template<typename Map> typename Map::const_iterator
  containing(Map const& m, typename Map::key_type const& k) {
  typename Map::const_iterator it = m.upper_bound(k);
  if(it != m.begin()) {
    return --it;
  }
  return m.end();
}

template<typename Map> typename Map::iterator
  containing(Map & m, typename Map::key_type const& k) {
  typename Map::iterator it = m.upper_bound(k);
  if(it != m.begin()) {
    return --it;
  }
  return m.end();
}


/// \brief Transform constant writes to the PC in jumps
///
/// This pass looks for all the calls to the `ExitTB` function calls, looks for
/// the last write to the PC before them, checks if the written value is
/// statically known, and, if so, replaces it with a jump to the corresponding
/// translated code. If the write to the PC is not constant, no action is
/// performed, and the call to `ExitTB` remains there for later handling.
class TranslateDirectBranchesPass : public llvm::FunctionPass {
public:
  static char ID;

  TranslateDirectBranchesPass() : llvm::FunctionPass(ID),
    JTM(nullptr) { }

  TranslateDirectBranchesPass(JumpTargetManager *JTM) :
    FunctionPass(ID),
    JTM(JTM) { }

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override;

  bool runOnFunction(llvm::Function &F) override;

private:
  /// Obtains the absolute address of the PC corresponding to the original
  /// assembly instruction coming after the specified LLVM instruction
  uint64_t getNextPC(llvm::Instruction *TheInstruction);

private:
  JumpTargetManager *JTM;
};

class JumpTargetManager {
private:
  /// \brief Data structure to collect statistics about an input basic block
  struct BBSummary {
    BBSummary(uint32_t Size) : Size(Size) { }

    /// Size in bytes of the basic block
    unsigned Size;
    /// Associative map keeping track of how many times a certain register has
    /// been read
    std::map<llvm::GlobalVariable *, unsigned> ReadState;
    /// Associative map keeping track of how many times a certain register has
    /// been written
    std::map<llvm::GlobalVariable *, unsigned> WrittenState;
    /// Associative map keeping track of how many times a certain function has
    /// been called in the associated basic block. This is particularly useful
    /// to count calls to QEMU helper functions and to count the amount of
    /// instructions (in the form of calls to `newpc`)
    std::map<llvm::Function *, unsigned> CalledFunctions;
    /// Associative map keeping track of how many times a certain LLVM
    /// instruction is used in the code generated translating the input basic
    /// block
    std::map<const char *, unsigned> Opcode;
  };

public:
  using BlockWithAddress = std::pair<uint64_t, llvm::BasicBlock *>;
  static const BlockWithAddress NoMoreTargets;

public:
  using RangesVector = std::vector<std::pair<uint64_t, uint64_t>>;

  /// \param TheFunction the translated function.
  /// \param PCReg the global variable representing the program counter.
  /// \param SourceArchitecture the input architecture.
  /// \param Segments a vector of SegmentInfo representing the program.
  /// \param EnableOSRA whether OSRA is enabled or not.
  JumpTargetManager(llvm::Function *TheFunction,
                    llvm::Value *PCReg,
                    Architecture& SourceArchitecture,
                    std::vector<SegmentInfo>& Segments,
                    bool EnableOSRA);

  /// \brief Collect jump targets from the program's segments
  void harvestGlobalData();

  /// Handle a new program counter. We might already have a basic block for that
  /// program counter, or we could even have a translation for it. Return one
  /// of these, if appropriate.
  ///
  /// \param PC the new program counter.
  /// \param ShouldContinue an out parameter indicating whether the returned
  ///        basic block was just a placeholder or actually contains a
  ///        translation.
  ///
  /// \return the basic block to use from now on, or `nullptr` if the program
  ///         counter is not associated to a basic block.
  // TODO: return pair
  llvm::BasicBlock *newPC(uint64_t PC, bool& ShouldContinue);

  /// \brief Save the PC-Instruction association for future use
  void registerInstruction(uint64_t PC, llvm::Instruction *Instruction);

  /// \brief Save the PC-BasicBlock association for future use
  void registerBlock(uint64_t PC, llvm::BasicBlock *Block);

  /// \brief Translate the non-constant jumps into jumps to the dispatcher
  void translateIndirectJumps();

  /// \brief Collect staticists about all the translated basic blocks
  ///
  /// Create a CSV containing all the information in BBSummary for all the
  /// translated basic blocks.
  ///
  /// \param OutputPath path where the output CSV file should be stored.
  void collectBBSummary(std::string OutputPath);

  /// \brief Return the most recent instruction writing the program counter
  ///
  /// Note that the search is performed only in the current basic block.  The
  /// function will assert if the write instruction is not found.
  ///
  /// \param TheInstruction instruction from which start the search.
  ///
  /// \return a pointer to the last `StoreInst` writing the program counter, or
  ///         `nullptr` if a call to an helper has been found before the write
  ///         to the PC.
  llvm::StoreInst *getPrevPCWrite(llvm::Instruction *TheInstruction);

  /// \brief Return a pointer to the `exitTB` function
  ///
  /// `exitTB` is called when jump to the current value of the PC must be
  /// performed.
  llvm::Function *exitTB() { return ExitTB; }

  bool isOSRAEnabled() { return EnableOSRA; }

  /// \brief Pop from the list of program counters to explore
  ///
  /// \return a pair containing the PC and the initial block to use, or
  ///         JumpTarget::NoMoreTargets if we're done.
  BlockWithAddress peek();

  /// \brief Return true if no unexplored jump targets are available
  bool empty() { return Unexplored.empty(); }

  /// \brief Rfeturns true if the whole [\p Start,\p End) range is in an executable
  ///        segment
  bool isExecutableRange(uint64_t Start, uint64_t End) const {
    for (std::pair<uint64_t, uint64_t> Range : ExecutableRanges)
      if (Range.first <= Start && Start < Range.second
          && Range.first <= End && End < Range.second)
        return true;
    return false;
  }

  /// \brief Returns true if the given PC respects the input architecture's
  ///        instruction alignment constraints
  bool isInstructionAligned(uint64_t PC) const {
    return PC % SourceArchitecture.instructionAlignment() == 0;
  }

  /// \brief Returns if the given PC is a good candidate for exploration
  ///
  /// \return true if the PC is properly aligned, in an executable segment and
  ///         not explored yet.
  bool isInterestingPC(uint64_t PC) const {
    return isExecutableAddress(PC)
      && isInstructionAligned(PC)
      && !JumpTargets.count(PC);
  }

  /// \brief Return true if \p PC is in an executable segment
  bool isExecutableAddress(uint64_t PC) const {
    for (std::pair<uint64_t, uint64_t> Range : ExecutableRanges)
      if (Range.first <= PC && PC < Range.second)
        return true;
    return false;
  }

  bool isJumpTarget(uint64_t PC) {
    return JumpTargets.count(PC);
  }

  /// \brief Return true if the given PC is "reliable"
  ///
  /// A PC is "reliable" if it's a reliable jump target or is contained in a
  /// basic block start by a reliable jump target.
  /// A jump target is reliable if it was obtained from an explicit write to the
  /// PC and it wasn't a fallthroug jump.
  bool isReliablePC(uint64_t PC) {
    // Get the PC of the basic block "not less than" the PC
    auto It = JumpTargets.lower_bound(PC);

    uint64_t BBPC = 0;
    if (It == JumpTargets.end()) {
      BBPC = JumpTargets.rbegin()->first;
      assert(BBPC < PC);
    } else {

      BBPC = It->first;

      // If it's not the PC itself, it's the PC of the next basic
      // block, so go back one position
      if (BBPC != PC) {
        assert(It != JumpTargets.begin());
        BBPC = (--It)->first;
      }
    }

    return ReliablePCs.count(BBPC);
  }

  /// \brief Get or create a block for the given PC
  ///
  /// This function can return `nullptr`.
  ///
  /// \param PC the PC for which a `BasicBlock` is requested.
  /// \param Reliable whether \p PC was obtained in a "reliable" way or not.
  ///
  /// \return a `BasicBlock`, it might be newly created and empty, empty and
  ///         created in the past or even a `BasicBlock` already containing the
  ///         translated code.  It might also return `nullptr` if the PC is not
  ///         valid or another error occurred.
  llvm::BasicBlock *getBlockAt(uint64_t PC, bool Reliable);

  /// \brief Removes a `BasicBlock` from the SET's visited list
  void unvisit(llvm::BasicBlock *BB);

  /// \brief Return a pointer to the dispatcher basic block.
  llvm::BasicBlock *dispatcher() const { return Dispatcher; }

  bool isPCReg(llvm::Value *TheValue) const { return TheValue == PCReg; }

  llvm::Value *pcReg() const { return PCReg; }

  /// \brief Get the PC associated to \p TheInstruction and the next one
  ///
  /// \return a pair containing the PC associated to \p TheInstruction and the
  ///         next one.
  std::pair<uint64_t, uint64_t> getPC(llvm::Instruction *TheInstruction) const;

  uint64_t getNextPC(llvm::Instruction *TheInstruction) const {
    auto Pair = getPC(TheInstruction);
    return Pair.first + Pair.second;
  }

  /// \brief Read an integer number from a segment
  ///
  /// \param Address the address from which to read.
  /// \param Size the size of the read in bytes.
  ///
  /// \return a `ConstantInt` with the read value or `nullptr` in case it wasn't
  ///         possible to read the value (e.g., \p Address is not inside any of
  ///         the segments).
  llvm::ConstantInt *readConstantInt(llvm::Constant *Address, unsigned Size);

  /// \brief Reads a pointer-sized value from a segment
  /// \see readConstantInt
  llvm::Constant *readConstantPointer(llvm::Constant *Address,
				      llvm::Type *PointerTy);

  /// \brief Register a new basic block in terms of the input architecture
  ///
  /// \param Address virtual address where the basic block starts.
  /// \param Size size, in bytes, of the given basic block.
  void registerOriginalBB(uint64_t Address, uint32_t Size) {
    // TODO: this part is useful in case of erroneus situations where a basic
    //       block includes another one, a more clean approach is probably to
    //       drop all those included and then split again where they were.
    auto StartIt = OriginalBBStats.lower_bound(Address);
    if (StartIt->first == Address && StartIt->second.Size == Size)
      return;

    auto NextIt = StartIt;
    while (NextIt != OriginalBBStats.end()
           && NextIt->first < Address + Size
           && NextIt->first + NextIt->second.Size < Address + Size) {
      NextIt++;
    }

    if (StartIt != NextIt)
      OriginalBBStats.erase(StartIt, NextIt);

    auto ItStart = containingOriginalBB(Address);
    bool StartMatches = ItStart != OriginalBBStats.end();

    if (Size == 0) {
      assert(StartMatches);
      uint32_t NewSize = ItStart->second.Size - (Address - ItStart->first);
      OriginalBBStats.insert({ Address, BBSummary(NewSize) });
      ItStart->second.Size = ItStart->first - Address;
      return;
    }

    auto ItEnd = containingOriginalBB(Address + Size);
    bool EndMatches = ItEnd != OriginalBBStats.end();

    if (!StartMatches && !EndMatches) {
      OriginalBBStats.insert({ Address, BBSummary(Size) });
    } else if (StartMatches && !EndMatches) {
      ItStart->second.Size = ItStart->first - Address;
      OriginalBBStats.insert({ Address, BBSummary(Size) });
    } else if (!StartMatches && EndMatches) {
      OriginalBBStats.insert({ Address, BBSummary(ItEnd->first - Address) });
    } else if (StartMatches && EndMatches) {
      // 100% match
      if (Address == ItStart->first && Size == ItStart->second.Size)
        return;

      // Reduce the previous basic block
      ItStart->second.Size = ItStart->first - Address;
      assert(ItStart->second.Size != 0);

      // Set the size of the new basic block
      if (ItEnd == ItStart) {
        if (!(Address + Size == ItEnd->first + ItEnd->second.Size)) {
          // We're in a mistranslation situation, just ignore the error
          // TODO: emit a warning
          return;
        }
      } else {
        Size = ItEnd->first - Address;
      }

      // Create the new basic block
      OriginalBBStats.insert({ Address, BBSummary(Size) });

    } else {
      llvm_unreachable("Unexpected situation");
    }
  }

  /// \brief Increment the counter of emitted branches since the last reset
  void newBranch() { NewBranches++; }

private:
  /// \brief Return an iterator to the entry containing the given address range
  typename std::map<uint64_t, BBSummary>::iterator
  containingOriginalBB(uint64_t Address) {
    // Get the less or equal entry
    auto It = containing(OriginalBBStats, Address);

    // Check if it's within the upper bound
    if (It == OriginalBBStats.end()
        || !(Address < It->first + It->second.Size))
      return OriginalBBStats.end();

    return It;
  }

  // TODO: instead of a gigantic switch case we could map the original memory
  //       area and write the address of the translated basic block at the jump
  //       target
  void createDispatcher(llvm::Function *OutputFunction,
                        llvm::Value *SwitchOnPtr,
                        bool JumpDirectly);

  template<typename value_type, unsigned endian>
  void findCodePointers(const unsigned char *Start, const unsigned char *End);

  void harvest();

  void handleSumJump(llvm::Instruction *SumJump);

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
  std::set<llvm::BasicBlock *> Visited;

  std::vector<SegmentInfo>& Segments;
  Architecture& SourceArchitecture;

  std::set<uint64_t> ReliablePCs;
  bool EnableOSRA;

  std::map<uint64_t, BBSummary> OriginalBBStats;
  unsigned NewBranches = 0;
};

#endif // _JUMPTARGETMANAGER_H
