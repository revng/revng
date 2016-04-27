#ifndef _INSTRUCTIONTRANSLATOR_H
#define _INSTRUCTIONTRANSLATOR_H

// Standard includes
#include <cstdint>
#include <map>
#include <vector>

// LLVM includes
#include "llvm/IR/IRBuilder.h"
#include "llvm/Pass.h"
#include "llvm/Support/ErrorOr.h"

// Local includes
#include "revamb.h"
#include "ptcdump.h"
#include "jumptargetmanager.h"

// Forward declarations
namespace llvm {
class BasicBlock;
class CallInst;
class Function;
class MDNode;
class Module;
}

class JumpTargetManager;
class VariableManager;

/// \brief Expands a PTC instruction to LLVM IR
class InstructionTranslator {
public:
  using LabeledBlocksMap = std::map<std::string, llvm::BasicBlock *>;

  /// \param Builder the IRBuilder to be used to create the translated
  ///                code.
  /// \param Variables reference to the VariableManager.
  /// \param JumpTargets reference to the JumpTargetManager.
  /// \param Blocks reference to a `vector` of `BasicBlock`s used to keep track
  ///        on which `BasicBlock`s the InstructionTranslator worked on, for
  ///        further processing.
  /// \param SourceArchitecture the input architecture.
  /// \param TargetArchitecture the output architecture.
  InstructionTranslator(llvm::IRBuilder<>& Builder,
                        VariableManager& Variables,
                        JumpTargetManager& JumpTargets,
                        std::vector<llvm::BasicBlock *> Blocks,
                        Architecture& SourceArchitecture,
                        Architecture& TargetArchitecture);

  /// \brief Handle a new instruction from the input code
  ///
  /// \param Instr the newly met PTCInstruction;
  /// \param Next the PTCInstruction immediately following \p Instr,
  ///             or `nullptr` if \p Instr is the last instruction
  ///             translated by libtinycode.
  /// \param EndPC PC of the instruction coming after the next
  ///              instruction after the last translated by
  ///              libtinycode.
  /// \param IsFirst true, if \p Instr is the first instruction
  ///                translated by libtinycode.
  /// \param ForceNew true if a new jump target (and therefore basic
  ///                 block should be created from the PC associate to
  ///                 \p Instr.
  ///
  /// \return a tuple with 4 entries: a `bool` specifying whether the
  ///         translation should stop or not, an `MDNode` containing the
  ///         disassembled instruction and the value of the PC and two
  ///         `uint64_t` representing the current and next PC.
  // TODO: rename to newPC
  // TODO: the signature of this function is ugly
  std::tuple<bool,
    llvm::MDNode *,
    uint64_t,
    uint64_t> newInstruction(PTCInstruction *Instr,
                             PTCInstruction *Next,
                             uint64_t EndPC,
                             bool IsFirst,
                             bool ForceNew);

  /// \brief Translate an ordinary instruction
  ///
  /// \param Instr the instruction to translate.
  /// \param PC the PC associated to \p Instr.
  /// \param NextPC the PC associated to instruction after \p Instr.
  ///
  /// \return true if the translation must be stopped (i.e., an error has been
  ///         met during translation).
  bool translate(PTCInstruction *Instr, uint64_t PC, uint64_t NextPC);

  /// \brief Translate a call to an helper
  ///
  /// \param Instr the PTCInstruction of the call to the helper.
  ///
  /// \return true if this call to the helper requires the creation of
  ///         a jump to the dispatcher and a new jump target for the
  ///         PC coming after the call, or, in other terms, if the
  ///         helper can change the program counter.
  bool translateCall(PTCInstruction *Instr);

  /// \brief Handle calls to `newPC` marker and emit coverage information
  ///
  /// This function can either remove calls to `newPC` markers or finalized them
  /// with updated information for run-time tracing purposes.
  ///
  /// \param CoveragePath path where the coverage information should be stored.
  /// \param EnableTracing whether calls to an external `newPC` function should
  ///        be removed or not.
  void finalizeNewPCMarkers(std::string &CoveragePath, bool EnableTracing);

  /// \brief Notifies InstructionTranslator about a new PTC translation
  void reset() { LabeledBasicBlocks.clear(); }

private:
  llvm::ErrorOr<std::vector<llvm::Value *>>
  translateOpcode(PTCOpcode Opcode,
                  std::vector<uint64_t> ConstArguments,
                  std::vector<llvm::Value *> InArguments);
private:
  llvm::IRBuilder<>& Builder;
  VariableManager& Variables;
  JumpTargetManager& JumpTargets;
  std::map<std::string, llvm::BasicBlock *> LabeledBasicBlocks;
  std::vector<llvm::BasicBlock *> Blocks;
  llvm::Module& TheModule;

  llvm::Function *TheFunction;

  Architecture& SourceArchitecture;
  Architecture& TargetArchitecture;

  llvm::Function *NewPCMarker;
};

#endif // _INSTRUCTIONTRANSLATOR_H
