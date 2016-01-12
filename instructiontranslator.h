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
    JTM(nullptr),
    NewPCMarker(nullptr) { }

  TranslateDirectBranchesPass(JumpTargetManager *JTM,
                              llvm::Function *NewPCMarker) :
    FunctionPass(ID),
    JTM(JTM),
    NewPCMarker(NewPCMarker) { }

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const;

  bool runOnFunction(llvm::Function &F) override;

private:
  /// Obtains the absolute address of the PC correspoding to the original
  /// assembly instruction coming after the specified LLVM instruction
  uint64_t getNextPC(llvm::Instruction *TheInstruction);

private:
  llvm::Value *PCReg;
  JumpTargetManager *JTM;
  llvm::Function *NewPCMarker;
};

/// \brief Translates PTC instruction in LLVM IR
class InstructionTranslator {
public:
  using LabeledBlocksMap = std::map<std::string, llvm::BasicBlock *>;
  InstructionTranslator(llvm::IRBuilder<>& Builder,
                        VariableManager& Variables,
                        JumpTargetManager& JumpTargets,
                        LabeledBlocksMap& LabeledBasicBlocks,
                        std::vector<llvm::BasicBlock *> Blocks,
                        llvm::Module& TheModule,
                        llvm::Function *TheFunction,
                        Architecture& SourceArchitecture,
                        Architecture& TargetArchitecture);

  TranslateDirectBranchesPass *createTranslateDirectBranchesPass();

  // TODO: rename to newPC
  // TODO: the signature of this funciton is ugly
  std::tuple<bool,
    llvm::MDNode *,
    uint64_t> newInstruction(PTCInstruction *Instr, bool IsFirst);
  bool translate(PTCInstruction *Instr, uint64_t PC);
  void translateCall(PTCInstruction *Instr);

  void removeNewPCMarkers();

  void closeLastInstruction(uint64_t PC);

private:
  llvm::ErrorOr<std::vector<llvm::Value *>>
  translateOpcode(PTCOpcode Opcode,
                  std::vector<uint64_t> ConstArguments,
                  std::vector<llvm::Value *> InArguments);
private:
  llvm::IRBuilder<>& Builder;
  VariableManager& Variables;
  JumpTargetManager& JumpTargets;
  std::map<std::string, llvm::BasicBlock *>& LabeledBasicBlocks;
  std::vector<llvm::BasicBlock *> Blocks;
  llvm::Module& TheModule;

  llvm::Function *TheFunction;

  Architecture& SourceArchitecture;
  Architecture& TargetArchitecture;

  llvm::Function *NewPCMarker;
  llvm::CallInst *LastMarker;
};

#endif // _INSTRUCTIONTRANSLATOR_H
