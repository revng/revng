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

  // TODO: rename to newPC
  // TODO: the signature of this funciton is ugly
  std::tuple<bool,
    llvm::MDNode *,
    uint64_t,
    uint64_t> newInstruction(PTCInstruction *Instr,
                             PTCInstruction *Next,
                             uint64_t EndPC,
                             bool IsFirst,
                             bool ForceNew);
  bool translate(PTCInstruction *Instr, uint64_t PC, uint64_t NextPC);
  bool translateCall(PTCInstruction *Instr);

  void removeNewPCMarkers(std::string &CoveragePath);

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
};

#endif // _INSTRUCTIONTRANSLATOR_H
