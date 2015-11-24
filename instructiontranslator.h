#ifndef _INSTRUCTIONTRANSLATOR_H
#define _INSTRUCTIONTRANSLATOR_H

// Standard includes
#include <cstdint>
#include <map>
#include <vector>

// LLVM includes
#include "llvm/IR/IRBuilder.h"
#include "llvm/Pass.h"

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
  uint64_t getNextPC(llvm::Instruction *TheInstruction);

private:
  llvm::Value *PCReg;
  JumpTargetManager *JTM;
  llvm::Function *NewPCMarker;
};

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

  std::pair<bool, llvm::MDNode *> newInstruction(PTCInstruction *Instr,
                                                 bool IsFirst);
  void translate(PTCInstruction *Instr);
  void translateCall(PTCInstruction *Instr);

  void removeNewPCMarkers();

  void closeLastInstruction(uint64_t PC);

 private:
  std::vector<llvm::Value *>
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
