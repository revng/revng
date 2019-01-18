#ifndef SET_H
#define SET_H

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <set>
#include <vector>

// LLVM includes
#include "llvm/Pass.h"

// Forward declarations
namespace llvm {
class BasicBlock;
class AnalysisUsage;
class Function;
class LoadInst;
class Value;
} // namespace llvm

class JumpTargetManager;

class SETPass : public llvm::ModulePass {
public:
  /// \brief Information about the possible destination of a jump instruction
  struct JumpInfo {
    JumpInfo(llvm::StoreInst *Instruction,
             bool Approximate,
             std::vector<uint64_t> Destinations) :
      Instruction(Instruction),
      Approximate(Approximate),
      Destinations(Destinations) {}

    llvm::StoreInst *Instruction; ///< The jump instruction
    bool Approximate; ///< Is the destination list approximate or exhaustive?
    std::vector<uint64_t> Destinations; ///< Possible target PCs
  };

public:
  static char ID;

  SETPass() :
    llvm::ModulePass(ID),
    JTM(nullptr),
    Visited(nullptr),
    UseOSRA(false) {}

  SETPass(JumpTargetManager *JTM,
          bool UseOSRA,
          std::set<llvm::BasicBlock *> *Visited) :
    llvm::ModulePass(ID),
    JTM(JTM),
    Visited(Visited),
    UseOSRA(UseOSRA) {}

  bool runOnModule(llvm::Module &M) override;

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override;

  const std::vector<JumpInfo> &jumps() const { return Jumps; }

  virtual void releaseMemory() override {
    revng_log(ReleaseLog, "SETPass is releasing memory");
    freeContainer(Jumps);
  }

private:
  JumpTargetManager *JTM;
  std::set<llvm::BasicBlock *> *Visited;
  bool UseOSRA;
  std::vector<JumpInfo> Jumps;
};

#endif // SET_H
