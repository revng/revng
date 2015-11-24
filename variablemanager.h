#ifndef _VARIABLEMANAGER_H
#define _VARIABLEMANAGER_H

// Standard includes
#include <cstdint>
#include <map>
#include <string>

// LLVM includes
#include "llvm/IR/IRBuilder.h"

// Local includes
#include "ptcdump.h"

namespace llvm {
class AllocaInst;
class BasicBlock;
class DataLayout;
class GlobalVariable;
class Module;
class StructType;
class Value;
}

/// \brief Maintains the list of variables required by PTC.
///
/// It can be queried for a variable, which, if not already existing, will be
/// created on the fly.
class VariableManager {
public:
  VariableManager(llvm::Module& TheModule,
                  llvm::StructType *CPUStateType,
                  const llvm::DataLayout *HelpersModuleLayout);

  /// Given a PTC temporary identifier, checks if it already exists in the
  /// generatd LLVM IR, and, if not, it creates it.
  ///
  /// \param TemporaryId the PTC temporary identifier.
  ///
  /// \return an Value wrapping the request global or local variable.
  // TODO: rename to getByTemporaryId
  llvm::Value *getOrCreate(unsigned int TemporaryId);

  llvm::GlobalVariable *getByCPUStateOffset(intptr_t Offset,
                                            std::string Name="");

  /// Informs the VariableManager that a new function has begun, so it can
  /// discard function- and basic block-level variables.
  ///
  /// \param Delimiter the new point where to insert allocations for local
  /// variables.
  /// \param Instructions the new PTCInstructionList to use from now on.
  void newFunction(llvm::Instruction *Delimiter=nullptr,
                   PTCInstructionList *Instructions=nullptr);

  /// Informs the VariableManager that a new basic block has begun, so it can
  /// discard basic block-level variables.
  ///
  /// \param Delimiter the new point where to insert allocations for local
  /// variables.
  /// \param Instructions the new PTCInstructionList to use from now on.
  void newBasicBlock(llvm::Instruction *Delimiter=nullptr,
                     PTCInstructionList *Instructions=nullptr);

  void newBasicBlock(llvm::BasicBlock *Delimiter,
                     PTCInstructionList *Instructions=nullptr);

  bool isEnv(llvm::Value *TheValue);

private:
  llvm::Module& TheModule;
  llvm::IRBuilder<> Builder;
  using TemporariesMap = std::map<unsigned int, llvm::AllocaInst *>;
  using GlobalsMap = std::map<intptr_t, llvm::GlobalVariable *>;
  GlobalsMap CPUStateGlobals;
  GlobalsMap OtherGlobals;
  TemporariesMap Temporaries;
  TemporariesMap LocalTemporaries;
  PTCInstructionList *Instructions;

  llvm::StructType *CPUStateType;
  const llvm::DataLayout *HelpersModuleLayout;

  llvm::Value *Env;
};

#endif // _VARIABLEMANAGER_H
