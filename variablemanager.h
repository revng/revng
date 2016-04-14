#ifndef _VARIABLEMANAGER_H
#define _VARIABLEMANAGER_H

// Standard includes
#include <cstdint>
#include <map>
#include <string>

// LLVM includes
#include "llvm/IR/IRBuilder.h"
#include "llvm/Pass.h"

// Local includes
#include "ptcdump.h"
#include "revamb.h"

namespace llvm {
class AllocaInst;
class BasicBlock;
class DataLayout;
class GlobalVariable;
class Module;
class StructType;
class Value;
}

class VariableManager;

class CorrectCPUStateUsagePass : public llvm::ModulePass {
public:
  static char ID;

  CorrectCPUStateUsagePass() :
    llvm::ModulePass(ID),
    Variables(nullptr),
    EnvOffset(0) { }

  CorrectCPUStateUsagePass(VariableManager *Variables, unsigned EnvOffset) :
    llvm::ModulePass(ID),
    Variables(Variables),
    EnvOffset(EnvOffset) { }

public:
  bool runOnModule(llvm::Module& TheModule) override;

private:
  VariableManager *Variables;
  unsigned EnvOffset;
};

/// \brief Maintains the list of variables required by PTC.
///
/// It can be queried for a variable, which, if not already existing, will be
/// created on the fly.
class VariableManager {
public:
  VariableManager(llvm::Module& TheModule,
                  llvm::Module& HelpersModule,
                  Architecture &TargetArchitecture);

  friend class CorrectCPUStateUsagePass;

  /// Given a PTC temporary identifier, checks if it already exists in the
  /// generatd LLVM IR, and, if not, it creates it.
  ///
  /// \param TemporaryId the PTC temporary identifier.
  ///
  /// \return an Value wrapping the request global or local variable.
  // TODO: rename to getByTemporaryId
  llvm::Value *getOrCreate(unsigned int TemporaryId);

  std::pair<llvm::GlobalVariable*,
            unsigned> getByEnvOffset(intptr_t Offset,
                                     std::string Name="") {
    return getByCPUStateOffsetInternal(EnvOffset + Offset, Name);
  }

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

  /// Returns true if the given variable is the env variable
  bool isEnv(llvm::Value *TheValue);

  CorrectCPUStateUsagePass *createCorrectCPUStateUsagePass() {
    return new CorrectCPUStateUsagePass(this, EnvOffset);
  }

  llvm::Value *computeEnvAddress(llvm::Type *TargetType,
                                 llvm::Instruction *InsertBefore,
                                 unsigned Offset = 0);

  void setDataLayout(const llvm::DataLayout *NewLayout) {
    ModuleLayout = NewLayout;
  }

  template<typename T>
  T *setAliasScope(T *Instruction);

  template<typename T>
  T *setNoAlias(T *Instruction);

  std::vector<llvm::Value *> locals() {
    std::vector<llvm::Value *> Locals;
    for (auto Pair : LocalTemporaries)
      Locals.push_back(Pair.second);
    return Locals;
  }

  llvm::Value *loadFromEnvOffset(llvm::IRBuilder<> &Builder,
                                 unsigned LoadSize,
                                 unsigned Offset) {
    return loadFromCPUStateOffset(Builder, LoadSize, EnvOffset + Offset);
  }

  bool storeToEnvOffset(llvm::IRBuilder<> &Builder,
                        unsigned StoreSize,
                        unsigned Offset,
                        llvm::Value *ToStore) {
    unsigned ActualOffset = EnvOffset + Offset;
    return storeToCPUStateOffset(Builder, StoreSize, ActualOffset, ToStore);
  }

private:
  llvm::Value *loadFromCPUStateOffset(llvm::IRBuilder<> &Builder,
                                      unsigned LoadSize,
                                      unsigned Offset);

  bool storeToCPUStateOffset(llvm::IRBuilder<> &Builder,
                             unsigned StoreSize,
                             unsigned Offset,
                             llvm::Value *ToStore);

  llvm::GlobalVariable *getByCPUStateOffset(intptr_t Offset,
                                            std::string Name="");
  std::pair<llvm::GlobalVariable*, unsigned>
    getByCPUStateOffsetInternal(intptr_t Offset,
                                std::string Name="");

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
  const llvm::DataLayout *ModuleLayout;
  unsigned EnvOffset;

  llvm::Value *Env;
  unsigned AliasScopeMDKindID;
  unsigned NoAliasMDKindID;
  llvm::MDNode *CPUStateScopeSet;

  Architecture &TargetArchitecture;
};

#endif // _VARIABLEMANAGER_H
