#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <cstdint>
#include <map>
#include <string>

#include "llvm/Pass.h"

#include "revng/Support/CommandLine.h"
#include "revng/Support/IRBuilder.h"
#include "revng/Support/IRHelpers.h"

namespace llvm {
class AllocaInst;
class BasicBlock;
class DataLayout;
class GlobalVariable;
class Module;
class StructType;
class Value;
} // namespace llvm

struct LibTcgInstructionList;
struct LibTcgInstruction;
struct LibTcgArgument;
struct LibTcgTemp;

// TODO: this class is used by CodeGenerator and by fixHelpers.
//       The latter only needs the part of this class that's related to the
//       CPU state.
//       There's an opportunity to split off this class in a class that only
//       manages the CPU state and another that handles temporaries and the
//       like. We should also likely have some RAII object that saves us from
//       having to manually clean up temporary state by calling
//       newTranslationBlock.

/// Maintain the list of variables required by PTC
///
/// It can be queried for a variable, which, if not already existing, will be
/// created on the fly.
class VariableManager {
public:
  using GlobalsMap = std::map<intptr_t, llvm::GlobalVariable *>;

private:
  llvm::Module &TheModule;
  revng::NonDebugInfoCheckingIRBuilder AllocaBuilder;
  GlobalsMap CPUStateGlobals;

  // QEMU terminology
  // - Translation Block (TB): All instructions that could be translated in one
  //   shot, might encompass multiple LLVM basic blocks.
  // - Extended Basic Block (EBB): Single entry, multiple exit region that falls
  //   through conditional branches, smaller than a TB.
  using TemporariesMap = std::map<LibTcgTemp *, llvm::AllocaInst *>;
  TemporariesMap TBTemporaries;
  TemporariesMap EBBTemporaries;

  llvm::StructType *ArchCPUStruct;
  const llvm::DataLayout *ModuleLayout;
  unsigned LibTcgEnvOffset;
  uint8_t *LibTcgEnvPtr;

  llvm::GlobalVariable *Env;
  bool TargetIsLittleEndian;
  const std::map<intptr_t, llvm::StringRef> &GlobalNames;

public:
  VariableManager(llvm::Module &M,
                  bool TargetIsLittleEndian,
                  unsigned LibTcgEnvOffset,
                  uint8_t *LibTcgEnvPtr,
                  const std::map<intptr_t, llvm::StringRef> &GlobalNames);

  void setAllocaInsertPoint(llvm::Instruction *I) {
    AllocaBuilder.SetInsertPoint(I);
  }

  llvm::Value *load(revng::IRBuilder &Builder, LibTcgArgument *Arg) {
    auto [IsNew, V] = getOrCreate(Arg, true);

    if (V == nullptr)
      return nullptr;

    if (llvm::isa<llvm::ConstantInt>(V))
      return V;

    if (IsNew) {
      auto *Undef = llvm::UndefValue::get(getVariableType(V));
      Builder.CreateStore(Undef, V);
    }

    return createLoadVariable(Builder, V);
  }

  /// Get or create the LLVM value associated to a PTC temporary
  ///
  /// Given a PTC temporary identifier, checks if it already exists in the
  /// generated LLVM IR, and, if not, it creates it.
  ///
  /// \param TemporaryId the PTC temporary identifier.
  ///
  /// \return a `Value` wrapping the requested global or local variable.
  llvm::Value *getOrCreate(LibTcgArgument *Arg) {
    return getOrCreate(Arg, false).second;
  }

  /// Return the global variable corresponding to \p Offset in the CPU state.
  ///
  /// \param Offset the offset in the CPU state (the `env` PTC variable).
  ///
  /// \return a pair composed by the request global variable and the offset in
  ///         it corresponding to \p Offset. For instance, if you're accessing
  ///         the third byte of a 32-bit integer it will 2.
  std::pair<llvm::GlobalVariable *, unsigned> getByEnvOffset(intptr_t Offset) {
    return getByCPUStateOffsetWithRemainder(LibTcgEnvOffset + Offset);
  }

  /// Notify VariableManager to reset all the Translation Block (TB) specific
  /// information
  ///
  /// Note: A TB refers to a set of instructions that could be translated by
  ///       QEMU in one shot, and might encompass multiple LLVM basic blocks.
  ///
  void newTranslationBlock() {
    TBTemporaries.clear();
    newExtendedBasicBlock();
  }

  /// Informs the VariableManager that a new Extended Basic Block (EBB) has
  /// begun. An EBB is a single entry, multiple exit region that fallst through
  /// conditional branches.
  void newExtendedBasicBlock() { EBBTemporaries.clear(); }

  /// Returns true if the given variable is the env variable
  bool isEnv(llvm::Value *TheValue);

  llvm::Value *computeEnvAddress(llvm::Type *TargetType,
                                 llvm::Instruction *InsertBefore,
                                 unsigned Offset = 0);

  void setDataLayout(const llvm::DataLayout *NewLayout) {
    ModuleLayout = NewLayout;
  }

  std::vector<llvm::AllocaInst *> getLiveVariables() {
    std::vector<llvm::AllocaInst *> LiveVariables;
    for (auto Pair : TBTemporaries)
      LiveVariables.push_back(Pair.second);
    for (auto Pair : EBBTemporaries)
      LiveVariables.push_back(Pair.second);
    return LiveVariables;
  }

  llvm::Value *loadFromEnvOffset(revng::IRBuilder &Builder,
                                 unsigned LoadSize,
                                 unsigned Offset) {
    return loadFromCPUStateOffset(Builder, LoadSize, LibTcgEnvOffset + Offset);
  }

  std::optional<llvm::StoreInst *> storeToEnvOffset(revng::IRBuilder &Builder,
                                                    unsigned StoreSize,
                                                    unsigned Offset,
                                                    llvm::Value *ToStore) {
    unsigned ActualOffset = LibTcgEnvOffset + Offset;
    return storeToCPUStateOffset(Builder, StoreSize, ActualOffset, ToStore);
  }

  /// Handle memcpy, memmove and memset
  void memOpAtEnvOffset(revng::IRBuilder &Builder,
                        llvm::CallInst *Call,
                        unsigned Offset,
                        bool EnvIsSrc);

  void memOpAtCPUStateOffset(revng::IRBuilder &Builder,
                             llvm::CallInst *Call,
                             unsigned Offset,
                             bool EnvIsSrc) {
    memOpAtEnvOffset(Builder, Call, Offset - LibTcgEnvOffset, EnvIsSrc);
  }

  /// Perform finalization steps on variables
  void finalize();

  void rebuildCSVList();

  llvm::Value *cpuStateToEnv(llvm::Value *CPUState,
                             llvm::Instruction *InsertBefore) const;

private:
  std::pair<bool, llvm::Value *> getOrCreate(LibTcgArgument *Arg, bool Reading);

public:
  llvm::Value *loadFromCPUStateOffset(revng::IRBuilder &Builder,
                                      unsigned LoadSize,
                                      unsigned Offset);

  std::optional<llvm::StoreInst *>
  storeToCPUStateOffset(revng::IRBuilder &Builder,
                        unsigned StoreSize,
                        unsigned Offset,
                        llvm::Value *ToStore);

  llvm::GlobalVariable *getByCPUStateOffset(intptr_t Offset);

  std::pair<llvm::GlobalVariable *, unsigned>
  getByCPUStateOffsetWithRemainder(intptr_t Offset);

  std::optional<std::pair<llvm::GlobalVariable *, unsigned>>
  getGlobalByCPUStateOffset(intptr_t Offset) const;
};
