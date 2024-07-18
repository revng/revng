#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <cstdint>
#include <map>
#include <string>

#include "llvm/IR/IRBuilder.h"
#include "llvm/Pass.h"

#include "revng/Support/CommandLine.h"
#include "revng/Support/IRHelpers.h"

#include "CPUStateAccessAnalysisPass.h"

#include "qemu/libtcg/libtcg.h"

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

class VariableManager;

// TODO: rename
extern llvm::cl::opt<bool> External;

/// Maintain the list of variables required by PTC
///
/// It can be queried for a variable, which, if not already existing, will be
/// created on the fly.
class VariableManager {
public:
  VariableManager(llvm::Module &M,
                  bool TargetIsLittleEndian,
                  llvm::StructType *CPUStruct,
                  unsigned LibTcgEnvOffset,
                  uint8_t *LibTcgEnvPtr);

  void setAllocaInsertPoint(llvm::Instruction *I) {
    AllocaBuilder.SetInsertPoint(I);
  }

  llvm::Value *load(llvm::IRBuilder<> &Builder, LibTcgArgument *Arg) {
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
  /// \param Name an optional name to force for the associate global variable.
  ///
  /// \return a pair composed by the request global variable and the offset in
  ///         it corresponding to \p Offset. For instance, if you're accessing
  ///         the third byte of a 32-bit integer it will 2.
  std::pair<llvm::GlobalVariable *, unsigned>
  getByEnvOffset(intptr_t Offset, std::string Name = "") {
    return getByCPUStateOffsetInternal(LibTcgEnvOffset + Offset, Name);
  }

  /// Notify VariableManager to reset all the Translation Block (TB) specific
  /// information
  ///
  /// Note: A TB refers to a set of instructions that could be translated by
  ///       QEMU in one shot, and might encompass multiple LLVM basic blocks.
  ///
  /// \param Instructions list of instructions returned by qemu for this TB
  void newTranslationBlock(LibTcgInstructionList *Instructions);

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

  llvm::Value *loadFromEnvOffset(llvm::IRBuilder<> &Builder,
                                 unsigned LoadSize,
                                 unsigned Offset) {
    return loadFromCPUStateOffset(Builder, LoadSize, LibTcgEnvOffset + Offset);
  }

  std::optional<llvm::StoreInst *> storeToEnvOffset(llvm::IRBuilder<> &Builder,
                                                    unsigned StoreSize,
                                                    unsigned Offset,
                                                    llvm::Value *ToStore) {
    unsigned ActualOffset = LibTcgEnvOffset + Offset;
    return storeToCPUStateOffset(Builder, StoreSize, ActualOffset, ToStore);
  }

  bool memcpyAtEnvOffset(llvm::IRBuilder<> &Builder,
                         llvm::CallInst *CallMemcpy,
                         unsigned Offset,
                         bool EnvIsSrc);

  /// Perform finalization steps on variables
  void finalize();

  void rebuildCSVList();

  /// Gets the CPUStateType
  llvm::StructType *getCPUStateType() const { return CPUStateType; }

  bool hasEnv() const { return Env != nullptr; }

  llvm::Value *cpuStateToEnv(llvm::Value *CPUState,
                             llvm::Instruction *InsertBefore) const;

private:
  std::pair<bool, llvm::Value *>
  getOrCreate(LibTcgArgument *Arg, bool Reading);

  llvm::Value *loadFromCPUStateOffset(llvm::IRBuilder<> &Builder,
                                      unsigned LoadSize,
                                      unsigned Offset);

  std::optional<llvm::StoreInst *>
  storeToCPUStateOffset(llvm::IRBuilder<> &Builder,
                        unsigned StoreSize,
                        unsigned Offset,
                        llvm::Value *ToStore);

  llvm::GlobalVariable *getByCPUStateOffset(intptr_t Offset,
                                            std::string Name = "");

  std::pair<llvm::GlobalVariable *, unsigned>
  getByCPUStateOffsetInternal(intptr_t Offset, std::string Name = "");

private:
  llvm::Module &TheModule;
  llvm::IRBuilder<> AllocaBuilder;
  using TemporariesMap = std::map<LibTcgTemp *, llvm::AllocaInst *>;
  using GlobalsMap = std::map<intptr_t, llvm::GlobalVariable *>;
  GlobalsMap CPUStateGlobals;
  GlobalsMap OtherGlobals;
  // QEMU terminology
  // - Translation Block (TB): All instructions that could be translated in one
  //   shot, might encompass multiple LLVM basic blocks.
  // - Extended Basic Block (EBB): Single entry, multiple exit region that falls
  //   through condtitional branches, smaller than a TB. 
  TemporariesMap TBTemporaries;
  TemporariesMap EBBTemporaries;
  LibTcgInstructionList *Instructions;

  llvm::StructType *CPUStateType;
  const llvm::DataLayout *ModuleLayout;
  unsigned LibTcgEnvOffset;
  uint8_t *LibTcgEnvPtr;

  llvm::GlobalVariable *Env;
  bool TargetIsLittleEndian;
};
