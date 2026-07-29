#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <memory>

#include "llvm/Pass.h"

#include "revng/BasicAnalyses/GeneratedCodeBasicInfo.h"
#include "revng/EarlyFunctionAnalysis/CollectCFG.h"

namespace revng::pypeline::piperuns {

class Isolate {
private:
  const model::Binary &Binary;
  const CFGMap &CFG;
  std::unique_ptr<llvm::Module> ClonedModule;
  LLVMFunctionContainer &Output;
  std::optional<GeneratedCodeBasicInfo> GCBI;
  std::vector<std::tuple<MetaAddress, llvm::Function *>> IsolatedFunctions;

  llvm::FunctionType *IsolatedFunctionType = nullptr;
  llvm::Function *FunctionDispatcher = nullptr;

  std::map<MetaAddress, llvm::Function *> IsolatedFunctionsMap;
  std::map<llvm::StringRef, llvm::Function *> DynamicFunctionsMap;

public:
  static constexpr llvm::StringRef Name = "isolate";
  using Arguments = TypeList<
    PipeRunArgument<const CFGMap, "CFG", "Function control flow data">,
    PipeRunArgument<LLVMRootContainer,
                    "Input",
                    "Input LLVM module to be isolated",
                    // The root container is first modified in-place to be
                    // isolated, then, as part of the destructor, the individual
                    // functions are split and put in their respective module in
                    // the LLVMFunctionContainer.
                    Access::Read>,
    PipeRunArgument<LLVMFunctionContainer,
                    "Output",
                    "Output LLVM modules with isolated functions",
                    Access::Write>>;

  Isolate(const class Model &Model,
          llvm::StringRef Config,
          llvm::StringRef DynamicConfig,
          const CFGMap &CFG,
          LLVMRootContainer &Root,
          LLVMFunctionContainer &Output);
  ~Isolate();

  void runOnFunction(const model::Function &TheFunction);

public:
  llvm::Function *getLocalFunction(const MetaAddress &Entry);
  llvm::Function *getDynamicFunction(llvm::StringRef SymbolName) const;
  llvm::Function *dispatcher() const { return FunctionDispatcher; }

private:
  void splitIsolatedFunctionsToOutput();

  void handleUnexpectedPCCloned(efa::OutlinedFunction &Outlined);
  void handleAnyPCJumps(efa::OutlinedFunction &Outlined,
                        const efa::ControlFlowGraph &FM);
};

} // namespace revng::pypeline::piperuns
