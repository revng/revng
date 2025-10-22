#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <memory>

#include "llvm/Pass.h"

#include "revng/BasicAnalyses/GeneratedCodeBasicInfo.h"
#include "revng/EarlyFunctionAnalysis/CollectCFG.h"

class IsolateFunctions : public llvm::ModulePass {
public:
  static char ID;

public:
  IsolateFunctions() : llvm::ModulePass(ID) {}

  bool runOnModule(llvm::Module &M) override;

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override;
};

class IsolateFunctionsImpl;

namespace revng::pypeline::piperuns {

class Isolate {
private:
  LLVMRootContainer &Root;
  LLVMFunctionContainer &Output;
  GeneratedCodeBasicInfo GCBI;
  // unique_ptr to the implementation. This is a temporary measure until the
  // old pipeline is dropped and the body of the `Impl` class can be inlined in
  // this one.
  std::unique_ptr<IsolateFunctionsImpl> Impl;
  std::vector<std::tuple<MetaAddress, llvm::Function *>> IsolatedFunctions;

public:
  static constexpr llvm::StringRef Name = "Isolate";
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
};

} // namespace revng::pypeline::piperuns
