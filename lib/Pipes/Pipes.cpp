/// \file Pipes.cpp
/// Contains the definition of the pipe registry.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <array>
#include <memory>
#include <string>

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/PassRegistry.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/TargetSelect.h"

#include "revng/Model/LoadModelPass.h"
#include "revng/Pipeline/AllRegistries.h"
#include "revng/Pipeline/Context.h"
#include "revng/Pipeline/LLVMKind.h"
#include "revng/Pipeline/Target.h"
#include "revng/Pipes/FileContainer.h"
#include "revng/Pipes/ModelGlobal.h"
#include "revng/Pipes/RootKind.h"
#include "revng/Recompile/CompileModulePipe.h"
#include "revng/Support/Statistics.h"

using namespace std;
using namespace pipeline;
using namespace ::revng::pipes;
using namespace ::revng::kinds;

namespace revng::pipes {

static RegisterLLVMPass<O2Pipe> P2;

static RegisterDefaultConstructibleContainer<BinaryFileContainer> F1;
static RegisterDefaultConstructibleContainer<ObjectFileContainer> F2;
static RegisterDefaultConstructibleContainer<TranslatedFileContainer> F4;
static RegisterDefaultConstructibleContainer<HexDumpFileContainer> F5;

class LLVMPipelineRegistry : public Registry {

public:
  void registerContainersAndPipes(Loader &Loader) override {
    using namespace llvm;
    auto &Ctx = Loader.getContext();
    auto MaybeLLVMContext = Ctx.getExternalContext<LLVMContext>("LLVMContext");

    if (!MaybeLLVMContext)
      return;

    auto &PipeContext = Loader.getContext();
    auto &LLVMContext = **MaybeLLVMContext;
    auto Factory = ContainerFactory::fromGlobal<LLVMContainer>(&PipeContext,
                                                               &LLVMContext);

    Loader.addContainerFactory("LLVMContainer", std::move(Factory));
  }

  void registerKinds(KindsRegistry &KindDictionary) override {}

  void libraryInitialization() override {

    llvm::codegen::RegisterCodeGenFlags Reg;
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::InitializeNativeTargetAsmParser();

    auto &Registry = *llvm::PassRegistry::getPassRegistry();
    llvm::initializeCore(Registry);
    llvm::initializeTransformUtils(Registry);
    llvm::initializeScalarOpts(Registry);
    llvm::initializeVectorization(Registry);
    llvm::initializeInstCombine(Registry);
    llvm::initializeIPO(Registry);
    llvm::initializeAnalysis(Registry);
    llvm::initializeCodeGen(Registry);
    llvm::initializeGlobalISel(Registry);
    llvm::initializeTarget(Registry);
  }

  ~LLVMPipelineRegistry() override = default;
};

static LLVMPipelineRegistry Registry;

} // namespace revng::pipes
