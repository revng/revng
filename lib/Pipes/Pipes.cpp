/// \file Pipes.cpp
/// \brief Pipes contains all the various pipes and kinds exposed by revng

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/InitializePasses.h"
#include "llvm/PassRegistry.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/TargetSelect.h"

#include "revng/Pipeline/LLVMGlobalKindBase.h"
#include "revng/Pipeline/Registry.h"
#include "revng/Pipeline/Target.h"
#include "revng/Pipes/CompileModulePipe.h"
#include "revng/Pipes/FileContainer.h"
#include "revng/Pipes/LiftPipe.h"
#include "revng/Pipes/LinkForTranslationPipe.h"
#include "revng/Pipes/LinkSupportPipe.h"
#include "revng/Pipes/RootKind.h"
#include "revng/Support/Statistics.h"

using namespace std;
using namespace pipeline;
using namespace revng::pipes;

namespace revng::pipes {

static RegisterLLVMPass<O2Pipe> P2;

static RegisterContainerFactory F1("Binary", makeFileContainerFactory(Binary));
static RegisterContainerFactory
  F2("Object", makeFileContainerFactory(Object, ".o"));
static RegisterContainerFactory
  F3("Translated", makeFileContainerFactory(Translated));

class LLVMPipelineRegistry : public Registry {

public:
  void registerContainersAndPipes(Loader &Loader) override {
    auto &Ctx = Loader.getContext();
    auto MaybeLLVMContext = Ctx.getGlobal<LLVMContextWrapper>("LLVMContext");

    if (!MaybeLLVMContext)
      return;

    auto &PipeContext = Loader.getContext();
    auto &LLVMContext = (*MaybeLLVMContext)->getContext();
    auto Factory = makeDefaultLLVMContainerFactory(PipeContext, LLVMContext);

    Loader.addContainerFactory("LLVMContainer", std::move(Factory));
  }

  void registerKinds(KindsRegistry &KindDictionary) override {}

  void libraryInitialization() override {

    llvm::codegen::RegisterCodeGenFlags Reg;
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::InitializeNativeTargetAsmParser();
    auto &Registry = *llvm::PassRegistry::getPassRegistry();
    llvm::initializeIPO(Registry);
    installStatistics();
  }

  ~LLVMPipelineRegistry() override = default;
};

static LLVMPipelineRegistry Registry;

} // namespace revng::pipes
