/// \file Pipes.cpp
/// \brief Pipes contains all the various pipes and kinds exposed by revng

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <array>

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

#include "revng/EarlyFunctionAnalysis/CollectFunctionsFromCalleesPass.h"
#include "revng/EarlyFunctionAnalysis/CollectFunctionsFromUnusedAddressesPass.h"
#include "revng/EarlyFunctionAnalysis/DetectABI.h"
#include "revng/Model/LoadModelPass.h"
#include "revng/Pipeline/AllRegistries.h"
#include "revng/Pipeline/Context.h"
#include "revng/Pipeline/LLVMGlobalKindBase.h"
#include "revng/Pipeline/Target.h"
#include "revng/Pipes/CompileModulePipe.h"
#include "revng/Pipes/FileContainer.h"
#include "revng/Pipes/LiftPipe.h"
#include "revng/Pipes/LinkSupportPipe.h"
#include "revng/Pipes/ModelGlobal.h"
#include "revng/Pipes/RootKind.h"
#include "revng/Support/Statistics.h"

using namespace std;
using namespace pipeline;
using namespace ::revng::pipes;

namespace revng::pipes {

static RegisterLLVMPass<O2Pipe> P2;

static RegisterContainerFactory
  F1("Binary", makeFileContainerFactory(Binary, "application/x-executable"));
static RegisterContainerFactory
  F2("Object", makeFileContainerFactory(Object, "application/x-object", ".o"));
static RegisterContainerFactory
  F3("Translated",
     makeFileContainerFactory(Translated, "application/x-executable"));

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
    llvm::initializeCore(Registry);
    llvm::initializeTransformUtils(Registry);
    llvm::initializeScalarOpts(Registry);
    llvm::initializeVectorization(Registry);
    llvm::initializeInstCombine(Registry);
    llvm::initializeAggressiveInstCombine(Registry);
    llvm::initializeIPO(Registry);
    llvm::initializeInstrumentation(Registry);
    llvm::initializeAnalysis(Registry);
    llvm::initializeCoroutines(Registry);
    llvm::initializeCodeGen(Registry);
    llvm::initializeGlobalISel(Registry);
    llvm::initializeTarget(Registry);

    installStatistics();
  }

  ~LLVMPipelineRegistry() override = default;
};

static LLVMPipelineRegistry Registry;

class DetectABIAnalysis {
public:
  static constexpr auto Name = "DetectABI";

  std::vector<std::vector<pipeline::Kind *>> AcceptedKinds = {
    { &revng::pipes::Root }
  };

  void run(const pipeline::Context &Ctx, pipeline::LLVMContainer &Container) {
    llvm::legacy::PassManager Manager;
    registerPasses(Ctx, Manager);
    Manager.run(Container.getModule());
  }

  void registerPasses(const pipeline::Context &Ctx,
                      llvm::legacy::PassManager &Manager) const {
    auto Global = llvm::cantFail(Ctx.getGlobal<ModelGlobal>(ModelGlobalName));
    Manager.add(new LoadModelWrapperPass(ModelWrapper(Global->get())));
    Manager.add(new CollectFunctionsFromCalleesWrapperPass());
    Manager.add(new efa::DetectABIPass());
    Manager.add(new CollectFunctionsFromUnusedAddressesWrapperPass());
    Manager.add(new efa::DetectABIPass());
  };
};

static pipeline::RegisterAnalysis<DetectABIAnalysis> A1;

} // namespace revng::pipes
