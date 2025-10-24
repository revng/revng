/// \file CompileModule.cpp
/// The compile module pipe transforms an llvm module into an object file.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <memory>
#include <optional>

#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/IR/AutoUpgrade.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Linker/Linker.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/IPO.h"

#include "revng/Lift/IRAnnotators.h"
#include "revng/Model/FunctionTags.h"
#include "revng/Model/Register.h"
#include "revng/Pipeline/AllRegistries.h"
#include "revng/Pipeline/LLVMContainer.h"
#include "revng/Pipeline/Target.h"
#include "revng/Pipes/Kinds.h"
#include "revng/Pipes/ModelGlobal.h"
#include "revng/Recompile/CompileModulePipe.h"
#include "revng/Recompile/OriginalAssemblyAnnotationWriter.h"
#include "revng/Support/Assert.h"
#include "revng/Support/IRHelpers.h"

using namespace llvm;
using namespace llvm::codegen;
using namespace std;
using namespace pipeline;
using namespace ::revng::pipes;
using namespace sys;
using namespace cl;

static cl::opt<char> OptLevel("compile-opt-level",
                              cl::desc("Optimization level. [-O0, -O1, -O2, or "
                                       "-O3] (default = '-O2')"),
                              cl::Prefix,
                              cl::ZeroOrMore,
                              cl::init(' '));

// TODO: should we use this in every LLVMContext?
class CustomDiagnosticHandler : public llvm::DiagnosticHandler {
public:
  bool handleDiagnostics(const llvm::DiagnosticInfo &DI) override {
    // Get diagnostic message
    std::string Message;
    {
      raw_string_ostream Stream(Message);
      DiagnosticPrinterRawOStream DP(Stream);
      DI.print(DP);
    }

    // Handle based on severity
    switch (DI.getSeverity()) {
    case llvm::DS_Error:
      revng_abort(Message.c_str());
      break;
    case llvm::DS_Warning:
    case llvm::DS_Remark:
    case llvm::DS_Note:
      // TODO: dump to a logger
      break;
    }

    // Return true to indicate we've handled the diagnostic
    return true;
  }
};

static void compileModuleRunImpl(const Context &Context,
                                 LLVMContainer &Module,
                                 ObjectFileContainer &TargetBinary) {
  using namespace revng;

  auto Enumeration = Module.enumerate();
  if (not Enumeration.contains(pipeline::Target(kinds::Root))
      and not Enumeration.contains(pipeline::Target(kinds::IsolatedRoot)))
    return;

  if (Enumeration.contains(pipeline::Target(kinds::IsolatedRoot))
      and not Enumeration.contains(kinds::Isolated.allTargets(Context)))
    return;

  StringMap<llvm::cl::Option *> &RegOptions(getRegisteredOptions());
  getOption<bool>(RegOptions, "disable-machine-licm")->setInitialValue(true);

  llvm::Module *M = &Module.getModule();

  M->getContext()
    .setDiagnosticHandler(std::make_unique<CustomDiagnosticHandler>());

  {
    auto Architecture = getModelFromContext(Context)->Architecture();
    auto ArchName = model::Architecture::getQEMUName(Architecture).str();

    // Note: here we use the full version of the helpers, i.e., where we all the
    //       definitions (as opposed to only those with revng_inline, as it
    //       happens with the slim version).
    const std::string LibHelpersName = "/share/revng/libtcg-helpers-annotated-"
                                       + ArchName + ".bc";
    auto OptionalHelpers = ResourceFinder.findFile(LibHelpersName);
    revng_assert(OptionalHelpers.has_value(), "Cannot find tinycode helpers");

    auto HelpersModule = parseIR(M->getContext(), OptionalHelpers.value());

    linkModules(std::move(HelpersModule), *M, GlobalValue::InternalLinkage);

    M->getFunction("main")->setLinkage(llvm::GlobalValue::ExternalLinkage);
  }

  for (Function &F : *M)
    F.setSection("");

  OriginalAssemblyAnnotationWriter OAAW(M->getContext());
  createSelfReferencingDebugInfo(M, Module.name(), &OAAW);

  // Get the target specific parser.
  std::string Error;
  Triple TheTriple(M->getTargetTriple());
  const auto *TheTarget = TargetRegistry::lookupTarget("", TheTriple, Error);
  revng_assert(TheTarget, Error.c_str());

  TargetOptions Options = InitTargetOptionsFromCodeGenFlags(TheTriple);

  CodeGenOpt::Level OLvl = CodeGenOpt::Default;
  switch (OptLevel) {
  case ' ':
    break;
  case '0':
    OLvl = CodeGenOpt::None;
    break;
  case '1':
    OLvl = CodeGenOpt::Less;
    break;
  case '2':
    OLvl = CodeGenOpt::Default;
    break;
  case '3':
    OLvl = CodeGenOpt::Aggressive;
    break;
  default:
    revng_abort("Wrong Optimization Level");
    return;
  }

  auto Ptr = TheTarget->createTargetMachine(TheTriple.getTriple(),
                                            "",
                                            "",
                                            Options,
                                            std::nullopt,
                                            M->getCodeModel(),
                                            OLvl);
  unique_ptr<TargetMachine> Target(Ptr);

  // Add the target data from the target machine, if it exists, or the module.
  M->setDataLayout(Target->createDataLayout());

  // This needs to be done after setting datalayout since it calls verifier
  // to check debug info whereas verifier relies on correct datalayout.
  UpgradeDebugInfo(*M);

  // Before compiling, do a last pass to collect all the globals (in particular,
  // helpers) we don't need. This saves from spurious linking errors.
  legacy::PassManager CleanupPM;
  CleanupPM.add(llvm::createGlobalDCEPass());
  CleanupPM.run(*M);

  // Create pass manager
  legacy::PassManager PM;
  // Add an appropriate TargetLibraryInfo pass for the module's triple.
  TargetLibraryInfoImpl TLII(Triple(M->getTargetTriple()));
  PM.add(new TargetLibraryInfoWrapperPass(TLII));

  std::error_code EC;
  raw_fd_ostream OutputStream(TargetBinary.getOrCreatePath(), EC);
  revng_assert(!EC);

  bool Err = Target->addPassesToEmitFile(PM,
                                         OutputStream,
                                         nullptr,
                                         CGFT_ObjectFile,
                                         true);
  revng_assert(not Err);

  revng::verify(M);
  PM.run(*M);
  revng::verify(M);

  auto Path = TargetBinary.path();

  auto Permissions = cantFail(errorOrToExpected(fs::getPermissions(*Path)));
  Permissions = Permissions | fs::owner_exe;
  fs::setPermissions(*TargetBinary.path(), Permissions);
}

void CompileModule::run(ExecutionContext &EC,
                        LLVMContainer &Module,
                        ObjectFileContainer &TargetBinary) {
  compileModuleRunImpl(EC.getContext(), Module, TargetBinary);
  EC.commitUniqueTarget(TargetBinary);
}

void CompileIsolatedModule::run(ExecutionContext &EC,
                                LLVMContainer &Module,
                                ObjectFileContainer &TargetBinary) {
  compileModuleRunImpl(EC.getContext(), Module, TargetBinary);
  EC.commitUniqueTarget(TargetBinary);
}

static RegisterPipe<CompileModule> E2;
static RegisterPipe<CompileIsolatedModule> E3;
