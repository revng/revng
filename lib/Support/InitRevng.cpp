//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/InitializePasses.h"
#include "llvm/PassRegistry.h"
#include "llvm/Support/DebugInfoPreservation.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/TargetSelect.h"

#include "revng/Support/Assert.h"
#include "revng/Support/Configuration.h"
#include "revng/Support/IRHelpers.h"
#include "revng/Support/InitRevng.h"
#include "revng/Support/OnQuit.h"

static std::optional<llvm::codegen::RegisterCodeGenFlags> CodegenFlags;

revng::InitRevng::InitRevng(int &Argc,
                            char **&Argv,
                            const char *Overview,
                            llvm::ArrayRef<const llvm::cl::OptionCategory *>
                              CategoriesToHide) :
  InitLLVM(Argc, Argv, true) {

  if (auto Error = revng::Configuration::reload()) {
    std::string Message = llvm::toString(std::move(Error));
    dbg << "Failed to parse configuration. " << Message << "\n";
    std::exit(EXIT_FAILURE);
  }

  revng_assert(not Initialized);
  Initialized = true;

  OnQuit->install();
  initializeLLVMLibraries();

  llvm::setBugReportMsg("PLEASE submit a bug report to "
                        "https://github.com/revng/revng and include the "
                        "crash backtrace\n");

  if (not CategoriesToHide.empty()) {
    // NOLINTNEXTLINE
    llvm::cl::HideUnrelatedOptions(CategoriesToHide);
  }

  // The EnvVar option in llvm::cl::ParseCommandLineOptions prepends the
  // contents of the EnvVar environment variable to the argv, this is
  // undesirable since we have `-load`s in the argv. Here we parse the env
  // manually (if present) and append the command-line options at the end of
  // the argv.
  llvm::BumpPtrAllocator A;
  llvm::StringSaver Saver(A);
  llvm::SmallVector<const char *, 0> Arguments(Argv, Argv + Argc);
  if (auto EnvValue = llvm::sys::Process::GetEnv("REVNG_OPTIONS")) {
    llvm::cl::TokenizeGNUCommandLine(*EnvValue, Saver, Arguments);
  }

  // NOLINTNEXTLINE
  bool Result = llvm::cl::ParseCommandLineOptions(Arguments.size(),
                                                  Arguments.data(),
                                                  Overview);

  if (not Result)
    std::exit(EXIT_FAILURE);

  // Force-enable `--enable-strict-debug-information-preservation-style` for
  // revng binaries even if it wasn't specified.
  llvm::EnableStrictDebugInformationPreservationStyle.setInitialValue(true);

  using namespace llvm;
  StringMap<cl::Option *> &Options(cl::getRegisteredOptions());

  const char *OptionName = "emit-hex-constant-literals-from";
  getOption<uint64_t>(Options, OptionName)->setInitialValue(4096);
}

revng::InitRevng::~InitRevng() {
  OnQuit->quit();
}

void revng::InitRevng::initializeLLVMLibraries() {
  CodegenFlags.emplace();
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
