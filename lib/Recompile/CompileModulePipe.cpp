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

// WIP: move to IRHelpers.cpp and merge with CodeGenerator.cpp version
#include "llvm/IRReader/IRReader.h"
static std::unique_ptr<Module> parseIR(StringRef Path, LLVMContext &Context) {
  std::unique_ptr<Module> Result;
  SMDiagnostic Errors;
  Result = parseIRFile(Path, Errors, Context);

  if (Result.get() == nullptr) {
    Errors.print("revng", dbgs());
    revng_abort();
  }

  return Result;
}

// WIP
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
      // WIP: write to logger
      break;
    }

    // Return true to indicate we've handled the diagnostic
    return true;
  }
};

// WIP: inline
/// \p FinalLinkage final linkage for all the globals. Use std::nullopt to
///    preserve the original one.
inline void link(std::unique_ptr<llvm::Module> &&Source,
                 llvm::Module &Destination,
                 std::optional<GlobalValue::LinkageTypes> FinalLinkage) {
  std::map<std::string, GlobalValue::LinkageTypes> HelperGlobals;

  auto HandleGlobals = [&HelperGlobals, &Destination](auto &&GlobalsRange) {
    using T = std::decay_t<decltype(*GlobalsRange.begin())>;
    for (T &HelperGlobal : GlobalsRange) {
      auto GlobalName = HelperGlobal.getName();

      if (not GlobalName.startswith("llvm.")) {

        // Register so we can change its linkage later
        HelperGlobals[GlobalName.str()] = HelperGlobal.getLinkage();

        GlobalObject *LocalGlobal = nullptr;
        if constexpr (std::is_same_v<T, llvm::GlobalVariable>) {
          LocalGlobal = Destination.getGlobalVariable(GlobalName);
        } else {
          static_assert(std::is_same_v<T, llvm::Function>);
          LocalGlobal = Destination.getFunction(GlobalName);
        }

        if (LocalGlobal != nullptr) {
          // We have a global with the same name
          HelperGlobal.setLinkage(llvm::GlobalValue::ExternalLinkage);

          bool AlreadyAvailable = not LocalGlobal->isDeclaration();
          if (AlreadyAvailable) {
            // Turn helper global into declaration
            if constexpr (std::is_same_v<T, llvm::GlobalVariable>) {
              HelperGlobal.setInitializer(nullptr);
            } else {
              static_assert(std::is_same_v<T, llvm::Function>);
              HelperGlobal.deleteBody();
            }
          } else {
            // Ensure it will be linked
            LocalGlobal->setLinkage(llvm::GlobalValue::ExternalLinkage);
          }
        }
      }
    }
  };

  HandleGlobals(Source->globals());
  HandleGlobals(Source->functions());

  Linker TheLinker(Destination);
  bool Failed = TheLinker.linkInModule(std::move(Source),
                                       Linker::LinkOnlyNeeded);
  revng_assert(not Failed, "Linking failed");

  for (auto [GlobalName, Linkage] : HelperGlobals) {
    if (auto *GV = Destination.getGlobalVariable(GlobalName))
      if (not GV->isDeclaration())
        GV->setLinkage(FinalLinkage.value_or(Linkage));

    if (auto *F = Destination.getFunction(GlobalName))
      if (not F->isDeclaration())
        F->setLinkage(FinalLinkage.value_or(Linkage));
  }
}

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

    // WIP
    #define Check(M) do { for (GlobalVariable &CSV : (M).globals()) if (FunctionTags::CSV.isTagOf(&CSV)) { int x; if (CSV.getName().contains(".") and not CSV.getName().substr(CSV.getName().find_last_of(".") + 1).consumeInteger(0, x)) { dbg << CSV.getName().substr(CSV.getName().find_last_of(".")).str() << "\n"; CSV.dump(); revng_abort();} } } while(0)

    auto HelpersModule = parseIR(OptionalHelpers.value(), M->getContext());

    Check(*HelpersModule);
    Check(*M);

    link(std::move(HelpersModule),
         *M,
         GlobalValue::InternalLinkage);

    Check(*M);

    M->getFunction("main")->setLinkage(llvm::GlobalValue::ExternalLinkage);
  }

  // WIP
  revng::forceVerify(M);

  OriginalAssemblyAnnotationWriter OAAW(M->getContext());
  createSelfReferencingDebugInfo(M, Module.name(), &OAAW);

  // Get the target specific parser.
  std::string Error;
  Triple TheTriple(M->getTargetTriple());
  const auto *TheTarget = TargetRegistry::lookupTarget("", TheTriple, Error);
  revng_assert(TheTarget);

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

  std::error_code EC;
  raw_fd_ostream OutputStream(TargetBinary.getOrCreatePath(), EC);
  revng_assert(!EC);

  // WIP
  {
    legacy::PassManager PM;
    PM.add(llvm::createGlobalDCEPass());
    PM.run(*M);
  }

  // Create pass manager
  legacy::PassManager PM;
  // Add an appropriate TargetLibraryInfo pass for the module's triple.
  TargetLibraryInfoImpl TLII(Triple(M->getTargetTriple()));
  PM.add(new TargetLibraryInfoWrapperPass(TLII));

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
