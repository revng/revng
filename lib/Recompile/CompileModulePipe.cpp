/// \file CompileModule.cpp
/// The compile module pipe transforms an llvm module into an object file.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <memory>

#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/IR/AutoUpgrade.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Target/TargetMachine.h"

#include "revng/Pipeline/AllRegistries.h"
#include "revng/Pipeline/LLVMContainer.h"
#include "revng/Pipeline/Target.h"
#include "revng/Pipes/Kinds.h"
#include "revng/Recompile/CompileModulePipe.h"
#include "revng/Support/Assert.h"
#include "revng/Support/IRAnnotators.h"
#include "revng/Support/IRHelpers.h"
#include "revng/Support/OriginalAssemblyAnnotationWriter.h"

using namespace llvm;
using namespace llvm::codegen;
using namespace std;
using namespace pipeline;
using namespace ::revng::pipes;
using namespace sys;
using namespace cl;

static cl::opt<char> OptLevel("compile-opt-level",
                              cl::desc("Optimization level. [-O0, -O1, -O2, or "
                                       "-O3] "
                                       "(default = '-O2')"),
                              cl::Prefix,
                              cl::ZeroOrMore,
                              cl::init(' '));

static void compileModuleRunImpl(const Context &Ctx,
                                 LLVMContainer &Module,
                                 ObjectFileContainer &TargetBinary) {
  using namespace revng;

  auto Enumeration = Module.enumerate();
  if (not Enumeration.contains(pipeline::Target(kinds::Root))
      and not Enumeration.contains(pipeline::Target(kinds::IsolatedRoot)))
    return;

  if (Enumeration.contains(pipeline::Target(kinds::IsolatedRoot))
      and not Enumeration.contains(kinds::Isolated.allTargets(Ctx)))
    return;

  StringMap<llvm::cl::Option *> &RegOptions(getRegisteredOptions());
  getOption<bool>(RegOptions, "disable-machine-licm")->setInitialValue(true);

  llvm::Module *M = &Module.getModule();

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
                                            getRelocModel(),
                                            M->getCodeModel(),
                                            OLvl);
  unique_ptr<TargetMachine> Target(Ptr);

  // Add the target data from the target machine, if it exists, or the module.
  M->setDataLayout(Target->createDataLayout());

  // This needs to be done after setting datalayout since it calls verifier
  // to check debug info whereas verifier relies on correct datalayout.
  UpgradeDebugInfo(*M);

  LLVMTargetMachine &LLVMTM = static_cast<LLVMTargetMachine &>(*Target);
  auto *MMIWP = new MachineModuleInfoWrapperPass(&LLVMTM);

  std::error_code EC;
  raw_fd_ostream OutputStream(TargetBinary.getOrCreatePath(), EC);
  revng_assert(!EC);

  // Create pass manager
  legacy::PassManager PM;

  // Add an appropriate TargetLibraryInfo pass for the module's triple.
  TargetLibraryInfoImpl TLII(Triple(M->getTargetTriple()));
  PM.add(new TargetLibraryInfoWrapperPass(TLII));

  bool Err = Target->addPassesToEmitFile(PM,
                                         OutputStream,
                                         nullptr,
                                         CGFT_ObjectFile,
                                         true,
                                         MMIWP);
  revng_assert(not Err);
  revng::verify(M);
  PM.run(*M);
  revng::verify(M);

  auto Path = TargetBinary.path();

  auto Permissions = cantFail(errorOrToExpected(fs::getPermissions(*Path)));
  Permissions = Permissions | fs::owner_exe;
  fs::setPermissions(*TargetBinary.path(), Permissions);
}

void CompileModule::run(const ExecutionContext &Ctx,
                        LLVMContainer &Module,
                        ObjectFileContainer &TargetBinary) {
  compileModuleRunImpl(Ctx.getContext(), Module, TargetBinary);
}

void CompileIsolatedModule::run(const ExecutionContext &Ctx,
                                LLVMContainer &Module,
                                ObjectFileContainer &TargetBinary) {
  compileModuleRunImpl(Ctx.getContext(), Module, TargetBinary);
}

static RegisterPipe<CompileModule> E2;
static RegisterPipe<CompileIsolatedModule> E3;
