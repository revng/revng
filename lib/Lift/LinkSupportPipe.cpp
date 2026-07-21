/// Link support adds the helper functions to a lifeted module

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//
#include <string>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SourceMgr.h"

#include "revng/Lift/LinkSupportPipe.h"
#include "revng/Model/Architecture.h"
#include "revng/Model/Binary.h"
#include "revng/Support/Assert.h"
#include "revng/Support/CommandLine.h"
#include "revng/Support/ResourceFinder.h"

using namespace llvm::cl;
using namespace revng;

static opt<bool> Tracing("link-trace",
                         cat(MainCategory),
                         init(false),
                         desc("enable tracing when linking support"));

static llvm::StringRef getSupportName(model::Architecture::Values V) {
  using namespace model::Architecture;
  switch (V) {
  case Invalid:
  case Count:
    revng_abort();
    return "Invalid";
  case x86:
    return "i386";
  case x86_64:
    return "x86_64";
  case arm:
    return "arm";
  case aarch64:
    return "aarch64";
  case mips:
    return "mips";
  case mipsel:
    return "mipsel";
  case systemz:
    return "s390x";
  }
  revng_abort();
  return "Invalid";
}

static std::string getSupportPath(const model::Binary &Model) {
  const char *SupportConfig = Tracing ? "trace" : "normal";

  auto ArchName = getSupportName(Model.Architecture()).str();
  std::string SupportSearchPath = ("/share/revng/support-" + ArchName + "-"
                                   + SupportConfig + ".ll");

  auto OptionalSupportPath = ResourceFinder.findFile(SupportSearchPath);
  revng_assert(OptionalSupportPath.has_value(),
               "Cannot find the support module");
  std::string SupportPath = OptionalSupportPath.value();

  return SupportPath;
}

namespace revng::pypeline::piperuns {

LinkSupport::LinkSupport(const class Model &Model,
                         llvm::StringRef Config,
                         llvm::StringRef DynamicConfig,
                         LLVMRootContainer &ModuleContainer) :
  Binary(*Model.get().get()), ModuleContainer(ModuleContainer) {
}

void LinkSupport::run() {
  std::string SupportPath = getSupportPath(Binary);

  llvm::SMDiagnostic Err;
  llvm::LLVMContext &Context = ModuleContainer.getModule().getContext();
  auto Module = llvm::parseIRFile(SupportPath, Err, Context);
  revng_assert(Module != nullptr);

  bool Failed = llvm::Linker::linkModules(ModuleContainer.getModule(),
                                          std::move(Module));
  revng_assert(not Failed);
}

} // namespace revng::pypeline::piperuns
