/// \file LinkSupport.cpp
/// \brief Link support adds the helper functions to a lifeted module

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SourceMgr.h"

#include "revng/Model/Architecture.h"
#include "revng/Model/Binary.h"
#include "revng/Pipeline/AllRegistries.h"
#include "revng/Pipes/Kinds.h"
#include "revng/Pipes/LinkSupportPipe.h"
#include "revng/Pipes/ModelGlobal.h"
#include "revng/Support/Assert.h"
#include "revng/Support/CommandLine.h"
#include "revng/Support/ResourceFinder.h"

using namespace llvm::cl;
using namespace pipeline;
using namespace revng::pipes;

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

static std::string getSupporPath(const Context &Ctx) {
  const auto &Model = getModelFromContext(Ctx);
  const char *SupportConfig = Tracing ? "trace" : "normal";

  auto ArchName = getSupportName(Model.Architecture).str();
  std::string SupportSearchPath = ("/share/revng/support-" + ArchName + "-"
                                   + SupportConfig + ".ll");

  auto OptionalSupportPath = revng::ResourceFinder.findFile(SupportSearchPath);
  revng_assert(OptionalSupportPath.has_value(),
               "Cannot find the support module");
  std::string SupportPath = OptionalSupportPath.value();

  return SupportPath;
}

void revng::pipes::LinkSupportPipe::print(const pipeline::Context &Ctx,
                                          llvm::raw_ostream &OS) const {
  OS << "llvm-link <lifted-module> " << getSupporPath(Ctx) << " -o <output>"
     << "\n";
}

void revng::pipes::LinkSupportPipe::run(const Context &Ctx,
                                        LLVMContainer &TargetsList) {
  if (TargetsList.enumerate().empty())
    return;

  std::string SupportPath = getSupporPath(Ctx);

  llvm::SMDiagnostic Err;
  auto Module = llvm::parseIRFile(SupportPath,
                                  Err,
                                  TargetsList.getModule().getContext());
  revng_assert(Module != nullptr);

  auto Failed = llvm::Linker::linkModules(TargetsList.getModule(),
                                          std::move(Module));

  revng_assert(not Failed);
}

static pipeline::RegisterPipe<LinkSupportPipe> E4;
