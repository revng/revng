/// \file Lift.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Lift/LibTcg.h"
#include "revng/Lift/Lift.h"
#include "revng/Support/CommandLine.h"
#include "revng/Support/IRHelpers.h"
#include "revng/Support/ResourceFinder.h"

#include "CodeGenerator.h"

using namespace llvm::cl;

namespace {
const char *EntryDescStr = "virtual address of the entry point where to start";
opt<unsigned long long> EntryPointAddress("entry",
                                          desc(EntryDescStr),
                                          value_desc("address"),
                                          cat(MainCategory));
alias A1("e",
         desc("Alias for -entry"),
         aliasopt(EntryPointAddress),
         cat(MainCategory));

} // namespace

char LiftPass::ID;

using Register = llvm::RegisterPass<LiftPass>;
static Register X("lift", "Lift Pass", true, true);

struct ExternalFilePaths {
  std::string LibHelpers;
  std::string EarlyLinked;
};

static ExternalFilePaths
findExternalFilePaths(const model::Architecture::Values Architecture) {
  // What symbols from the revng namespace are actually used here?
  using namespace revng;

  const std::string ArchName = model::Architecture::getQEMUName(Architecture)
                                 .str();

  ExternalFilePaths Paths = {};

  // Note: here we use the slim version of the helpers, i.e., where we only have
  //       definitions for revng_inline functions.
  const std::string LibHelpersName = "/share/revng/libtcg-helpers-annotated"
                                     "-slim-"
                                     + ArchName + ".bc";
  auto OptionalHelpers = ResourceFinder.findFile(LibHelpersName);
  revng_assert(OptionalHelpers.has_value(), "Cannot find tinycode helpers");
  Paths.LibHelpers = OptionalHelpers.value();

  const std::string EarlyLinkedName = "/share/revng/early-linked-" + ArchName
                                      + ".ll";
  auto OptionalEarlyLinked = ResourceFinder.findFile(EarlyLinkedName);
  revng_assert(OptionalEarlyLinked.has_value(), "Cannot find early-linked.ll");

  Paths.EarlyLinked = OptionalEarlyLinked.value();

  return Paths;
}

bool LiftPass::runOnModule(llvm::Module &M) {
  llvm::Task T(4, "Lift pass");
  const auto &ModelWrapper = getAnalysis<LoadModelWrapperPass>().get();
  const TupleTree<model::Binary> &Model = ModelWrapper.getReadOnlyModel();

  T.advance("findFiles", false);
  const auto Paths = findExternalFilePaths(Model->Architecture());

  // Look for the library in the system's paths
  T.advance("Load libtcg", false);
  auto TheLibTcg = LibTcg::get(Model->Architecture());

  // Get access to raw binary data
  RawBinaryView &RawBinary = getAnalysis<LoadBinaryWrapperPass>().get();

  T.advance("Construct CodeGenerator", false);
  CodeGenerator Generator(RawBinary,
                          &M,
                          Model,
                          Paths.LibHelpers,
                          Paths.EarlyLinked,
                          model::Architecture::x86_64);

  std::optional<uint64_t> EntryPointAddressOptional;
  if (EntryPointAddress.getNumOccurrences() != 0)
    EntryPointAddressOptional = EntryPointAddress;
  T.advance("Translate", true);

  Generator.translate(TheLibTcg, EntryPointAddressOptional);

  sortModule(M);

  return false;
}
