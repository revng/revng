/// \file Lift.cpp

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

#include "revng/Lift/Lift.h"
#include "revng/Support/CommandLine.h"
#include "revng/Support/ResourceFinder.h"

#include "CodeGenerator.h"
#include "PTCInterface.h"

using namespace llvm::cl;

namespace {
#define DESCRIPTION desc("virtual address of the entry point where to start")
opt<unsigned long long> EntryPointAddress("entry",
                                          DESCRIPTION,
                                          value_desc("address"),
                                          cat(MainCategory));
#undef DESCRIPTION
alias A1("e",
         desc("Alias for -entry"),
         aliasopt(EntryPointAddress),
         cat(MainCategory));

} // namespace

char LiftPass::ID;

using Register = llvm::RegisterPass<LiftPass>;
static Register X("lift", "Lift Pass", true, true);

/// The interface with the PTC library.
PTCInterface ptc = {};

static std::string LibTinycodePath;
static std::string LibHelpersPath;
static std::string EarlyLinkedPath;

// When LibraryPointer is destroyed, the destructor calls
// LibraryDestructor::operator()(LibraryPointer::get()).
// The problem is that LibraryDestructor::operator() does not take arguments,
// while the destructor tries to pass a void * argument, so it does not match.
// However, LibraryDestructor is an alias for
// std::intgral_constant<decltype(&dlclose), &dlclose >, which has an implicit
// conversion operator to value_type, which unwraps the &dlclose from the
// std::integral_constant, making it callable.
using LibraryDestructor = std::integral_constant<int (*)(void *) noexcept,
                                                 &dlclose>;
using LibraryPointer = std::unique_ptr<void, LibraryDestructor>;

static void findFiles(model::Architecture::Values Architecture) {
  using namespace revng;

  std::string ArchName = model::Architecture::getQEMUName(Architecture).str();

  std::string LibtinycodeName = "/lib/libtinycode-" + ArchName + ".so";
  auto OptionalLibtinycode = ResourceFinder.findFile(LibtinycodeName);
  revng_assert(OptionalLibtinycode.has_value(), "Cannot find libtinycode");
  LibTinycodePath = OptionalLibtinycode.value();

  std::string LibHelpersName = "/lib/libtinycode-helpers-" + ArchName + ".bc";
  auto OptionalHelpers = ResourceFinder.findFile(LibHelpersName);
  revng_assert(OptionalHelpers.has_value(), "Cannot find tinycode helpers");
  LibHelpersPath = OptionalHelpers.value();

  std::string EarlyLinkedName = "/share/revng/early-linked-" + ArchName + ".ll";
  auto OptionalEarlyLinked = ResourceFinder.findFile(EarlyLinkedName);
  revng_assert(OptionalEarlyLinked.has_value(), "Cannot find early-linked.ll");
  EarlyLinkedPath = OptionalEarlyLinked.value();
}

/// Given an architecture name, loads the appropriate version of the PTC
/// library, and initializes the PTC interface.
///
/// \param Architecture the name of the architecture, e.g. "arm".
/// \param PTCLibrary a reference to the library handler.
///
/// \return EXIT_SUCCESS if the library has been successfully loaded.
static int loadPTCLibrary(LibraryPointer &PTCLibrary) {
  ptc_load_ptr_t PTCLoad = nullptr;
  void *LibraryHandle = nullptr;

  // Look for the library in the system's paths
  LibraryHandle = dlopen(LibTinycodePath.c_str(), RTLD_LAZY | RTLD_NODELETE);

  if (LibraryHandle == nullptr) {
    fprintf(stderr, "Couldn't load the PTC library: %s\n", dlerror());
    return EXIT_FAILURE;
  }

  // The library has been loaded, initialize the pointer, the caller will take
  // care of dlclose it from now on
  PTCLibrary.reset(LibraryHandle);

  // Obtain the address of the ptc_load entry point
  PTCLoad = reinterpret_cast<ptc_load_ptr_t>(dlsym(LibraryHandle, "ptc_load"));

  if (PTCLoad == nullptr) {
    fprintf(stderr, "Couldn't find ptc_load: %s\n", dlerror());
    return EXIT_FAILURE;
  }

  // Initialize the ptc interface
  if (PTCLoad(LibraryHandle, &ptc) != 0) {
    fprintf(stderr, "Couldn't find PTC functions.\n");
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

bool LiftPass::runOnModule(llvm::Module &M) {
  llvm::Task T(4, "Lift pass");
  const auto &ModelWrapper = getAnalysis<LoadModelWrapperPass>().get();
  const TupleTree<model::Binary> &Model = ModelWrapper.getReadOnlyModel();

  T.advance("findFiles", false);
  findFiles(Model->Architecture());

  // Load the appropriate libtyncode version
  T.advance("loadPTC", false);
  LibraryPointer PTCLibrary;
  if (loadPTCLibrary(PTCLibrary) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  // Get access to raw binary data
  RawBinaryView &RawBinary = getAnalysis<LoadBinaryWrapperPass>().get();

  T.advance("Construct CodeGenerator", false);
  CodeGenerator Generator(RawBinary,
                          &M,
                          Model,
                          LibHelpersPath,
                          EarlyLinkedPath,
                          model::Architecture::x86_64);

  std::optional<uint64_t> EntryPointAddressOptional;
  if (EntryPointAddress.getNumOccurrences() != 0)
    EntryPointAddressOptional = EntryPointAddress;
  T.advance("Translate", true);
  Generator.translate(EntryPointAddressOptional);

  return false;
}
