/// \file Lift.cpp
/// \brief Lift transform a binary into a llvm module

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

extern "C" {
#include "dlfcn.h"
}

#include "revng/Lift/BinaryFile.h"
#include "revng/Lift/CodeGenerator.h"
#include "revng/Lift/PTCInterface.h"
#include "revng/Pipeline/Registry.h"
#include "revng/Pipes/FileContainer.h"
#include "revng/Pipes/Kinds.h"
#include "revng/Pipes/LiftPipe.h"
#include "revng/Pipes/ModelGlobal.h"
#include "revng/Pipes/RootKind.h"
#include "revng/Support/IRAnnotators.h"
#include "revng/Support/ResourceFinder.h"

using namespace llvm::cl;

#define DESCRIPTION desc("virtual address of the entry point where to start")
static opt<unsigned long long> EntryPointAddress("entry",
                                                 DESCRIPTION,
                                                 value_desc("address"),
                                                 cat(MainCategory));
#undef DESCRIPTION
static alias A1("e",
                desc("Alias for -entry"),
                aliasopt(EntryPointAddress),
                cat(MainCategory));

#define DESCRIPTION desc("base address where dynamic objects should be loaded")
opt<unsigned long long> BaseAddress("base",
                                    DESCRIPTION,
                                    value_desc("address"),
                                    cat(MainCategory),
                                    init(0x400000));
#undef DESCRIPTION

#define DESCRIPTION desc("Alias for -base")
static alias A2("B", DESCRIPTION, aliasopt(BaseAddress), cat(MainCategory));
#undef DESCRIPTION

PTCInterface ptc = {}; ///< The interface with the PTC library.

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

static void findFiles(const char *Architecture) {
  using namespace revng;

  std::string ArchName(Architecture);

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
  ptc_load_ptr_t PtcLoad = nullptr;
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
  PtcLoad = reinterpret_cast<ptc_load_ptr_t>(dlsym(LibraryHandle, "ptc_load"));

  if (PtcLoad == nullptr) {
    fprintf(stderr, "Couldn't find ptc_load: %s\n", dlerror());
    return EXIT_FAILURE;
  }

  // Initialize the ptc interface
  if (PtcLoad(LibraryHandle, &ptc) != 0) {
    fprintf(stderr, "Couldn't find PTC functions.\n");
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

using namespace revng::pipes;
using namespace pipeline;

void LiftPipe::run(Context &Ctx,
                   const FileContainer &SourceBinary,
                   LLVMContainer &TargetsList) {
  if (not SourceBinary.exists())
    return;

  auto &Model = getWritableModelFromContext(Ctx);
  revng_check(BaseAddress % 4096 == 0, "Base address is not page aligned");

  BinaryFile TheBinary(SourceBinary.path()->str(), BaseAddress);

  findFiles(TheBinary.architecture().name());

  // Load the appropriate libtyncode version
  LibraryPointer PTCLibrary;
  auto Result = loadPTCLibrary(PTCLibrary);
  revng_assert(Result == EXIT_SUCCESS);

  // Translate everything
  Architecture TargetArchitecture;
  CodeGenerator Generator(TheBinary,
                          TargetArchitecture,
                          &TargetsList.getModule(),
                          Model,
                          LibHelpersPath,
                          EarlyLinkedPath);

  llvm::Optional<uint64_t> EntryPointAddressOptional;
  if (EntryPointAddress.getNumOccurrences() != 0)
    EntryPointAddressOptional = EntryPointAddress;
  Generator.translate(EntryPointAddressOptional);
}

static RegisterPipe<LiftPipe> E1;
