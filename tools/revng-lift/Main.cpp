/// \file main.cpp
/// \brief This file takes care of handling command-line parameters and loading
/// the appropriate flavour of libtinycode-*.so

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

extern "C" {
#include <dlfcn.h>
#include <libgen.h>
#include <unistd.h>
}

// LLVM includes
#include "llvm/ADT/ArrayRef.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/ELF.h"

// Local libraries includes
#include "revng/Support/CommandLine.h"
#include "revng/Support/Debug.h"
#include "revng/Support/Statistics.h"
#include "revng/Support/revng.h"

// Local includes
#include "BinaryFile.h"
#include "CodeGenerator.h"
#include "PTCInterface.h"

PTCInterface ptc = {}; ///< The interface with the PTC library.

using namespace llvm::cl;

using std::string;

// TODO: drop short aliases

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

#define DESCRIPTION desc("base address where dynamic objects should be loaded")
opt<unsigned long long> BaseAddress("base",
                                    DESCRIPTION,
                                    value_desc("address"),
                                    cat(MainCategory),
                                    init(0x50000000));
#undef DESCRIPTION

#define DESCRIPTION desc("Alias for -base")
alias A2("B", DESCRIPTION, aliasopt(BaseAddress), cat(MainCategory));
#undef DESCRIPTION

opt<string> InputPath(Positional, Required, desc("<input path>"));
opt<string> OutputPath(Positional, Required, desc("<output path>"));

} // namespace

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
  // TODO: make this optional
  char *FullPath = realpath("/proc/self/exe", nullptr);
  revng_assert(FullPath != nullptr);
  std::string Directory(dirname(FullPath));
  free(FullPath);

  // TODO: add other search paths?
  std::vector<std::string> SearchPaths;
#ifdef INSTALL_PATH
  SearchPaths.push_back(std::string(INSTALL_PATH) + "/lib");
  SearchPaths.push_back(std::string(INSTALL_PATH) + "/share/revng");
#endif
  SearchPaths.push_back(Directory);
#ifdef QEMU_INSTALL_PATH
  SearchPaths.push_back(std::string(QEMU_INSTALL_PATH) + "/lib");
#endif

  bool LibtinycodeFound = false;
  bool EarlyLinkedFound = false;
  for (auto &Path : SearchPaths) {

    if (not LibtinycodeFound) {
      std::stringstream LibraryPath;
      LibraryPath << Path << "/libtinycode-" << Architecture << ".so";
      std::stringstream HelpersPath;
      HelpersPath << Path << "/libtinycode-helpers-" << Architecture << ".ll";
      if (access(LibraryPath.str().c_str(), F_OK) != -1
          && access(HelpersPath.str().c_str(), F_OK) != -1) {
        LibTinycodePath = LibraryPath.str();
        LibHelpersPath = HelpersPath.str();
        LibtinycodeFound = true;
      }
    }

    if (not EarlyLinkedFound) {
      std::stringstream TestPath;
      TestPath << Path << "/early-linked-" << Architecture << ".ll";
      if (access(TestPath.str().c_str(), F_OK) != -1) {
        EarlyLinkedPath = TestPath.str();
        EarlyLinkedFound = true;
      }
    }
  }

  revng_assert(LibtinycodeFound, "Couldn't find libtinycode and the helpers");
  revng_assert(EarlyLinkedFound, "Couldn't find early-linked.ll");
}

/// Given an architecture name, loads the appropriate version of the PTC
/// library, and initializes the PTC interface.
///
/// \param Architecture the name of the architecture, e.g. "arm".
/// \param PTCLibrary a reference to the library handler.
///
/// \return EXIT_SUCCESS if the library has been successfully loaded.
static int loadPTCLibrary(LibraryPointer &PTCLibrary) {
  ptc_load_ptr_t ptc_load = nullptr;
  void *LibraryHandle = nullptr;

  // Look for the library in the system's paths
  LibraryHandle = dlopen(LibTinycodePath.c_str(), RTLD_LAZY);

  if (LibraryHandle == nullptr) {
    fprintf(stderr, "Couldn't load the PTC library: %s\n", dlerror());
    return EXIT_FAILURE;
  }

  // The library has been loaded, initialize the pointer, the caller will take
  // care of dlclose it from now on
  PTCLibrary.reset(LibraryHandle);

  // Obtain the address of the ptc_load entry point
  ptc_load = reinterpret_cast<ptc_load_ptr_t>(dlsym(LibraryHandle, "ptc_load"));

  if (ptc_load == nullptr) {
    fprintf(stderr, "Couldn't find ptc_load: %s\n", dlerror());
    return EXIT_FAILURE;
  }

  // Initialize the ptc interface
  if (ptc_load(LibraryHandle, &ptc) != 0) {
    fprintf(stderr, "Couldn't find PTC functions.\n");
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

int main(int argc, const char *argv[]) {
  HideUnrelatedOptions({ &MainCategory });
  ParseCommandLineOptions(argc, argv);
  installStatistics();

  BinaryFile TheBinary(InputPath, BaseAddress);

  findFiles(TheBinary.architecture().name());

  // Load the appropriate libtyncode version
  LibraryPointer PTCLibrary;
  if (loadPTCLibrary(PTCLibrary) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  // Translate everything
  Architecture TargetArchitecture;
  llvm::LLVMContext RevambGlobalContext;
  CodeGenerator Generator(TheBinary,
                          TargetArchitecture,
                          RevambGlobalContext,
                          std::string(OutputPath),
                          LibHelpersPath,
                          EarlyLinkedPath);

  Generator.translate(EntryPointAddress);
  Generator.serialize();

  return EXIT_SUCCESS;
}
