/// \file Main.cpp
/// \brief This file takes care of handling command-line parameters and loading
/// the appropriate flavour of libtinycode-*.so

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Optional.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/ELF.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/DynamicLibrary.h"

#include "revng/Support/CommandLine.h"
#include "revng/Support/Debug.h"
#include "revng/Support/ResourceFinder.h"
#include "revng/Support/Statistics.h"
#include "revng/Support/revng.h"

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
                                    init(0x400000));
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
/// \return EXIT_SUCCESS if the library has been successfully loaded.
static int loadPTCLibrary() {
  using namespace llvm::sys;

  ptc_load_ptr_t ptc_load = nullptr;
  void *LibraryHandle = nullptr;

  // Look for the library in the system's paths
  bool Result = DynamicLibrary::LoadLibraryPermanently(LibTinycodePath.c_str());
  revng_assert(Result);

  // Obtain the address of the ptc_load entry point
  ptc_load = reinterpret_cast<ptc_load_ptr_t>(DynamicLibrary::SearchForAddressOfSymbol("ptc_load"));
  revng_assert(ptc_load != nullptr);

  // Initialize the ptc interface
  if (ptc_load(LibraryHandle, &ptc) != 0) {
    fprintf(stderr, "Couldn't find PTC functions.\n");
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

int main(int argc, const char *argv[]) {
  // Enable LLVM stack trace
  llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);

  HideUnrelatedOptions({ &MainCategory });
  ParseCommandLineOptions(argc, argv);
  installStatistics();

  revng_check(BaseAddress % 4096 == 0, "Base address is not page aligned");

  BinaryFile TheBinary(InputPath, BaseAddress);

  findFiles(TheBinary.architecture().name());

  // Load the appropriate libtyncode version
  if (loadPTCLibrary() != EXIT_SUCCESS)
    return EXIT_FAILURE;

  // Translate everything
  Architecture TargetArchitecture;
  llvm::LLVMContext Context;
  CodeGenerator Generator(TheBinary,
                          TargetArchitecture,
                          Context,
                          std::string(OutputPath),
                          LibHelpersPath,
                          EarlyLinkedPath);

  llvm::Optional<uint64_t> EntryPointAddressOptional;
  if (EntryPointAddress.getNumOccurrences() != 0)
    EntryPointAddressOptional = EntryPointAddress;
  Generator.translate(EntryPointAddressOptional);
  Generator.serialize();

  return EXIT_SUCCESS;
}
