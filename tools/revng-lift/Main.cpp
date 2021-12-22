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
#include "llvm/Support/raw_os_ostream.h"

#include "revng/Lift/BinaryFile.h"
#include "revng/Lift/CodeGenerator.h"
#include "revng/Lift/PTCInterface.h"
#include "revng/Support/CommandLine.h"
#include "revng/Support/Debug.h"
#include "revng/Support/IRAnnotators.h"
#include "revng/Support/IRHelpers.h"
#include "revng/Support/OriginalAssemblyAnnotationWriter.h"
#include "revng/Support/ResourceFinder.h"
#include "revng/Support/Statistics.h"
#include "revng/Support/revng.h"

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

namespace DebugInfoType {

/// \brief Type of debug information to produce
enum Values {
  /// No debug information
  None,
  /// Produce a file containing the assembly code of the input binary
  OriginalAssembly,
  /// Produce the PTC as translated by libtinycode
  PTC,
  /// Prduce an LLVM IR with debug metadata referring to itself
  LLVMIR
};

} // namespace DebugInfoType

namespace DIT = DebugInfoType;

static auto X = values(clEnumValN(DIT::None, "none", "no debug information"),
                       clEnumValN(DIT::OriginalAssembly,
                                  "asm",
                                  "debug information referred to the "
                                  "assembly "
                                  "of the input file"),
                       clEnumValN(DIT::PTC,
                                  "ptc",
                                  "debug information referred to the "
                                  "Portable "
                                  "Tiny Code"),
                       clEnumValN(DIT::LLVMIR,
                                  "ll",
                                  "debug information referred to the LLVM "
                                  "IR"));
static opt<DIT::Values> DebugInfo("debug-info",
                                  desc("emit debug information"),
                                  X,
                                  cat(MainCategory),
                                  init(DIT::LLVMIR));

static alias A6("g",
                desc("Alias for -debug-info"),
                aliasopt(DebugInfo),
                cat(MainCategory));

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
  ptc_load_ptr_t ptc_load = nullptr;
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
  // Enable LLVM stack trace
  llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);

  HideUnrelatedOptions({ &MainCategory });
  ParseCommandLineOptions(argc, argv);
  installStatistics();

  revng_check(BaseAddress % 4096 == 0, "Base address is not page aligned");

  BinaryFile TheBinary(InputPath, BaseAddress);

  findFiles(TheBinary.architecture().name());

  // Load the appropriate libtyncode version
  LibraryPointer PTCLibrary;
  if (loadPTCLibrary(PTCLibrary) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  // Translate everything
  Architecture TargetArchitecture;
  llvm::LLVMContext Context;
  llvm::Module M("top", Context);
  CodeGenerator Generator(TheBinary,
                          TargetArchitecture,
                          &M,
                          LibHelpersPath,
                          EarlyLinkedPath);

  llvm::Optional<uint64_t> EntryPointAddressOptional;
  if (EntryPointAddress.getNumOccurrences() != 0)
    EntryPointAddressOptional = EntryPointAddress;
  Generator.translate(EntryPointAddressOptional);

  OriginalAssemblyAnnotationWriter OAAW(M.getContext());

  switch (DebugInfo) {
  case DebugInfoType::None:
    break;

  case DebugInfoType::OriginalAssembly:
    createOriginalAssemblyDebugInfo(&M, OutputPath);
    break;

  case DebugInfoType::PTC:
    createPTCDebugInfo(&M, OutputPath);
    break;

  case DebugInfoType::LLVMIR:
    createSelfReferencingDebugInfo(&M, OutputPath, &OAAW);
    break;
  }

  std::ofstream OutputStream(OutputPath);
  llvm::raw_os_ostream LLVMOutputStream(OutputStream);
  M.print(LLVMOutputStream, &OAAW);
}
