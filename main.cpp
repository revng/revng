/// \file main.cpp
/// \brief This file takes care of handling command-line parameters and loading
/// the appropriate flavour of libtinycode-*.so

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <fstream>
#include <iostream>
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
#include "llvm/Object/Binary.h"
#include "llvm/Object/ELF.h"

// Local includes
#include "argparse.h"
#include "binaryfile.h"
#include "codegenerator.h"
#include "debug.h"
#include "ptcinterface.h"
#include "revamb.h"

PTCInterface ptc = {}; ///< The interface with the PTC library.
static std::string LibTinycodePath;
static std::string LibHelpersPath;
static std::string EarlyLinkedPath;

struct ProgramParameters {
  const char *InputPath;
  const char *OutputPath;
  size_t EntryPointAddress;
  DebugInfoType DebugInfo;
  const char *DebugPath;
  const char *LinkingInfoPath;
  const char *CoveragePath;
  const char *BBSummaryPath;
  int NoOSRA;
  int UseSections;
  int DetectFunctionsBoundaries;
  int NoLink;
  int External;
};

// When LibraryPointer is destroyed, the destructor calls
// LibraryDestructor::operator()(LibraryPointer::get()).
// The problem is that LibraryDestructor::operator() does not take arguments,
// while the destructor tries to pass a void * argument, so it does not match.
// However, LibraryDestructor is an alias for
// std::intgral_constant<decltype(&dlclose), &dlclose >, which has an implicit
// conversion operator to value_type, which unwraps the &dlclose from the
// std::integral_constant, making it callable.
using LibraryDestructor = std::integral_constant<decltype(&dlclose), &dlclose>;
using LibraryPointer = std::unique_ptr<void, LibraryDestructor>;

static const char *const Usage[] = {
  "revamb [options] [--] INFILE OUTFILE",
  nullptr,
};

static void findFiles(const char *Architecture) {
  // TODO: make this optional
  char *FullPath = realpath("/proc/self/exe", nullptr);
  assert(FullPath != nullptr);
  std::string Directory(dirname(FullPath));
  free(FullPath);

  // TODO: add other search paths?
  std::vector<std::string> SearchPaths;
#ifdef INSTALL_PATH
  SearchPaths.push_back(std::string(INSTALL_PATH) + "/lib");
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

  assert(LibtinycodeFound && "Couldn't find libtinycode and the helpers");
  assert(EarlyLinkedFound && "Couldn't find early-linked.ll");
}

/// Given an architecture name, loads the appropriate version of the PTC library,
/// and initializes the PTC interface.
///
/// \param Architecture the name of the architecture, e.g. "arm".
/// \param PTCLibrary a reference to the library handler.
///
/// \return EXIT_SUCCESS if the library has been successfully loaded.
static int loadPTCLibrary(LibraryPointer& PTCLibrary) {
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
  ptc_load = (ptc_load_ptr_t) dlsym(LibraryHandle, "ptc_load");

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

/// Parses the input arguments to the program.
///
/// \param Argc number of arguments.
/// \param Argv array of strings containing the arguments.
/// \param Parameters where to store the parsed parameters.
///
/// \return EXIT_SUCCESS if the parameters have been successfully parsed.
static int parseArgs(int Argc, const char *Argv[],
                     ProgramParameters *Parameters) {
  const char *DebugString = nullptr;
  const char *DebugLoggingString = nullptr;
  const char *EntryPointAddressString = nullptr;
  long long EntryPointAddress = 0;

  // Initialize argument parser
  struct argparse Arguments;
  struct argparse_option Options[] = {
    OPT_HELP(),
    OPT_GROUP("Input description"),
    OPT_STRING('e', "entry",
               &EntryPointAddressString,
               "virtual address of the entry point where to start."),
    OPT_STRING('s', "debug-path",
               &Parameters->DebugPath,
               "destination path for the generated debug source."),
    OPT_STRING('c', "coverage-path",
               &Parameters->CoveragePath,
               "destination path for the CSV containing translated ranges."),
    OPT_STRING('i', "linking-info",
               &Parameters->LinkingInfoPath,
               "destination path for the CSV containing linking info."),
    OPT_STRING('g', "debug-info",
               &DebugString,
               "emit debug information. Possible values are 'none' for no debug"
               " information, 'asm' for debug information referring to the"
               " assembly of the input file, 'ptc' for debug information"
               " referred to the Portable Tiny Code, or 'll' for debug"
               " information referred to the LLVM IR."),
    OPT_STRING('d', "debug",
               &DebugLoggingString,
               "enable verbose logging."),
    OPT_BOOLEAN('O', "no-osra", &Parameters->NoOSRA,
                "disable OSRA."),
    OPT_BOOLEAN('L', "no-link", &Parameters->NoLink,
                "do not link the output to QEMU helpers."),
    OPT_BOOLEAN('E', "external", &Parameters->External,
                "set CSVs linkage to external, useful for debugging purposes."),
    OPT_BOOLEAN('S', "use-sections", &Parameters->UseSections,
                "use section informations, if available."),
    OPT_STRING('b', "bb-summary",
               &Parameters->BBSummaryPath,
               "destination path for the CSV containing the statistics about "
               "the translated basic blocks."),
    OPT_BOOLEAN('f', "functions-boundaries",
                &Parameters->DetectFunctionsBoundaries,
                "enable functions boundaries detection."),
    OPT_END(),
  };

  argparse_init(&Arguments, Options, Usage, 0);
  argparse_describe(&Arguments, "\nrevamb.",
                    "\nTranslates a binary into a program for a different "
                    "architecture.\n");
  Argc = argparse_parse(&Arguments, Argc, Argv);

  // Handle positional arguments
  if (Argc != 2) {
    fprintf(stderr, "Too many arguments.\n");
    return EXIT_FAILURE;
  }

  Parameters->InputPath = Argv[0];
  Parameters->OutputPath = Argv[1];

  // Check parameters
  if (EntryPointAddressString != nullptr) {
    if (sscanf(EntryPointAddressString, "%lld", &EntryPointAddress) != 1) {
      fprintf(stderr, "Entry point parameter (-e, --entry) is not a"
              " number.\n");
      return EXIT_FAILURE;
    }

    Parameters->EntryPointAddress = (size_t) EntryPointAddress;
  }

  if (DebugString != nullptr) {
    if (strcmp("none", DebugString) == 0) {
      Parameters->DebugInfo = DebugInfoType::None;
    } else if (strcmp("asm", DebugString) == 0) {
      Parameters->DebugInfo = DebugInfoType::OriginalAssembly;
    } else if (strcmp("ptc", DebugString) == 0) {
      Parameters->DebugInfo = DebugInfoType::PTC;
    } else if (strcmp("ll", DebugString) == 0) {
      Parameters->DebugInfo = DebugInfoType::LLVMIR;
    } else {
      fprintf(stderr, "Unexpected value for the debug type parameter"
              " (-g, --debug).\n");
      return EXIT_FAILURE;
    }
  }

  if (DebugLoggingString != nullptr) {
    DebuggingEnabled = true;
    std::string Input(DebugLoggingString);
    std::stringstream Stream(Input);
    std::string Type;
    while (std::getline(Stream, Type, ','))
      enableDebugFeature(Type.c_str());
  }

  if (Parameters->DebugPath == nullptr)
    Parameters->DebugPath = "";

  if (Parameters->LinkingInfoPath == nullptr)
    Parameters->LinkingInfoPath = "";

  if (Parameters->CoveragePath == nullptr)
    Parameters->CoveragePath = "";

  if (Parameters->BBSummaryPath == nullptr)
    Parameters->BBSummaryPath = "";

  return EXIT_SUCCESS;
}

int main(int argc, const char *argv[]) {
  // Parse arguments
  ProgramParameters Parameters {};
  if (parseArgs(argc, argv, &Parameters) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  BinaryFile TheBinary(Parameters.InputPath, Parameters.UseSections);

  findFiles(TheBinary.architecture().name());

  // Load the appropriate libtyncode version
  LibraryPointer PTCLibrary;
  if (loadPTCLibrary(PTCLibrary) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  // Translate everything
  Architecture TargetArchitecture;
  CodeGenerator Generator(TheBinary,
                          TargetArchitecture,
                          std::string(Parameters.OutputPath),
                          LibHelpersPath,
                          EarlyLinkedPath,
                          Parameters.DebugInfo,
                          std::string(Parameters.DebugPath),
                          std::string(Parameters.LinkingInfoPath),
                          std::string(Parameters.CoveragePath),
                          std::string(Parameters.BBSummaryPath),
                          !Parameters.NoOSRA,
                          Parameters.DetectFunctionsBoundaries,
                          !Parameters.NoLink,
                          Parameters.External);

  Generator.translate(Parameters.EntryPointAddress);

  Generator.serialize();

  return EXIT_SUCCESS;
}
