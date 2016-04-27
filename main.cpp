/// \file
/// \brief This file takes care of handling command-line parameters and loading
/// the appropriate flavour of libtinycode-*.so

// Standard includes
#include <cstdio>
#include <vector>
#include <memory>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

extern "C" {
#include <dlfcn.h>
}

// LLVM includes
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/ELF.h"

// Local includes
#include "debug.h"
#include "revamb.h"
#include "argparse.h"
#include "ptcinterface.h"
#include "codegenerator.h"

PTCInterface ptc = {}; ///< The interface with the PTC library.

struct ProgramParameters {
  const char *Architecture;
  const char *InputPath;
  const char *OutputPath;
  size_t EntryPointAddress;
  DebugInfoType DebugInfo;
  const char *DebugPath;
  const char *LinkingInfoPath;
  const char *CoveragePath;
  bool NoOSRA;
  bool EnableTracing;
};

using LibraryDestructor = GenericFunctor<decltype(&dlclose), &dlclose>;
using LibraryPointer = std::unique_ptr<void, LibraryDestructor>;

static const char *const Usage[] = {
  "revamb [options] [--] INFILE OUTFILE",
  nullptr,
};

/// Given an architecture name, loads the appropriate version of the PTC library,
/// and initializes the PTC interface.
///
/// \param Architecture the name of the architecture, e.g. "arm".
/// \param PTCLibrary a reference to the library handler.
///
/// \return EXIT_SUCCESS if the library has been successfully loaded.
static int loadPTCLibrary(const char *Architecture,
                          LibraryPointer& PTCLibrary) {
  ptc_load_ptr_t ptc_load = nullptr;
  void *LibraryHandle = nullptr;

  // Build library name
  std::stringstream LibraryName;
  LibraryName << QEMU_LIB_PATH << "/libtinycode-" << Architecture << ".so";

  // Look for the library in the system's paths
  LibraryHandle = dlopen(LibraryName.str().c_str(), RTLD_LAZY);

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
    OPT_STRING('a', "architecture",
               &Parameters->Architecture,
               "the input architecture."),
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
                "disable OSRA"),
    OPT_BOOLEAN('t', "tracing", &Parameters->EnableTracing,
                "enable PC tracing in the output binary (through newPC)"),
    OPT_END(),
  };

  argparse_init(&Arguments, Options, Usage, 0);
  argparse_describe(&Arguments, "\nPTC translator.",
                    "\nTranslates a binary into QEMU Portable Tiny Code.\n");
  Argc = argparse_parse(&Arguments, Argc, Argv);

  // Handle positional arguments
  if (Argc != 2) {
    fprintf(stderr, "Too many arguments.\n");
    return EXIT_FAILURE;
  }

  Parameters->InputPath = Argv[0];
  Parameters->OutputPath = Argv[1];

  // Check parameters
  if (Parameters->Architecture == nullptr) {
    fprintf(stderr, "Please specify the input architecture.\n");
    return EXIT_FAILURE;
  }

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

  return EXIT_SUCCESS;
}

int main(int argc, const char *argv[]) {
  // Parse arguments
  ProgramParameters Parameters {};
  if (parseArgs(argc, argv, &Parameters) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  // Load the appropriate libtyncode version
  LibraryPointer PTCLibrary;
  if (loadPTCLibrary(Parameters.Architecture, PTCLibrary) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::stringstream HelpersPath;
  HelpersPath << QEMU_LIB_PATH
              << "/libtinycode-helpers-"
              << Parameters.Architecture
              << ".ll";

  // Translate everything
  Architecture TargetArchitecture;
  CodeGenerator Generator(std::string(Parameters.InputPath),
                          TargetArchitecture,
                          std::string(Parameters.OutputPath),
                          HelpersPath.str(),
                          Parameters.DebugInfo,
                          std::string(Parameters.DebugPath),
                          std::string(Parameters.LinkingInfoPath),
                          std::string(Parameters.CoveragePath),
                          !Parameters.NoOSRA,
                          Parameters.EnableTracing);

  Generator.translate(Parameters.EntryPointAddress, "root");

  Generator.serialize();

  return EXIT_SUCCESS;
}
