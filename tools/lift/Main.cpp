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
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/ELF.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/raw_os_ostream.h"

#include "revng/Lift/Lift.h"
#include "revng/Model/Importer/Binary/BinaryImporter.h"
#include "revng/Model/Importer/Binary/BinaryImporterOptions.h"
#include "revng/Model/Importer/Dwarf/DwarfImporter.h"
#include "revng/Model/SerializeModelPass.h"
#include "revng/Support/CommandLine.h"
#include "revng/Support/Debug.h"
#include "revng/Support/IRAnnotators.h"
#include "revng/Support/IRHelpers.h"
#include "revng/Support/OriginalAssemblyAnnotationWriter.h"
#include "revng/Support/ResourceFinder.h"
#include "revng/Support/Statistics.h"

// TODO: drop short aliases

namespace {

using namespace llvm::cl;
using std::string;

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

int main(int argc, const char *argv[]) {
  // Enable LLVM stack trace
  llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);

  HideUnrelatedOptions({ &MainCategory });
  ParseCommandLineOptions(argc, argv);
  installStatistics();

  auto MaybeBuffer = llvm::MemoryBuffer::getFileOrSTDIN(InputPath);
  if (not MaybeBuffer) {
    dbg << "Couldn't open input file\n";
    return EXIT_FAILURE;
  }
  llvm::MemoryBuffer &Buffer = **MaybeBuffer;

  TupleTree<model::Binary> Model;

  revng_check(not importBinary(Model, InputPath, BaseAddress));

  if (ImportDebugInfo.size() > 0) {
    DwarfImporter Importer(Model);
    for (const std::string &Path : ImportDebugInfo)
      Importer.import(Path);
  }

  // Translate everything
  llvm::LLVMContext Context;
  llvm::Module M("top", Context);
  writeModel(*Model, M);

  // Perform lifting
  llvm::legacy::PassManager PM;
  PM.add(new LoadModelWrapperPass(Model));
  PM.add(new LoadBinaryWrapperPass(Buffer.getBuffer()));
  PM.add(new LiftPass);
  PM.run(M);

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
