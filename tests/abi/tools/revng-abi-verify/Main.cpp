/// \file Main.cpp
/// \brief This tool is used for ABI conversion testing. It's purpose is to
/// make sure that the function still corresponds to the intended ABI no matter
/// what kind of work was done on top of it. It works with both
/// `RawFunctionType` and `CABIFunctionType`.
///
/// \note: This tool requires a runtime abi artifact. It can be obtained using
/// revng-qa. For more information see the corresponding repository.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <fstream>
#include <streambuf>
#include <string>

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Signals.h"

#include "Verify.h"

namespace Options {

using namespace llvm::cl;

OptionCategory Category("ABI Options");

template<std::size_t... Indices>
static auto packValues(std::integer_sequence<size_t, Indices...>) {
  using namespace llvm::cl;
  using namespace model::ABI;
  return values(OptionEnumValue{ getName(Values(Indices)),
                                 int(Indices),
                                 getDescription(Values(Indices)) }...);
}

constexpr const char *Description = "Specifies the default ABI of the binary.";
constexpr auto ABIList = std::make_index_sequence<model::ABI::Count - 1>{};
static auto Values = packValues(ABIList);

extern llvm::cl::OptionCategory ModelPassCategory;
llvm::cl::opt<model::ABI::Values> TargetABI("abi",
                                            Values,
                                            llvm::cl::desc(Description),
                                            llvm::cl::cat(Category));

constexpr const char *FnDesc = "<input file name>";
opt<std::string> Filename(Positional, Required, desc(FnDesc), cat(Category));

constexpr const char *AtDesc = "<runtime abi artifact name>";
opt<std::string> Artifact(Positional, Required, desc(AtDesc), cat(Category));

opt<std::string> Output("o",
                        desc("Optional output filename, if not specified, the "
                             "output is dumped to `stdout`"),
                        value_desc("path"),
                        cat(Category));

} // namespace Options

int main(int argc, const char *argv[]) {
  // Enable LLVM stack trace
  llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);

  llvm::cl::HideUnrelatedOptions(Options::Category);
  llvm::cl::ParseCommandLineOptions(argc, argv);

  llvm::raw_fd_ostream *OutputStreamPtr;
  if (Options::Output.empty()) {
    OutputStreamPtr = &llvm::outs();
  } else {
    std::error_code EC;
    static llvm::raw_fd_ostream OutputStream(Options::Output, EC);
    if (!EC || OutputStream.has_error()) {
      dbg << "Unable to open an output file: '" << Options::Output << "'.\n";
      return ExitCode::FailedOpeningTheOutputFile;
    }
    OutputStreamPtr = &OutputStream;
  }

  auto InputOrError = llvm::MemoryBuffer::getFileOrSTDIN(Options::Filename);
  if (!InputOrError) {
    dbg << "Unable to open an input file: '" << Options::Filename << "'.\n";
    return ExitCode::FailedOpeningTheInputFile;
  }
  llvm::StringRef InputText = (*InputOrError)->getBuffer();

  auto ArtifactOrError = llvm::MemoryBuffer::getFileOrSTDIN(Options::Artifact);
  if (!ArtifactOrError) {
    dbg << "Unable to open the artifact file: '" << Options::Artifact << "'.\n";
    return ExitCode::FailedOpeningTheArtifactFile;
  }
  llvm::StringRef Artifact = (*ArtifactOrError)->getBuffer();

  auto Deserialized = TupleTree<model::Binary>::deserialize(InputText);
  if (!Deserialized) {
    dbg << "Unable to deserialize the model: '" << Options::Filename << "'.\n";
    return ExitCode::FailedDeserializingTheModel;
  }
  if (!Deserialized->verify()) {
    dbg << "Model verification failed: '" << Options::Filename << "'.\n";
    return ExitCode::FailedVerifyingTheModel;
  }

  auto &Model = *Deserialized;
  return verifyABI(Model, Artifact, Options::TargetABI, *OutputStreamPtr);
}
