//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

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
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Signals.h"

#include "revng/Support/InitRevng.h"

#include "ABIArtifactParser.h"
#include "Verify.h"

namespace Options {

using namespace llvm::cl;

static OptionCategory ThisToolCategory("Tool options", "");

template<std::size_t... Indices>
static auto packValues(std::integer_sequence<size_t, Indices...>) {
  using namespace llvm::cl;
  using namespace model::ABI;
  return values(OptionEnumValue{ getName(Values(Indices)),
                                 int(Indices),
                                 getDescription(Values(Indices)) }...);
}

constexpr auto ABIList = std::make_index_sequence<model::ABI::Count - 1>{};
constexpr const char *Description = "Specifies the default ABI of the binary.";
static opt<model::ABI::Values> TargetABI("abi",
                                         packValues(ABIList),
                                         desc(Description),
                                         cat(ThisToolCategory));

static opt<std::string> Filename(Positional,
                                 Required,
                                 desc("<input file name>"),
                                 cat(ThisToolCategory));

static opt<std::string> Artifact(Positional,
                                 Required,
                                 desc("<runtime abi artifact name>"),
                                 cat(ThisToolCategory));

} // namespace Options

static llvm::Error impl() {
  auto InputOrError = llvm::MemoryBuffer::getFileOrSTDIN(Options::Filename);
  if (!InputOrError)
    return ERROR(ExitCode::FailedOpeningTheInputFile,
                 "Unable to open an input file: '" + Options::Filename
                   + "'.\n");
  llvm::StringRef InputText = (*InputOrError)->getBuffer();

  auto ArtifactOrError = llvm::MemoryBuffer::getFileOrSTDIN(Options::Artifact);
  if (!ArtifactOrError)
    return ERROR(ExitCode::FailedOpeningTheArtifactFile,
                 "Unable to open the artifact file: '" + Options::Artifact
                   + "'.\n");
  llvm::StringRef Artifact = (*ArtifactOrError)->getBuffer();

  auto Deserialized = TupleTree<model::Binary>::deserialize(InputText);
  if (!Deserialized)
    return ERROR(ExitCode::FailedDeserializingTheModel,
                 "Unable to deserialize the model: '" + Options::Filename
                   + "'.\n");
  if (!Deserialized->verify() || !(*Deserialized)->verify())
    return ERROR(ExitCode::FailedVerifyingTheModel,
                 "Unable to verify the model: '" + Options::Filename + "'.\n");

  auto &Model = *Deserialized;
  return verifyABI(Model, Artifact, Options::TargetABI);
}

int main(int argc, const char *argv[]) {
  revng::InitRevng X(argc, argv);

  llvm::cl::HideUnrelatedOptions(Options::ThisToolCategory);
  llvm::cl::ParseCommandLineOptions(argc, argv);

  int ErrorCode = EXIT_SUCCESS;
  auto StringErrorHandler = [&ErrorCode](const llvm::StringError &Error) {
    dbg << "\nABI compatibility error: " << Error.getMessage() << '\n';
    ErrorCode = Error.convertToErrorCode().value();
  };
  llvm::handleAllErrors(impl(), StringErrorHandler);
  return ErrorCode;
}
