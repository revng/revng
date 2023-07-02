/// \file Main.cpp
/// This tool is used for ABI conversion testing.
///
/// It's purpose is to make sure that the function still corresponds to the
/// intended ABI no matter what kind of work was done on top of it. It works
/// with both `RawFunctionType` and `CABIFunctionType`.
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

#include "revng/Model/Binary.h"
#include "revng/Support/InitRevng.h"

#include "ABIArtifactParser.h"

void verifyABI(const TupleTree<model::Binary> &Binary,
               llvm::StringRef RuntimeArtifact,
               model::ABI::Values ABI);

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

int main(int argc, char *argv[]) {
  revng::InitRevng X(argc, argv, "", { &Options::ThisToolCategory });

  auto InputOrError = llvm::MemoryBuffer::getFileOrSTDIN(Options::Filename);
  if (!InputOrError) {
    std::string Error = "Unable to open an input file: '" + Options::Filename
                        + "'.";
    revng_abort(Error.c_str());
  }
  llvm::StringRef InputText = (*InputOrError)->getBuffer();

  auto ArtifactOrError = llvm::MemoryBuffer::getFileOrSTDIN(Options::Artifact);
  if (!ArtifactOrError) {
    std::string Error = "Unable to open the artifact file: '"
                        + Options::Artifact + "'.";
    revng_abort(Error.c_str());
  }
  llvm::StringRef Artifact = (*ArtifactOrError)->getBuffer();
  if (Artifact.empty()) {
    std::string Error = "The artifact is empty: '" + Options::Artifact + "'.";
    revng_abort(Error.c_str());
  }

  auto Deserialized = TupleTree<model::Binary>::deserialize(InputText);
  if (!Deserialized) {
    std::string Error = "Unable to deserialize the model: '" + Options::Filename
                        + "'.";
    revng_abort(Error.c_str());
  }
  if (!Deserialized->verify() || !(*Deserialized)->verify()) {
    std::string Error = "Unable to verify the model: '" + Options::Filename
                        + "'.";
    revng_abort(Error.c_str());
  }

  verifyABI(*Deserialized, Artifact, Options::TargetABI);
}
