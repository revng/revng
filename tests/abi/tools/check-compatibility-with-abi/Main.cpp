/// This tool is used for ABI conversion testing.
///
/// It's purpose is to make sure that the function still corresponds to the
/// intended ABI no matter what kind of work was done on top of it. It works
/// with both `RawFunctionDefinition` and `CABIFunctionDefinition`.
///
/// \note: This tool requires a runtime abi artifact. It can be obtained using
/// revng-qa. For more information see the corresponding repository.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <fstream>
#include <optional>
#include <streambuf>
#include <string>

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Signals.h"

#include "revng/Model/Binary.h"
#include "revng/Model/TypeDefinitionKind.h"
#include "revng/Support/Debug.h"
#include "revng/Support/InitRevng.h"
#include "revng/Support/YAMLTraits.h"

#include "ABIRuntimeTestResultParser.h"
#include "ExpectedFailures.h"
#include "Verify.h"

namespace Options {

using namespace llvm::cl;

static OptionCategory ThisToolCategory("Tool options", "");

template<size_t... Indices>
static auto packValues(std::integer_sequence<size_t, Indices...>) {
  using namespace llvm::cl;
  using namespace model::ABI;
  return values(OptionEnumValue{ getName(Values(Indices + 1)),
                                 int(Indices + 1),
                                 getDescription(Values(Indices + 1)) }...);
}

constexpr auto ABIList = std::make_index_sequence<model::ABI::Count - 1>{};
constexpr const char *Description = "Specifies the default ABI of the binary.";
static opt<model::ABI::Values> TargetABI("abi",
                                         packValues(ABIList),
                                         desc(Description),
                                         cat(ThisToolCategory));

constexpr const char *VerifyTypeDescription = "Require every `test_` function "
                                              "of the binary to use a "
                                              "prototype "
                                              "of this kind.";
namespace TDK = model::TypeDefinitionKind;
static constexpr auto CABIFunctionDefinition = TDK::CABIFunctionDefinition;
static constexpr auto RawFunctionDefinition = TDK::RawFunctionDefinition;
static opt<model::TypeDefinitionKind::Values> // formatting
  VerifyType("verify-type",
             desc(VerifyTypeDescription),
             values(clEnumVal(CABIFunctionDefinition,
                              "Verify that every test function is a cabi "
                              "function."),
                    clEnumVal(RawFunctionDefinition,
                              "Verify that every test function is a raw "
                              "function.")),
             cat(ThisToolCategory));
static std::optional<model::TypeDefinitionKind::Values> readVerifyType() {
  if (VerifyType.getNumOccurrences() == 0)
    return std::nullopt;

  return VerifyType;
}

static opt<std::string> Filename(Positional,
                                 Required,
                                 desc("<input file name>"),
                                 cat(ThisToolCategory));

static opt<std::string> Artifact(Positional,
                                 Required,
                                 desc("<runtime abi artifact name>"),
                                 cat(ThisToolCategory));

static list<std::string> ExpectedFailures("expected-failures",
                                          Required,
                                          desc("<expected failure list>"),
                                          cat(ThisToolCategory));

constexpr const char *SaveLayoutsDescription = "Save the ABI layout of every "
                                               "tested function to the given "
                                               "YAML file.";
static opt<std::string> SaveLayouts("save-layouts",
                                    desc(SaveLayoutsDescription),
                                    value_desc("<output file name>"),
                                    cat(ThisToolCategory));

} // namespace Options

int main(int argc, char *argv[]) {
  revng::InitRevng X(argc, argv, "", { &Options::ThisToolCategory });

  auto InputOrError = llvm::MemoryBuffer::getFileOrSTDIN(Options::Filename);
  if (!InputOrError) {
    std::string Error = "Unable to open an input file: `" + Options::Filename
                        + "`.";
    revng_abort(Error.c_str());
  }
  llvm::StringRef InputText = (*InputOrError)->getBuffer();

  auto ArtifactOrError = llvm::MemoryBuffer::getFileOrSTDIN(Options::Artifact);
  if (!ArtifactOrError) {
    std::string Error = "Unable to open the artifact file: `"
                        + Options::Artifact + "`.";
    revng_abort(Error.c_str());
  }
  llvm::StringRef Artifact = (*ArtifactOrError)->getBuffer();
  if (Artifact.empty()) {
    std::string Error = "The artifact is empty: `" + Options::Artifact + "`.";
    revng_abort(Error.c_str());
  }

  auto Deserialized = TupleTree<model::Binary>::fromString(InputText);
  if (not Deserialized) {
    std::string Error = "Unable to deserialize the model in \""
                        + Options::Filename
                        + "\": " + consumeToString(Deserialized) + ".";
    revng_abort(Error.c_str());
  }

  if (!Deserialized->verify() || !(*Deserialized)->verify()) {
    std::string Error = "Unable to verify the model: `" + Options::Filename
                        + "`.";
    revng_abort(Error.c_str());
  }

  std::optional<std::vector<FunctionLayoutEntry>> Layouts;
  if (Options::SaveLayouts.getNumOccurrences() != 0)
    Layouts.emplace();

  auto Failures = verifyABI(*Deserialized,
                            Artifact,
                            Options::TargetABI,
                            Options::readVerifyType(),
                            Layouts);

  if (Layouts) {
    auto Error = serializeToFile(*Layouts, Options::SaveLayouts);
    if (Error) {
      std::string Message = "Unable to save the layouts to `"
                            + Options::SaveLayouts
                            + "`: " + consumeToString(std::move(Error)) + ".";
      revng_abort(Message.c_str());
    }
  }

  llvm::StringRef ABIName = model::ABI::getName(Options::TargetABI);
  const std::vector<std::string> &ExpectedFailures = Options::ExpectedFailures;
  auto Expectations = loadExpectedFailures(ExpectedFailures, ABIName);

  auto Filtered = filterFailures(Failures, Expectations);
  for (const Failure &Failure : Filtered)
    dbg << "- " << Failure.Test << ": " << Failure.Reason << "\n";

  return static_cast<int>(Filtered.size());
}
