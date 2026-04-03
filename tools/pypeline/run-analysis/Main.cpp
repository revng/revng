//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/PluginLoader.h"

#include "revng/PipeboxCommon/Helpers/Native/CLI/ContainerHandler.h"
#include "revng/PipeboxCommon/Helpers/Native/CLI/Helpers.h"
#include "revng/PipeboxCommon/Helpers/Native/Registry.h"
#include "revng/Support/InitRevng.h"

static llvm::ExitOnError AbortOnError;

namespace Options {
using namespace llvm::cl;
using OptString = opt<std::string>;

static OptionCategory RunAnalysisCategory("run-analysis options");

static OptString Configuration("configuration",
                               desc("dynamic configuration"),
                               cat(RunAnalysisCategory));

static list<std::string> Objects("objects",
                                 ZeroOrMore,
                                 multi_val(2),
                                 desc("pair-wise container outgoing requests"),
                                 cat(RunAnalysisCategory));

static OptString
  Output("o", desc("output model path"), cat(RunAnalysisCategory), init("-"));

static alias OutputA("output", desc("Alias for -o"), aliasopt(Output));

static OptString
  Analysis(Positional, Required, desc("ANALYSIS"), cat(RunAnalysisCategory));

static OptString
  Model(Positional, Required, desc("MODEL"), cat(RunAnalysisCategory));

static list<std::string> Arguments(Positional,
                                   ZeroOrMore,
                                   desc("CONTAINER_FILES..."),
                                   cat(RunAnalysisCategory));

} // namespace Options

int main(int Argc, char *Argv[]) {
  using namespace revng::pypeline;
  using namespace revng::pypeline::helpers::native;

  revng::InitRevng X(Argc, Argv, "", { &Options::RunAnalysisCategory });

  // Create the analysis instance
  revng_assert(Registry.Analyses.count(Options::Analysis) != 0);
  std::unique_ptr<Analysis> TheAnalysis = Registry
                                            .Analyses[Options::Analysis]();
  auto Signature = TheAnalysis->signature();

  // Deserialize model
  Model TheModel = AbortOnError(cli::modelFromPath(Options::Model));

  // Create containers and deserialize them if needed
  cli::ContainerHandler ContainerHandler(Signature);
  AbortOnError(ContainerHandler.parseCommandline(Options::Arguments));
  AbortOnError(ContainerHandler.loadContainers());

  // Deserialize Outgoing requests
  revng_assert(Options::Objects.size() % 2 == 0);
  Request Outgoing(Signature.size());
  // Since Request objects do not own the objects, store them in this vector
  std::vector<std::unique_ptr<ObjectID>> ObjectsPool;
  AbortOnError(cli::parseRequest(Signature,
                                 Options::Objects,
                                 ObjectsPool,
                                 Outgoing));

  // Get the vector of container pointers
  auto ContainersPointers = ContainerHandler.getContainers();

  // Disable caching
  TheModel.disableCaching();

  // Run the actual analysis
  AbortOnError(TheAnalysis->run(TheModel,
                                ContainersPointers,
                                Outgoing,
                                Options::Configuration));

  // Save modified model
  {
    std::error_code EC;
    llvm::raw_fd_ostream OS(Options::Output, EC);
    AbortOnError(llvm::errorCodeToError(EC));
    OS << TheModel.serialize();
  }

  return EXIT_SUCCESS;
}
