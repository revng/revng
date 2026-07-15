//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/PluginLoader.h"
#include "llvm/Support/YAMLTraits.h"

#include "revng/PipeboxCommon/Helpers/Native/CLI/ContainerHandler.h"
#include "revng/PipeboxCommon/Helpers/Native/CLI/Helpers.h"
#include "revng/PipeboxCommon/Helpers/Native/Registry.h"
#include "revng/Support/InitRevng.h"

static llvm::ExitOnError AbortOnError;

namespace Options {
using namespace llvm::cl;
using OptString = opt<std::string>;

static OptionCategory RunPipeCategory("pipe run options");

static OptString FileStorage("file-storage",
                             desc("Unsupported, present for compatibility"),
                             cat(RunPipeCategory));

static OptString Dependencies("dependencies",
                              desc("dependencies tar output"),
                              cat(RunPipeCategory));

static OptString StaticConfiguration("static-configuration",
                                     desc("path to the static configuration "
                                          "file"),
                                     cat(RunPipeCategory));

static OptString Configuration("configuration",
                               desc("path to the dynamic configuration file"),
                               cat(RunPipeCategory));

static list<std::string> Objects("objects",
                                 ZeroOrMore,
                                 multi_val(2),
                                 desc("pair-wise container outgoing requests"),
                                 cat(RunPipeCategory));

static OptString Pipe(Positional, Required, desc("PIPE"), cat(RunPipeCategory));

static OptString
  Model(Positional, Required, desc("MODEL"), cat(RunPipeCategory));

static list<std::string> Arguments(Positional,
                                   ZeroOrMore,
                                   desc("CONTAINER_FILES..."),
                                   cat(RunPipeCategory));

} // namespace Options

static void
serializeDependencies(const revng::pypeline::ObjectDependencies &Dependencies,
                      llvm::raw_ostream &OS) {
  bool Written = false;
  for (size_t Index = 0; Index < Dependencies.size(); Index++) {
    for (const auto &Entry : Dependencies[Index]) {
      std::string SerializedObject = std::get<0>(Entry).serialize();
      OS << "- [" << Index << ", \"" << llvm::yaml::escape(SerializedObject)
         << "\", \"" << llvm::yaml::escape(std::get<1>(Entry)) << "\"]\n";
      Written = true;
    }
  }

  // Handle the case where the Dependencies object is empty, write an empty
  // array
  if (not Written)
    OS << "[]";
}

int main(int Argc, char *Argv[]) {
  using namespace revng::pypeline;
  using namespace revng::pypeline::helpers::native;
  using cli::configurationFromPath;

  revng::InitRevng X(Argc, Argv, "", { &Options::RunPipeCategory });

  // Create the pipe instance
  revng_assert(Registry.Pipes.count(Options::Pipe));
  std::string
    StaticConfiguration = configurationFromPath(Options::StaticConfiguration);
  std::unique_ptr<Pipe> ThePipe = Registry
                                    .Pipes[Options::Pipe](StaticConfiguration);
  auto Signature = ThePipe->signature();

  // Deserialize model
  Model TheModel = AbortOnError(cli::modelFromPath(Options::Model));

  // Create containers and deserialize them if needed
  cli::ContainerHandler ContainerHandler(Signature);
  AbortOnError(ContainerHandler.parseCommandline(Options::Arguments));
  AbortOnError(ContainerHandler.loadContainers());

  // Deserialize Outgoing requests
  revng_assert(Options::Objects.size() % 2 == 0);
  Request Incoming(Signature.size());
  Request Outgoing(Signature.size());
  // Since Request objects do not own the objects, store them in this vector
  std::vector<std::unique_ptr<ObjectID>> ObjectsPool;
  AbortOnError(cli::parseRequest(Signature,
                                 Options::Objects,
                                 ObjectsPool,
                                 Outgoing));

  // Get the vector of container pointers
  auto ContainersPointers = ContainerHandler.getContainers();

  // Enable caching
  TheModel.enableCaching();

  // Run the actual pipe
  auto Configuration = configurationFromPath(Options::Configuration);
  PipeOutput Output = ThePipe->run(TheModel,
                                   ContainersPointers,
                                   Incoming,
                                   Outgoing,
                                   Configuration);

  AbortOnError(ContainerHandler.storeContainers());

  // Store dependency output
  if (not Options::Dependencies.empty()) {
    std::error_code EC;
    llvm::raw_fd_ostream OS(Options::Dependencies, EC);
    if (EC)
      AbortOnError(llvm::errorCodeToError(EC));

    revng::TarWriter Writer(OS, revng::TarFormat::Plain);

    // Serialize normal dependencies as a yaml
    {
      std::string DependenciesString;
      {
        llvm::raw_string_ostream OS(DependenciesString);
        serializeDependencies(Output.Dependencies, OS);
      }
      Writer.addMember("dependencies.yml",
                       llvm::ArrayRef<char>{ &*DependenciesString.begin(),
                                             &*DependenciesString.end() });
    }

    // Serialize custom invalidation
    for (size_t Index = 0; Index < Output.CustomInvalidation.size(); Index++) {
      for (auto &Element : Output.CustomInvalidation[Index]) {
        std::string Name = std::to_string(Index)
                           + std::get<0>(Element).serialize();
        Writer.addMember(Name, std::get<1>(Element).data());
      }
    }
  }

  return EXIT_SUCCESS;
}
