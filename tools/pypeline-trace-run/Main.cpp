//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/PluginLoader.h"

#include "revng/Pypeline/Model.h"
#include "revng/Pypeline/TraceRunner/Runner.h"
#include "revng/Pypeline/TraceRunner/TraceFile.h"
#include "revng/Support/Assert.h"
#include "revng/Support/InitRevng.h"

static llvm::ExitOnError AbortOnError;

namespace Options {
using namespace llvm::cl;

static OptionCategory TraceRunToolCategory("Trace run options");

static opt<std::string> TraceFile(Positional,
                                  Required,
                                  cat(TraceRunToolCategory),
                                  desc("<input trace file>"));
static opt<std::string> ModelFile(Positional,
                                  Required,
                                  cat(TraceRunToolCategory),
                                  desc("<input model file>"));
static opt<std::string> StorageFile(Positional,
                                    Required,
                                    cat(TraceRunToolCategory),
                                    desc("<input sqlite3 storage>"));
} // namespace Options

int main(int Argc, const char *Argv[]) {
  using namespace revng::pypeline::tracerunner;
  using llvm::MemoryBuffer;

  revng::InitRevng X(Argc, Argv, "", {});

  TraceFile Trace = AbortOnError(TraceFile::fromFile(Options::TraceFile));

  auto ModelBuffer = revng::cantFail(MemoryBuffer::getFile(Options::ModelFile));
  Model TheModel;
  AbortOnError(TheModel.deserialize(ModelBuffer->getBuffer()));

  SavePoint SP(Options::StorageFile);

  Runner Runner;
  Runner.run(TheModel, Trace, SP);

  revng::pypeline::Buffer NewModel = TheModel.serialize();
  {
    std::error_code EC;
    llvm::raw_fd_stream OS(Options::ModelFile, EC);
    revng::cantFail(EC);
    OS << NewModel.get();
    OS.flush();
  }

  return EXIT_SUCCESS;
}
