//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <iostream>

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ToolOutputFile.h"

#include "revng/PipelineC/Tracing/Trace.h"
#include "revng/Support/CommandLine.h"
#include "revng/Support/InitRevng.h"

using llvm::StringRef;

using revng::tracing::BufferLocation;
using revng::tracing::Trace;

namespace Options {
using namespace llvm::cl;
using std::string;

static OptionCategory ThisToolCategory("Tool options", "");

static opt<string> Input(Positional,
                         cat(ThisToolCategory),
                         desc("<input trace file>"),
                         value_desc("input trace file"));

static opt<bool> ListBuffers("list-buffers",
                             desc("List the buffers present on the trace"),
                             cat(ThisToolCategory),
                             init(false));

static opt<string> ExtractBuffer("extract-buffer",
                                 cat(ThisToolCategory),
                                 desc("Buffer to extract"),
                                 value_desc("COMMANDNO:ARGNO"));

static alias ExtractBufferA("e",
                            cat(ThisToolCategory),
                            desc("Alias for --extract-buffer"),
                            aliasopt(ExtractBuffer));

static opt<string> Output("output",
                          cat(ThisToolCategory),
                          desc("Output file when extracting"),
                          value_desc("file"));

static alias OutputA("o",
                     cat(ThisToolCategory),
                     desc("Alias for --output"),
                     aliasopt(Output));
} // namespace Options

static llvm::ExitOnError AbortOnError;

int main(int argc, char *argv[]) {
  revng::InitRevng X(argc, argv, "", { &Options::ThisToolCategory });

  if (Options::ListBuffers xor Options::ExtractBuffer.empty()) {
    dbg << "Please specify either --list-buffers or --extract-buffer\n";
    return EXIT_FAILURE;
  }

  if (Options::ListBuffers) {
    Trace TheTrace = AbortOnError(Trace::fromFile(Options::Input));
    std::vector<BufferLocation> Result = TheTrace.listBuffers();
    for (auto &Location : Result) {
      std::cout << "Buffer on command " << Location.CommandName << " (Command #"
                << Location.CommandNumber << "), argument #"
                << Location.ArgumentNumber << "\n";
    }
    return EXIT_SUCCESS;
  }

  if (!Options::ExtractBuffer.empty()) {
    if (Options::Output.empty()) {
      dbg << "Please specify an output file\n";
      return EXIT_FAILURE;
    }

    size_t CommandNo = 0;
    size_t ArgNo = 0;
    auto &&[CommandNoStr,
            ArgNoStr] = StringRef(Options::ExtractBuffer).split(":");
    if (CommandNoStr.getAsInteger(10, CommandNo)
        || ArgNoStr.getAsInteger(10, ArgNo)) {
      dbg << "Error parsing extract value\n";
      return EXIT_FAILURE;
    }

    Trace TheTrace = AbortOnError(Trace::fromFile(Options::Input));
    using Buffer = std::vector<char>;
    Buffer Result = AbortOnError(TheTrace.getBuffer(CommandNo, ArgNo));

    std::error_code EC;
    llvm::ToolOutputFile OutputFile(Options::Output,
                                    EC,
                                    llvm::sys::fs::OF_None);
    OutputFile.os() << llvm::StringRef(Result.data(), Result.size());
    OutputFile.keep();

    return EXIT_SUCCESS;
  }

  return EXIT_FAILURE;
}
