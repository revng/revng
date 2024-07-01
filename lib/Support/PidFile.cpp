/// \file PidFile.cpp
/// \brief Command-line option to print the PID to a file.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Support/Process.h"

#include "revng/Support/Assert.h"
#include "revng/Support/CommandLine.h"

namespace {

struct PidFileParser : llvm::cl::parser<std::string> {
  using llvm::cl::parser<std::string>::parser;

  bool parse(llvm::cl::Option &O,
             llvm::StringRef ArgName,
             llvm::StringRef ArgValue,
             std::string &Val) {
    using llvm::sys::Process;

    if (not ArgValue.empty()) {
      std::error_code EC;
      llvm::raw_fd_ostream Output(ArgValue, EC);
      revng_assert(!EC);

      Process::Pid Pid = Process::getProcessId();
      Output << Pid;
      Output.close();
    }
    return false;
  }
};

using opt = llvm::cl::opt<std::string, false, PidFileParser>;
using llvm::cl::desc;

opt PidFile("pid-file",
            desc("If present, the currently-running PID will be written to the "
                 "specified file"),
            llvm::cl::cat(MainCategory));

}; // namespace
