//
// This file is distributed under the MIT License. See LICENSE.md for details.
//
// rcc-ignore: initrevng

#include <iostream>

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"

#include "revng/PipelineC/PipelineC.h"
#include "revng/PipelineC/Tracing/Trace.h"
#include "revng/Support/Assert.h"
#include "revng/Support/CommandLine.h"

using revng::tracing::Trace;
static llvm::ExitOnError AbortOnError;

namespace Options {
using namespace llvm::cl;

namespace detail {
using llvm::StringRef;

struct BoolString {
  bool Set = false;
  std::string String;
};

class BoolStringParser : public parser<BoolString> {
public:
  BoolStringParser(Option &O) : parser(O) {}

  bool parse(Option &O,
             StringRef ArgName,
             const StringRef ArgValue,
             BoolString &Val) {
    Val.Set = true;
    Val.String = ArgValue.str();
    return false;
  }

  enum ValueExpected getValueExpectedFlagDefault() const {
    return ValueOptional;
  }
};
} // namespace detail

using BoolStringOpt = opt<detail::BoolString, false, detail::BoolStringParser>;

static OptionCategory TraceRunToolCategory("Tool options", "");

static opt<std::string> TraceFile(Positional,
                                  Required,
                                  cat(TraceRunToolCategory),
                                  desc("<input trace file>"));

static BoolStringOpt TemporaryRoot("temporary-root",
                                   cat(TraceRunToolCategory),
                                   desc("Directory used to create temporary "
                                        "directories needed for execution"));
static opt<bool> SoftAsserts("soft-asserts",
                             init(false),
                             cat(TraceRunToolCategory),
                             desc("Turn some assertions into warnings"));
static llvm::cl::list<uint64_t> BreakAt("break-at",
                                        ZeroOrMore,
                                        cat(TraceRunToolCategory),
                                        desc("Command Indexes to break at"));

static alias SoftAssertsA("s",
                          desc("Alias for --soft-asserts"),
                          aliasopt(SoftAsserts),
                          cat(TraceRunToolCategory),
                          NotHidden);
static alias BreakAtA("b",
                      desc("Alias for --break-at"),
                      aliasopt(BreakAt),
                      NotHidden,
                      cat(TraceRunToolCategory));
static alias TemporaryRootA("t",
                            desc("Alias for --temporary-root"),
                            aliasopt(TemporaryRoot),
                            NotHidden,
                            cat(TraceRunToolCategory));

} // namespace Options

int main(int argc, const char *argv[]) {
  // NOLINTNEXTLINE
  llvm::cl::HideUnrelatedOptions(Options::TraceRunToolCategory);
  rp_initialize(argc, argv, 0, {});

  Trace TheTrace = AbortOnError(Trace::fromFile(Options::TraceFile));

  std::string TemporaryRoot;
  if (Options::TemporaryRoot.Set) {
    if (Options::TemporaryRoot.String.empty()) {
      llvm::SmallString<128> Output;
      llvm::sys::fs::createUniqueDirectory("revng-run-trace-root", Output);
      TemporaryRoot = Output.str().str();
      dbg << "Creating temporary root directory: " << TemporaryRoot << "\n";
    } else {
      TemporaryRoot = Options::TemporaryRoot.String;
    }
  }

  revng::tracing::RunTraceOptions Options = {
    .SoftAsserts = Options::SoftAsserts,
    .BreakAt = { Options::BreakAt.begin(), Options::BreakAt.end() },
    .TemporaryRoot = TemporaryRoot,
  };
  AbortOnError(TheTrace.run(Options));

  rp_shutdown();
  return EXIT_SUCCESS;
}
