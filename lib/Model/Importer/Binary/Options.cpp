//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Support/CommandLine.h"

#include "revng/Model/Importer/Binary/Options.h"
#include "revng/Support/CommandLine.h"

using Enum = llvm::cl::OptionEnumValue;
using SR = llvm::StringRef;
namespace cl = llvm::cl;

constexpr SR DescBA = "Base address where dynamic objects should be loaded.";
cl::opt<uint64_t> BaseAddress("base",
                              cl::desc(DescBA),
                              cl::value_desc("address"),
                              cl::cat(MainCategory),
                              cl::init(0x400000));

constexpr SR DescLevel = "Controls the debug information processing when "
                         "importing a binary.";
constexpr Enum No = clEnumValN(DebugInfoLevel::No,
                               "no",
                               "Ignore debug information even if it's "
                               "present.");
constexpr Enum Yes = clEnumValN(DebugInfoLevel::Yes,
                                "yes",
                                "Load debug information from the input file "
                                "and the libraries it directly depends on.");
constexpr Enum NoLib = clEnumValN(DebugInfoLevel::IgnoreLibraries,
                                  "ignore-libraries",
                                  "Load debug information from the input file "
                                  "only.");
cl::opt<DebugInfoLevel> DebugInfo("debug-info",
                                  cl::desc(DescLevel),
                                  cl::value_desc("level"),
                                  cl::values(No, Yes, NoLib),
                                  cl::cat(MainCategory),
                                  cl::init(DebugInfoLevel::Yes));

constexpr SR DescRemote = "Allow fetching debug information from "
                          "canonical places or web.";
cl::opt<bool> EnableRemoteDebugInfo("enable-remote-debug-info",
                                    cl::desc(DescRemote),
                                    cl::cat(MainCategory),
                                    cl::init(false));

const ImporterOptions importerOptions() {
  return ImporterOptions{ .BaseAddress = BaseAddress,
                          .DebugInfo = DebugInfo,
                          .EnableRemoteDebugInfo = EnableRemoteDebugInfo };
}
