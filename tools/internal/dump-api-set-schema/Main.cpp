//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <cstdlib>

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/Support/CommandLine.h"
#include "revng/Support/InitRevng.h"
#include "revng/Support/WindowsApiSetSchemaParser.h"

using namespace llvm;

static cl::opt<std::string> Input(cl::Positional,
                                  cl::Required,
                                  cl::desc("APISETSCHEMA_DLL"),
                                  cl::cat(MainCategory));

int main(int argc, char *argv[]) {
  revng::InitRevng X(argc, argv, "", { &MainCategory });

  auto MaybeMap = parseWindowsApiSetSchemaFromDLL(Input);
  if (not MaybeMap) {
    errs() << "Failed to parse API set schema from " << Input << "\n";
    return EXIT_FAILURE;
  }

  if (MaybeMap->PrefixLength > 0)
    outs() << "PrefixLength: " << MaybeMap->PrefixLength << "\n";

  for (const auto &[Source, Target] : MaybeMap->Map)
    outs() << Source << " -> " << Target << "\n";

  return EXIT_SUCCESS;
}
