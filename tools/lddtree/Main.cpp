/// \file Main.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_os_ostream.h"

#include "revng/Support/CommandLine.h"
#include "revng/Support/InitRevng.h"
#include "revng/Support/LDDTree.h"

using namespace llvm;
using namespace llvm::cl;
using std::string;

class StringPositionalArgument : public opt<string> {
public:
  StringPositionalArgument(const char *Description) :
    opt<string>(Positional, Required, desc(Description), cat(MainCategory)) {}
};

static opt<unsigned> DependencyLevel("dependency-level",
                                     desc("Resolve dependencies of the "
                                          "depending libraries as well."),
                                     cat(MainCategory),
                                     init(1));

// TODO: Add --root option to act as SYSROOT.

StringPositionalArgument Input("Input binary");

int main(int argc, char *argv[]) {
  revng::InitRevng X(argc, argv, "", { &MainCategory });

  LDDTree Dependencies;
  lddtree(Dependencies, Input, DependencyLevel);
  for (auto &Library : Dependencies) {
    llvm::outs() << "Dependencies for " << Library.first << ":\n";
    for (auto &DependencyLibrary : Library.second)
      llvm::outs() << " " << DependencyLibrary << "\n";
  }

  return EXIT_SUCCESS;
}
