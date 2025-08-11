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

namespace cl = llvm::cl;

class StringPositionalArgument : public cl::opt<std::string> {
public:
  StringPositionalArgument(const char *Description) :
    cl::opt<std::string>(cl::Positional,
                         cl::Required,
                         cl::desc(Description),
                         cl::cat(MainCategory)) {}
};

static cl::opt<unsigned> DependencyLevel("dependency-level",
                                         cl::desc("Resolve dependencies of the "
                                                  "depending libraries as "
                                                  "well."),
                                         cl::cat(MainCategory),
                                         cl::init(1));

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
