// \file Main.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Pipes/ToolCLOptions.h"
#include "revng/Support/CommandLine.h"
#include "revng/Support/InitRevng.h"

static revng::pipes::ToolCLOptions BaseOptions(MainCategory);

static llvm::ExitOnError AbortOnError;

static std::string Overview = "This command will initialize the specified "
                              "directory in the same manner as running "
                              "\"artifact\" or \"analyze\" on it but without "
                              "doing any operation and returning immediately.";

int main(int argc, char *argv[]) {
  revng::InitRevng X(argc, argv, Overview.c_str(), { &MainCategory });
  pipeline::Registry::runAllInitializationRoutines();
  auto Manager = AbortOnError(BaseOptions.makeManager());
  return EXIT_SUCCESS;
}
