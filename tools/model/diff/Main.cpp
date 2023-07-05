/// \file Main.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <string>

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/Model/ToolHelpers.h"
#include "revng/Support/InitRevng.h"
#include "revng/Support/MetaAddress/YAMLTraits.h"
#include "revng/Support/YAMLTraits.h"
#include "revng/TupleTree/TupleTreeDiff.h"

using namespace llvm;

static cl::OptionCategory ThisToolCategory("Tool options", "");

static cl::opt<std::string> LeftModelPath(cl::Positional,
                                          cl::cat(ThisToolCategory),
                                          cl::desc("<left input model>"),
                                          cl::init("-"),
                                          cl::value_desc("left model"));

static cl::opt<std::string> RightModelPath(cl::Positional,
                                           cl::cat(ThisToolCategory),
                                           cl::desc("<right input model>"),
                                           cl::value_desc("right model"));

static cl::opt<std::string> OutputFilename("o",
                                           cl::cat(ThisToolCategory),
                                           cl::init("-"),
                                           llvm::cl::desc("Override "
                                                          "output "
                                                          "filename"),
                                           llvm::cl::value_desc("filename"));

int main(int Argc, char *Argv[]) {
  revng::InitRevng X(Argc, Argv, "", { &ThisToolCategory });

  ExitOnError ExitOnError;

  auto LeftModel = ModelInModule::load(LeftModelPath);
  if (not LeftModel)
    ExitOnError(LeftModel.takeError());

  auto RightModel = ModelInModule::load(RightModelPath);
  if (not RightModel)
    ExitOnError(RightModel.takeError());

  std::error_code EC;
  llvm::ToolOutputFile OutputFile(OutputFilename,
                                  EC,
                                  sys::fs::OpenFlags::OF_Text);
  if (EC)
    ExitOnError(llvm::createStringError(EC, EC.message()));
  auto &Stream = OutputFile.os();

  auto Diff = diff(*LeftModel->Model, *RightModel->Model);
  Diff.dump(Stream);
  OutputFile.keep();

  if (Diff.Changes.empty())
    return EXIT_SUCCESS;
  else
    return EXIT_FAILURE;
}
