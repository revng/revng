/// \file Common.cpp
/// \brief

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <string>

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/Model/RawBinaryView.h"
#include "revng/Pipes/Kinds.h"
#include "revng/Support/MetaAddress/YAMLTraits.h"
#include "revng/Support/YAMLTraits.h"
#include "revng/TupleTree/TupleTree.h"

#include "Common.h"

namespace options {

using namespace llvm::cl;

OptionCategory ThisToolCategory("Tool options", "");

static opt<std::string> BinaryPath(llvm::cl::Positional,
                                   llvm::cl::Required,
                                   desc("<input binary>"),
                                   value_desc("binary"),
                                   cat(ThisToolCategory));

static opt<std::string> LLVMModulePath(llvm::cl::Positional,
                                       llvm::cl::Required,
                                       desc("<input llvm-module>"),
                                       value_desc("llvm-module"),
                                       cat(ThisToolCategory));

static opt<std::string> ModelPath(llvm::cl::Positional,
                                  desc("<input model>"),
                                  value_desc("model"),
                                  init(""),
                                  cat(ThisToolCategory));

} // namespace options

static revng::pipes::FunctionStringMap
createMap(const TupleTree<model::Binary> &Model) {
  using C = revng::pipes::FunctionStringMap;
  return C("", "", pipeline::Kind{ "", &revng::pipes::RootRank }, Model);
}

ReturnValueType parseCommandLineOptions(int Argc, char *Argv[]) {
  llvm::cl::HideUnrelatedOptions({ &options::ThisToolCategory });
  llvm::cl::ParseCommandLineOptions(Argc, Argv);

  llvm::ExitOnError ExitOnError;
  ObjectLifetimeController Result;

  Result.Context = std::make_unique<llvm::LLVMContext>();

  llvm::SMDiagnostic Diagnostic;
  Result.IR = llvm::parseIRFile(options::LLVMModulePath,
                                Diagnostic,
                                *Result.Context);
  revng_assert(Result.IR != nullptr);

  using ModelBinaryTree = TupleTree<model::Binary>;
  if (options::ModelPath.empty()) {
    revng_assert(hasModel(*Result.IR));
    Result.Model = std::make_unique<ModelBinaryTree>(loadModel(*Result.IR));
  } else {
    auto ErrorOrModel = ModelBinaryTree::fromFile(options::ModelPath);
    auto Model = ExitOnError(llvm::errorOrToExpected(std::move(ErrorOrModel)));
    Result.Model = std::make_unique<ModelBinaryTree>(std::move(Model));
  }
  revng_assert(Result.Model != nullptr);

  auto [BinView,
        Storage] = ExitOnError(loadBinary(**Result.Model, options::BinaryPath));
  Result.Binary = std::move(Storage);
  revng_assert(Result.Binary != nullptr);

  return ReturnValueType(*Result.IR,
                         *Result.Model,
                         BinView,
                         createMap(*Result.Model),
                         std::move(Result));
}
