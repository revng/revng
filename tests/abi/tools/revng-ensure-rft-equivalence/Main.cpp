//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

/// \file Main.cpp
/// \brief This tool is a specialized `revng model diff` extension designed
/// to work around the shortcomings of the abi testing pipeline.
///
/// It takes two models and produces a diff (and a non-zero exit code) if
/// they are not equivalent.
///
/// The main differences from `diff` is the fact that it invokes
/// `model::purgeUnnamedAndUnreachableTypes` on both models and also
/// specifically handles the IDs of `RawFunctionType`'s stack arguments.

#include <string>

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/Model/Pass/PurgeUnnamedAndUnreachableTypes.h"
#include "revng/Model/ToolHelpers.h"
#include "revng/Support/InitRevng.h"
#include "revng/Support/MetaAddress/YAMLTraits.h"
#include "revng/Support/YAMLTraits.h"
#include "revng/TupleTree/TupleTree.h"
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
  revng::InitRevng X(Argc, Argv);

  cl::HideUnrelatedOptions({ &ThisToolCategory });
  cl::ParseCommandLineOptions(Argc, Argv);

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

  // Gather `RawFunctionType`s present in both the models based on their IDs.
  struct FunctionPair {
    model::RawFunctionType *Left;
    model::RawFunctionType *Right;
  };
  std::map<model::Type::IDType, FunctionPair> Functions;
  for (model::UpcastableType &LeftType : LeftModel->Model->Types())
    if (auto *Left = llvm::dyn_cast<model::RawFunctionType>(LeftType.get()))
      Functions[Left->ID()].Left = Left;
  for (model::UpcastableType &RightType : RightModel->Model->Types())
    if (auto *Right = llvm::dyn_cast<model::RawFunctionType>(RightType.get()))
      Functions[Right->ID()].Right = Right;

  // Ensure their stack arguments have the same ID.
  //
  // If two `RawFunctionType`s have the same ID but their stack argument structs
  // do not - replace the ID of the "Right" struct to be that of the "Left" one.
  std::map<model::TypePath, model::TypePath> Replacements;
  for (auto [_, Pair] : Functions) {
    auto [Left, Right] = Pair;

    // Try and access the argument struct.
    revng_check(Left->StackArgumentsType().Qualifiers().empty());
    revng_check(Right->StackArgumentsType().Qualifiers().empty());
    model::Type
      *LeftStackArguments = Left->StackArgumentsType().UnqualifiedType().get();
    model::Type *
      RightStackArguments = Right->StackArgumentsType().UnqualifiedType().get();

    // XOR the `bool`eans - make sure that either both functions have stack
    // argument or neither one does.
    revng_check(!LeftStackArguments == !RightStackArguments);

    // Ignore function pairs without stack arguments.
    if (LeftStackArguments == nullptr)
      continue;

    // If IDs differ - replace the ID.
    if (LeftStackArguments->ID() != RightStackArguments->ID()) {
      model::TypePath FromPath = Right->StackArgumentsType().UnqualifiedType();

      RightModel->Model->Types().erase(LeftStackArguments->key());
      auto *Struct = llvm::dyn_cast<model::StructType>(RightStackArguments);
      revng_check(Struct != nullptr);
      auto Copy = model::UpcastableType::make<model::StructType>(*Struct);
      Copy->ID() = LeftStackArguments->ID();
      auto ToPath = RightModel->Model->recordNewType(std::move(Copy));

      Replacements.emplace(FromPath, ToPath);
    }
  }

  // Replace references to modified types.
  RightModel->Model.replaceReferences(Replacements);

  // Streamline both models and diff them
  model::purgeUnnamedAndUnreachableTypes(LeftModel->Model);
  model::purgeUnnamedAndUnreachableTypes(RightModel->Model);
  auto Diff = diff(*LeftModel->Model, *RightModel->Model);

  if (Diff.Changes.empty()) {
    return EXIT_SUCCESS;
  } else {
    Diff.dump(OutputFile.os());
    return EXIT_FAILURE;
  }
}
