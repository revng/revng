/// \file Main.cpp
/// This tool is a specialized `revng model diff` extension designed to work
/// around the shortcomings of the abi testing pipeline.
///
/// It takes two models and produces a diff (and a non-zero exit code) if
/// they are not equivalent.
///
/// The main differences from `diff` is the fact that it invokes
/// `model::purgeUnnamedAndUnreachableTypes` on both models and also
/// specifically handles the IDs of `RawFunctionType`'s stack arguments.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <string>
#include <unordered_set>

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

static std::optional<std::pair<model::TypePath, model::TypePath>>
ensureMatchingIDs(const model::Type::Key &Left,
                  const model::Type::Key &Right,
                  model::Binary &Model) {
  auto &[LeftID, LeftKind] = Left;
  auto &[RightID, RightKind] = Right;
  revng_assert(LeftKind == RightKind);

  if (LeftID != RightID) {
    // Find the type
    model::TypePath OldPath = Model.getTypePath(Right);
    auto Iterator = Model.Types().find(OldPath.get()->key());

    // Since the ID is a part of the key, we cannot just modify the type,
    // we need to move the type out, and then reinsert it back in.
    model::UpcastableType MovedOut = std::move(*Iterator);
    Model.Types().erase(Iterator);

    // Replace the ID of the moved out type.
    MovedOut->ID() = LeftID;

    // Reinsert the type
    auto [It, Success] = Model.Types().insert(std::move(MovedOut));
    revng_assert(Success);
    model::TypePath NewPath = Model.getTypePath(It->get());

    // Return the "replacement pair", so that the caller is able to gather them
    // and invoke `TupleTree::replaceReferences` on all of them at once.
    return std::make_pair(OldPath, NewPath);
  } else {
    return std::nullopt;
  }
}

int main(int Argc, char *Argv[]) {
  revng::InitRevng X(Argc, Argv, "", { &ThisToolCategory });

  ExitOnError ExitOnError;

  auto LeftModule = ModelInModule::load(LeftModelPath);
  if (not LeftModule)
    ExitOnError(LeftModule.takeError());
  TupleTree<model::Binary> &LeftModel = LeftModule->Model;

  auto RightModule = ModelInModule::load(RightModelPath);
  if (not RightModule)
    ExitOnError(RightModule.takeError());
  TupleTree<model::Binary> &RightModel = RightModule->Model;

  std::error_code EC;
  llvm::ToolOutputFile OutputFile(OutputFilename,
                                  EC,
                                  sys::fs::OpenFlags::OF_Text);
  if (EC)
    ExitOnError(llvm::createStringError(EC, EC.message()));

  // Introduce a function pair container to allow grouping them based on their
  // `CustomName`.
  struct FunctionPair {
    model::RawFunctionType *Left = nullptr;
    model::RawFunctionType *Right = nullptr;
  };
  struct TransparentComparator {
    using is_transparent = std::true_type;

    bool operator()(llvm::StringRef LHS, llvm::StringRef RHS) const {
      return LHS < RHS;
    }
  };
  std::map<model::Identifier, FunctionPair, TransparentComparator> Functions;
  std::unordered_set<std::size_t> FunctionIDLookup;

  // Make sure the default prototype is valid.
  revng_assert(LeftModel->DefaultPrototype().isValid());
  const auto &DefaultPrototype = *LeftModel->DefaultPrototype().get();

  // Gather all the `RawFunctionType`s prototypes present in the first model.
  for (model::Function &LF : LeftModel->Functions()) {
    if (LF.Prototype().empty())
      continue; // Skip functions without prototypes.

    revng_assert(LF.name() != "",
                 "This test uses names to differentiate functions, as such "
                 "having unnamed functions in the model would break it, "
                 "hence it's not allowed.");

    auto *Left = llvm::cast<model::RawFunctionType>(LF.Prototype().get());
    FunctionIDLookup.emplace(Left->ID());

    if (Left->ID() == DefaultPrototype.ID())
      continue; // Skip the default prototype.

    auto [Iterator, Success] = Functions.try_emplace(LF.name());
    revng_assert(Success);
    Iterator->second.Left = Left;
  }

  // Gather all the `RawFunctionType`s prototypes present in the second model.
  for (model::Function &RF : RightModel->Functions()) {
    if (RF.Prototype().empty())
      continue; // Skip functions without prototypes.

    revng_assert(RF.name() != "",
                 "This test uses names to differentiate functions, as such "
                 "having unnamed functions in the model would break it, "
                 "hence it's not allowed.");

    auto *Right = llvm::cast<model::RawFunctionType>(RF.Prototype().get());

    if (Right->ID() == DefaultPrototype.ID())
      continue; // Skip the default prototype.

    auto Iterator = Functions.find(RF.name());
    if (Iterator == Functions.end()) {
      std::string Error = "A function present in the right model is missing in "
                          "the left one: "
                          + RF.name().str().str();
      revng_abort(Error.c_str());
    }
    revng_assert(Iterator->second.Right == nullptr);
    Iterator->second.Right = Right;
  }

  // Ensure both the functions themselves and their stack argument structs have
  // the same IDs. This makes it possible to use simple diff for comparing two
  // models instead of doing that manually, since the ID is the only piece
  // of the types that is allowed to change.
  std::map<model::TypePath, model::TypePath> LeftReplacements;
  std::map<model::TypePath, model::TypePath> RightReplacements;
  for (auto [Name, Pair] : Functions) {
    auto [Left, Right] = Pair;

    revng_assert(Left != nullptr,
                 "This should never happen, something is VERY wrong.");
    if (Right == nullptr) {
      std::string Error = "A function present in the left model is missing in "
                          "the right one: "
                          + Name.str().str();
      revng_abort(Error.c_str());
    }

    // Try and access the argument struct.
    auto *LeftStack = Left->StackArgumentsType().get();
    auto *RightStack = Right->StackArgumentsType().get();

    // XOR the `bool`eans - make sure that either both functions have stack
    // argument or neither one does.
    revng_check(!LeftStack == !RightStack);
    if (LeftStack != nullptr) {
      model::Binary &RM = *RightModel;
      if (auto R = ensureMatchingIDs(LeftStack->key(), RightStack->key(), RM))
        RightReplacements.emplace(std::move(R.value()));
    }

    if (auto R = ensureMatchingIDs(Left->key(), Right->key(), *RightModel))
      RightReplacements.emplace(std::move(R.value()));
  }

  // Replace references to modified types.
  LeftModel.replaceReferences(LeftReplacements);
  RightModel.replaceReferences(RightReplacements);

  // Remove all the dynamic functions so that they don't interfere
  LeftModel->ImportedDynamicFunctions().clear();
  RightModel->ImportedDynamicFunctions().clear();

  // Erase the default prototypes because they interfere with the test.
  LeftModel->Types().erase(LeftModel->DefaultPrototype().get()->key());
  RightModel->Types().erase(RightModel->DefaultPrototype().get()->key());
  LeftModel->DefaultPrototype() = model::TypePath{};
  RightModel->DefaultPrototype() = model::TypePath{};

  // Streamline both models and diff them
  model::purgeUnreachableTypes(LeftModel);
  model::purgeUnreachableTypes(RightModel);
  TupleTreeDiff Diff = diff(*LeftModel, *RightModel);

  if (Diff.Changes.empty()) {
    return EXIT_SUCCESS;
  } else {
    Diff.dump(OutputFile.os());
    return EXIT_FAILURE;
  }
}
