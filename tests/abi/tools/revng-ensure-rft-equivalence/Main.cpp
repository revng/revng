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
ensureIDMatch(const model::Type::Key &Left,
              const model::Type::Key &Right,
              llvm::StringRef ExpectedName,
              model::Binary &Model) {
  auto &[LeftID, LeftKind] = Left;
  auto &[RightID, RightKind] = Right;
  revng_assert(LeftKind == RightKind);

  if (LeftID != RightID) {
    // Find the type
    model::TypePath OldPath = Model.getTypePath(Right);
    auto LeftIterator = Model.Types().find(Left);
    auto RightIterator = Model.Types().find(Right);

    // Make sure the replacement ID is not in use already.
    if (LeftIterator != Model.Types().end()) {
      if (ExpectedName == LeftIterator->get()->OriginalName()) {
        // Original names match: assume the types are the same and skip
        // the replacement.
        return std::nullopt;
      }

      std::string Error = "Key in use already: "
                          + serializeToString(LeftIterator->get()->key())
                          + "\nLHS name is '" + ExpectedName.str()
                          + "' while RHS name is '"
                          + LeftIterator->get()->OriginalName() + "'.";
      revng_abort(Error.c_str());
    }

    // Since the ID is a part of the key, we cannot just modify the type,
    // we need to move the type out, and then reinsert it back in.
    model::UpcastableType MovedOut = std::move(*RightIterator);
    Model.Types().erase(RightIterator);

    // Replace the ID of the moved out type.
    MovedOut->ID() = LeftID;
    model::TypePath NewPath = Model.getTypePath(MovedOut->key());

    // Reinsert the type.
    auto [_, Success] = Model.Types().insert(std::move(MovedOut));
    revng_assert(Success);

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

  using Model = TupleTree<model::Binary>;
  auto LeftModule = llvm::errorOrToExpected(Model::fromFile(LeftModelPath));
  if (not LeftModule)
    ExitOnError(LeftModule.takeError());
  TupleTree<model::Binary> &LeftModel = *LeftModule;

  auto RightModule = llvm::errorOrToExpected(Model::fromFile(RightModelPath));
  if (not RightModule)
    ExitOnError(RightModule.takeError());
  TupleTree<model::Binary> &RightModel = *RightModule;

  std::error_code EC;
  llvm::ToolOutputFile OutputFile(OutputFilename,
                                  EC,
                                  sys::fs::OpenFlags::OF_Text);
  if (EC)
    ExitOnError(llvm::createStringError(EC, EC.message()));

  // Introduce a function pair container to allow grouping them based on their
  // `CustomName`.
  struct KeyPair {
    std::optional<model::Type::Key> Left = std::nullopt;
    std::optional<model::Type::Key> Right = std::nullopt;
    bool operator<(const KeyPair &Another) const {
      return Left < Another.Left && Right < Another.Right;
    }
  };
  struct TransparentComparator {
    using is_transparent = std::true_type;

    bool operator()(llvm::StringRef LHS, llvm::StringRef RHS) const {
      return LHS < RHS;
    }
  };
  std::map<model::Identifier, KeyPair, TransparentComparator> Functions;
  std::unordered_set<uint64_t> FunctionIDLookup;

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
    if (Left->ID() == DefaultPrototype.ID())
      continue; // Skip the default prototype.

    auto [Iterator, Success] = Functions.try_emplace(LF.name());
    revng_assert(Success);
    Iterator->second.Left = Left->key();
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
    revng_assert(Iterator->second.Right == std::nullopt);
    Iterator->second.Right = Right->key();
  }

  // Deduplicate the list of IDs to replace.
  // We have to do this because in case a single prototype is used for multiple
  // functions (common in binaries lifted from PE/COFF) we still only want to
  // "replace" it once.
  std::set<KeyPair> DeduplicationHelper;
  for (auto Iterator = Functions.begin(); Iterator != Functions.end();) {
    if (Iterator->second.Left == std::nullopt) {
      std::string Error = "This should never happen, something is VERY wrong. "
                          "A function is missing in the left model: "
                          + Iterator->first.str().str() + "?";
    }
    if (Iterator->second.Right == std::nullopt) {
      std::string Error = "A function present in the left model is missing in "
                          "the right one: "
                          + Iterator->first.str().str();
      revng_abort(Error.c_str());
    }

    auto [_, Success] = DeduplicationHelper.emplace(Iterator->second);
    if (!Success)
      Iterator = Functions.erase(Iterator);
    else
      ++Iterator;
  }

  // Ensure both the functions themselves and their stack argument structs have
  // the same IDs. This makes it possible to use simple diff for comparing two
  // models instead of doing that manually, since the ID is the only piece
  // of the types that is allowed to change.
  std::map<model::TypePath, model::TypePath> RightReplacements;
  for (auto [Name, Pair] : Functions) {
    auto [LeftKey, RightKey] = Pair;

    auto LeftIterator = LeftModel->Types().find(LeftKey.value());
    revng_assert(LeftIterator != LeftModel->Types().end());
    auto *Left = llvm::cast<model::RawFunctionType>(LeftIterator->get());

    auto RightIterator = RightModel->Types().find(RightKey.value());
    revng_assert(RightIterator != RightModel->Types().end());
    auto *Right = llvm::cast<model::RawFunctionType>(RightIterator->get());

    revng_check(Left->Kind() == Right->Kind());

    // Try and access the argument struct.
    auto *LeftStack = Left->StackArgumentsType().get();
    auto *RightStack = Right->StackArgumentsType().get();

    // XOR the `bool`eans - make sure that either both functions have stack
    // argument or neither one does.
    revng_check(!LeftStack == !RightStack);
    if (LeftStack != nullptr) {
      model::Type::Key LeftSKey = LeftStack->key();
      model::Type::Key RightSKey = RightStack->key();
      llvm::StringRef ExpName = LeftModel->Types().at(LeftSKey)->OriginalName();
      if (auto R = ensureIDMatch(LeftSKey, RightSKey, ExpName, *RightModel))
        RightReplacements.emplace(std::move(R.value()));
    }

    llvm::StringRef ExpName = LeftModel->Types().at(*LeftKey)->OriginalName();
    if (auto R = ensureIDMatch(*LeftKey, *RightKey, ExpName, *RightModel))
      RightReplacements.emplace(std::move(R.value()));
  }

  // Replace references to modified types.
  RightModel.replaceReferences(RightReplacements);

  // Remove all the dynamic functions so that they don't interfere
  LeftModel->ImportedDynamicFunctions().clear();
  RightModel->ImportedDynamicFunctions().clear();

  // Remove all the functions that are not strictly related to the test.
  auto IsUnrelatedToTheTest = [](const model::Function &Function) {
    llvm::StringRef Name = Function.OriginalName();
    if (Name.take_front(5) == "test_")
      return false;
    if (Name.take_front(6) == "setup_")
      return false;
    if (Name == "main")
      return false;

    return true;
  };
  llvm::erase_if(LeftModel->Functions(), IsUnrelatedToTheTest);
  llvm::erase_if(RightModel->Functions(), IsUnrelatedToTheTest);

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
