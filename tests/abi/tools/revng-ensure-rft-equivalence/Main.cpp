/// \file Main.cpp
/// This tool is a specialized `revng model diff` extension designed to work
/// around the shortcomings of the abi testing pipeline.
///
/// It takes two models and produces a diff (and a non-zero exit code) if
/// they are not equivalent.
///
/// The main differences from `diff` is the fact that it invokes
/// `model::purgeUnnamedAndUnreachableTypes` on both models and also
/// specifically handles the IDs of `RawFunctionDefinition`'s stack arguments.

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

using DefinitionPair = std::pair<model::DefinitionReference,
                                 model::DefinitionReference>;
static std::optional<DefinitionPair>
ensureIDMatch(const model::TypeDefinition::Key &Left,
              const model::TypeDefinition::Key &Right,
              llvm::StringRef ExpectedName,
              model::Binary &Model) {
  auto &[LeftID, LeftKind] = Left;
  auto &[RightID, RightKind] = Right;
  revng_assert(LeftKind == RightKind);

  if (LeftID != RightID) {
    // Find the type
    model::DefinitionReference OldPath = Model.getDefinitionReference(Right);
    auto LeftIterator = Model.TypeDefinitions().find(Left);
    auto RightIterator = Model.TypeDefinitions().find(Right);

    // Make sure the replacement ID is not in use already.
    if (LeftIterator != Model.TypeDefinitions().end()) {
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
    model::UpcastableTypeDefinition MovedOut = std::move(*RightIterator);
    Model.TypeDefinitions().erase(RightIterator);

    // Replace the ID of the moved out type.
    MovedOut->ID() = LeftID;
    auto NewPath = Model.getDefinitionReference(MovedOut->key());

    // Reinsert the type.
    auto [_, Success] = Model.TypeDefinitions().insert(std::move(MovedOut));
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
    std::optional<model::TypeDefinition::Key> Left = std::nullopt;
    std::optional<model::TypeDefinition::Key> Right = std::nullopt;
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

  // Gather all the `RawFunctionDefinition` prototypes present in the first
  // model.
  for (model::Function &F : LeftModel->Functions()) {
    if (F.Prototype().empty())
      continue; // Skip functions without prototypes.

    revng_assert(F.name() != "",
                 "This test uses names to differentiate functions, as such "
                 "having unnamed functions in the model would break it, "
                 "hence it's not allowed.");

    if (const model::RawFunctionDefinition *Left = F.rawPrototype()) {
      if (Left->ID() == LeftModel->defaultPrototype()->ID())
        continue; // Skip the default prototype.

      auto [Iterator, Success] = Functions.try_emplace(F.name());
      revng_assert(Success);
      Iterator->second.Left = Left->key();
    }
  }

  // Gather all the `RawFunctionDefinition` prototypes present in the second
  // model.
  for (model::Function &F : RightModel->Functions()) {
    if (F.Prototype().empty())
      continue; // Skip functions without prototypes.

    revng_assert(F.name() != "",
                 "This test uses names to differentiate functions, as such "
                 "having unnamed functions in the model would break it, "
                 "hence it's not allowed.");

    if (const model::RawFunctionDefinition *Right = F.rawPrototype()) {
      if (Right->ID() == LeftModel->defaultPrototype()->ID())
        continue; // Skip the default prototype.

      auto Iterator = Functions.find(F.name());
      if (Iterator == Functions.end()) {
        std::string Error = "A function present in the right model is missing "
                            "in "
                            "the left one: "
                            + F.name().str().str();
        revng_abort(Error.c_str());
      }
      revng_assert(Iterator->second.Right == std::nullopt);
      Iterator->second.Right = Right->key();
    }
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
  std::map<model::DefinitionReference, model::DefinitionReference> Replacements;
  for (auto [Name, Pair] : Functions) {
    auto [LKey, RKey] = Pair;

    auto LeftIt = LeftModel->TypeDefinitions().find(LKey.value());
    revng_assert(LeftIt != LeftModel->TypeDefinitions().end());
    auto &Left = *llvm::cast<model::RawFunctionDefinition>(LeftIt->get());

    auto RightIt = RightModel->TypeDefinitions().find(RKey.value());
    revng_assert(RightIt != RightModel->TypeDefinitions().end());
    auto &Right = *llvm::cast<model::RawFunctionDefinition>(RightIt->get());

    revng_check(Left.Kind() == Right.Kind());

    // Try and access the argument struct.
    auto *LeftStack = Left.stackArgumentsType();
    auto *RightStack = Right.stackArgumentsType();

    // XOR the `bool`eans - make sure that either both functions have stack
    // argument or neither one does.
    revng_check(!LeftStack == !RightStack);
    if (LeftStack != nullptr) {
      const model::TypeDefinition::Key &LSK = LeftStack->key();
      const model::TypeDefinition::Key &RSK = RightStack->key();
      llvm::StringRef N = LeftModel->TypeDefinitions().at(LSK)->OriginalName();
      if (auto Replacement = ensureIDMatch(LSK, RSK, N, *RightModel))
        Replacements.emplace(std::move(Replacement.value()));
    }

    llvm::StringRef N = LeftModel->TypeDefinitions().at(*LKey)->OriginalName();
    if (auto Replacement = ensureIDMatch(*LKey, *RKey, N, *RightModel))
      Replacements.emplace(std::move(Replacement.value()));
  }

  // Replace references to modified types.
  RightModel.replaceReferences(Replacements);

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
  LeftModel->TypeDefinitions().erase(LeftModel->defaultPrototype()->key());
  RightModel->TypeDefinitions().erase(RightModel->defaultPrototype()->key());
  LeftModel->DefaultPrototype() = {};
  RightModel->DefaultPrototype() = {};

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
