//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <optional>
#include <string>
#include <vector>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Error.h"

#include "revng/Clift/Clift.h"
#include "revng/Clift/Helpers.h"
#include "revng/EditCBody/EditByNameAnalysis.h"
#include "revng/Model/Binary.h"
#include "revng/Model/GotoLabel.h"
#include "revng/Model/LocalVariable.h"
#include "revng/Support/Debug.h"
#include "revng/Support/Error.h"
#include "revng/Support/MetaAddress.h"
#include "revng/Support/MetaAddress/YAMLTraits.h"
#include "revng/Support/YAMLTraits.h"

#include "ClangParse.h"
#include "ModelEdits.h"

using namespace llvm;
using namespace revng::editcbody;

static Logger Log("edit-by-name");

namespace {

/// One edit: the name of the entity to act on, and the new name, type and/or
/// comment to give it. An empty string means the directive was not written.
struct NameEdit {
  std::string Target;
  std::string Rename;
  std::string Retype;
  std::string Comment;
};

struct EditByNameConfiguration {
  MetaAddress Function;
  std::vector<NameEdit> Edits;
};

} // namespace

template<>
struct llvm::yaml::MappingTraits<NameEdit> {
  static void mapping(IO &IO, NameEdit &Fields) {
    IO.mapRequired("Target", Fields.Target);
    IO.mapOptional("Rename", Fields.Rename);
    IO.mapOptional("Retype", Fields.Retype);
    IO.mapOptional("Comment", Fields.Comment);
  }
};

LLVM_YAML_IS_SEQUENCE_VECTOR(NameEdit);

template<>
struct llvm::yaml::MappingTraits<EditByNameConfiguration> {
  static void mapping(IO &IO, EditByNameConfiguration &Fields) {
    IO.mapRequired("Function", Fields.Function);
    IO.mapRequired("Edits", Fields.Edits);
  }
};

/// Report an edit that could not be applied, and drop it.
///
/// Each edit stands on its own, so one that cannot be applied does not stop the
/// others: dropping it silently would leave the user wondering why their edit
/// had no effect, so it is reported on the `edit-by-name` logger, which
/// `--debug-log=edit-by-name` turns on.
static void
reportDropped(llvm::Error Error, StringRef Directive, StringRef Target) {
  // The error has to be consumed whether or not the logger is enabled.
  std::string Reason = consumeToString(std::move(Error));
  revng_log(Log,
            "Ignoring " << Directive << " on `" << Target << "`: " << Reason);
}

namespace {

/// The local variables and goto labels of a function, by the name the C backend
/// emits for them.
///
/// A name is what tells them apart in the decompiled code, so an edit naming
/// one that belongs to more than one of them names nothing in particular:
/// those are kept out of the maps and collected in \ref Shared instead.
struct NamedEntities {
  llvm::StringMap<clift::LocalVariableOp> Variables;
  llvm::StringMap<clift::MakeLabelOp> Labels;
  llvm::StringSet<> Shared;

  /// The names an edit can name, for the message reporting one that names
  /// something else.
  std::string describe() const {
    std::vector<llvm::StringRef> Names;
    for (const auto &Entry : Variables)
      Names.push_back(Entry.first());
    for (const auto &Entry : Labels)
      Names.push_back(Entry.first());
    llvm::sort(Names);

    if (Names.empty())
      return "the function has none";
    return "the function has " + llvm::join(Names, ", ");
  }
};

} // namespace

static NamedEntities collectNamedEntities(clift::FunctionOp Function) {
  NamedEntities Result;

  Function.walk([&](clift::LocalVariableOp Op) {
    if (not Result.Variables.try_emplace(Op.getName(), Op).second)
      Result.Shared.insert(Op.getName());
  });

  Function.walk([&](clift::MakeLabelOp Op) {
    if (not Result.Labels.try_emplace(Op.getName(), Op).second)
      Result.Shared.insert(Op.getName());
  });

  // A name a local variable and a goto label both go by is shared just the
  // same, whichever map each of them landed in.
  for (const auto &Entry : Result.Variables)
    if (Result.Labels.count(Entry.first()) != 0)
      Result.Shared.insert(Entry.first());

  for (llvm::StringRef Name : Result.Shared.keys()) {
    Result.Variables.erase(Name);
    Result.Labels.erase(Name);
  }

  return Result;
}

/// Collect the type named by each `Retype`, in order of first appearance and
/// without duplicates, so they can all be resolved by one Clang invocation.
static std::vector<std::string>
collectTypeNames(llvm::ArrayRef<NameEdit> Edits) {
  std::vector<std::string> Result;
  for (const NameEdit &Edit : Edits)
    if (not Edit.Retype.empty() and not llvm::is_contained(Result, Edit.Retype))
      Result.push_back(Edit.Retype);
  return Result;
}

namespace revng::pypeline::analyses {

llvm::Error EditByName::run(Model &Model,
                            const Request &Incoming,
                            llvm::StringRef Configuration,
                            const CliftFunctionContainer &Clift,
                            const PTMLCContainer &TypeAndGlobalHeader,
                            const PTMLCContainer &HelperHeader) {
  auto MaybeConfiguration = fromString<EditByNameConfiguration>(Configuration);
  if (not MaybeConfiguration)
    return MaybeConfiguration.takeError();

  EditByNameConfiguration &ParsedConfiguration = *MaybeConfiguration;

  model::Binary &Binary = *Model.get().get();

  MetaAddress Entry = ParsedConfiguration.Function;
  if (Entry.isInvalid()) {
    return revng::createError("the configuration does not specify a valid "
                              "function address");
  }

  auto FunctionIterator = Binary.Functions().find(Entry);
  if (FunctionIterator == Binary.Functions().end())
    return revng::createError("no function at address " + Entry.toString());

  ObjectID Object(Entry);
  mlir::ModuleOp FunctionModule = Clift.getModule(Object);
  clift::FunctionOp Function = clift::getUniqueIsolatedFunction(FunctionModule,
                                                                Entry);

  NamedEntities Entities = collectNamedEntities(Function);
  AmbiguousLocations Ambiguous = collectAmbiguousLocations(Function);

  // Resolve every type named by a `Retype` up front, so the header is written
  // and Clang is run once, whatever the number of edits.
  ResolvedTypeMap ResolvedTypes;
  std::vector<std::string> TypeNames = collectTypeNames(ParsedConfiguration
                                                          .Edits);
  if (not TypeNames.empty()) {
    auto MaybeHeader = writeHeader(TypeAndGlobalHeader, HelperHeader);
    if (not MaybeHeader)
      return MaybeHeader.takeError();

    auto MaybeTypes = resolveTypeNames(MaybeHeader->path(), TypeNames, Binary);
    if (not MaybeTypes)
      return MaybeTypes.takeError();

    ResolvedTypes = std::move(*MaybeTypes);
  }

  // Build the edits before touching the model. Nothing below can fail any more,
  // but keeping the two phases apart means the model is only ever written once
  // the whole configuration has been accounted for.
  std::vector<model::LocalVariable> NewVariables;
  std::vector<model::GotoLabel> NewLabels;

  for (const NameEdit &Edit : ParsedConfiguration.Edits) {
    std::optional<std::string> NewName;
    if (not Edit.Rename.empty())
      NewName = Edit.Rename;

    std::optional<std::string> NewTypeName;
    if (not Edit.Retype.empty())
      NewTypeName = Edit.Retype;

    std::optional<std::string> NewComment;
    if (not Edit.Comment.empty())
      NewComment = Edit.Comment;

    if (not NewName.has_value() and not NewTypeName.has_value()
        and not NewComment.has_value()) {
      reportDropped(revng::createError("it has none of `Rename`, `Retype` and "
                                       "`Comment`"),
                    "the edit",
                    Edit.Target);
      continue;
    }

    if (Entities.Shared.contains(Edit.Target)) {
      reportDropped(revng::createError("more than one local variable or goto "
                                       "label goes by that name, so it does "
                                       "not say which one to edit"),
                    "the edit",
                    Edit.Target);
      continue;
    }

    if (auto Iterator = Entities.Variables.find(Edit.Target);
        Iterator != Entities.Variables.end()) {
      auto MaybeVariable = makeLocalVariableEdit(Iterator->second,
                                                 NewName,
                                                 NewTypeName,
                                                 NewComment,
                                                 ResolvedTypes,
                                                 Ambiguous);
      if (MaybeVariable)
        NewVariables.push_back(std::move(*MaybeVariable));
      else
        reportDropped(MaybeVariable.takeError(), "the edit", Edit.Target);
      continue;
    }

    if (auto Iterator = Entities.Labels.find(Edit.Target);
        Iterator != Entities.Labels.end()) {
      // A label has a name and a comment but no type, so `Retype` does not
      // apply to it. Dropping the directive alone leaves a `Rename` or a
      // `Comment` on the same label working.
      if (NewTypeName.has_value()) {
        reportDropped(revng::createError("`Retype` cannot be applied to a "
                                         "label"),
                      "`Retype`",
                      Edit.Target);
      }

      if (not NewName.has_value() and not NewComment.has_value())
        continue;

      auto MaybeLabel = makeGotoLabelEdit(Iterator->second,
                                          NewName,
                                          NewComment,
                                          Ambiguous);
      if (MaybeLabel)
        NewLabels.push_back(std::move(*MaybeLabel));
      else
        reportDropped(MaybeLabel.takeError(), "the edit", Edit.Target);
      continue;
    }

    reportDropped(revng::createError("no local variable or goto label goes by "
                                     "that name; "
                                     + Entities.describe()),
                  "the edit",
                  Edit.Target);
  }

  model::Function &ModelFunction = *FunctionIterator;

  // Apply the variable edits, replacing any variable already located at the
  // same set of addresses.
  for (model::LocalVariable &Variable : NewVariables) {
    ModelFunction.LocalVariables().erase_if([&](const auto &Existing) {
      return Existing.Location() == Variable.Location();
    });
    ModelFunction.LocalVariables().insert(std::move(Variable));
  }

  // Apply the label edits, replacing any label already located at the same set
  // of addresses.
  for (model::GotoLabel &Label : NewLabels) {
    ModelFunction.GotoLabels().erase_if([&](const auto &Existing) {
      return Existing.Location() == Label.Location();
    });
    ModelFunction.GotoLabels().insert(std::move(Label));
  }

  return llvm::Error::success();
}

} // namespace revng::pypeline::analyses
