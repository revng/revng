//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/Clift/Clift.h"
#include "revng/Clift/CliftOpHelpers.h"
#include "revng/Clift/Helpers.h"
#include "revng/Clift/LocationAddresses.h"
#include "revng/EditCBody/EditCBodyAnalysis.h"
#include "revng/Model/Binary.h"
#include "revng/Model/GotoLabel.h"
#include "revng/Model/LocalVariable.h"
#include "revng/Model/StatementComment.h"
#include "revng/Ranks/Location.h"
#include "revng/Ranks/Ranks.h"
#include "revng/Support/Debug.h"
#include "revng/Support/MetaAddress.h"
#include "revng/Support/MetaAddress/YAMLTraits.h"
#include "revng/Support/TemporaryFile.h"
#include "revng/Support/YAMLTraits.h"

#include "ClangParse.h"
#include "CliftFlatten.h"
#include "Statements.h"

using namespace llvm;
using namespace revng::editcbody;

static Logger Log("edit-c-body");

namespace rr = revng::ranks;

/// The model type resolved for each `RETYPE:` directive, keyed by its text.
using ResolvedTypeMap = std::map<std::string, model::UpcastableType>;

//
// Header assembly
//

/// Write, to a temporary file, the type/global header and the helper header, so
/// that a single decompiled function definition can be parsed by Clang. Both
/// are the tagless headers the pipeline already produced, so nothing is
/// re-emitted here.
static llvm::Expected<TemporaryFile>
writeHeader(const revng::pypeline::PTMLCContainer &TypeAndGlobalHeader,
            const revng::pypeline::PTMLCContainer &HelperHeader) {
  ObjectID Root = ObjectID::root();
  if (not TypeAndGlobalHeader.contains(Root) or not HelperHeader.contains(Root))
    return revng::createError("the decompiler headers have not been produced");

  auto MaybeFile = TemporaryFile::make("import-comments-header", "h");
  if (not MaybeFile)
    return revng::createError("Could not create a temporary header file");

  std::error_code ErrorCode;
  llvm::raw_fd_ostream Output(MaybeFile->path(), ErrorCode);
  if (ErrorCode)
    return revng::createError("Could not open the temporary header file");

  Output << TypeAndGlobalHeader.getMemoryBuffer(Root)->getBuffer();
  Output << "\n";
  Output << HelperHeader.getMemoryBuffer(Root)->getBuffer();
  Output.flush();

  return std::move(*MaybeFile);
}

//
// Location ambiguity
//

namespace {

/// The address sets that identify more than one local variable, or more than
/// one goto label, of a function.
///
/// Both are identified by the addresses of the instructions using them, so two
/// of them sharing a set cannot be told apart: a model entry located there is
/// picked up by whichever of them comes first (see
/// \ref model::Function::findByLocation), no matter which one it was meant
/// for.
struct AmbiguousLocations {
  std::set<SortedVector<MetaAddress>> Variables;
  std::set<SortedVector<MetaAddress>> Labels;
};

} // namespace

/// Collect the address sets appearing more than once in \p Locations. An empty
/// set identifies nothing to begin with and is reported on its own, so it never
/// counts as ambiguous.
static std::set<SortedVector<MetaAddress>>
findDuplicates(llvm::ArrayRef<SortedVector<MetaAddress>> Locations) {
  std::set<SortedVector<MetaAddress>> Seen;
  std::set<SortedVector<MetaAddress>> Duplicates;

  for (const SortedVector<MetaAddress> &Location : Locations)
    if (not Location.empty() and not Seen.insert(Location).second)
      Duplicates.insert(Location);

  return Duplicates;
}

/// Gather the address sets that cannot be attributed to a single local variable
/// or a single goto label of \p Function.
static AmbiguousLocations
collectAmbiguousLocations(clift::FunctionOp Function) {
  std::vector<SortedVector<MetaAddress>> Variables;
  Function.walk([&](clift::LocalVariableOp Op) {
    // The stack frame variable is identified by its handle, not by addresses.
    if (pipeline::locationFromString(rr::LocalVariable, Op.getHandle()))
      Variables.push_back(clift::getUserAddressSet(Op.getResult()));
  });

  std::vector<SortedVector<MetaAddress>> Labels;
  Function.walk([&](clift::MakeLabelOp Op) {
    if (pipeline::locationFromString(rr::GotoLabel, Op.getHandle()))
      Labels.push_back(clift::getUserAddressSet(Op.getResult()));
  });

  return { findDuplicates(Variables), findDuplicates(Labels) };
}

//
// Annotation building
//

/// A plain comment becomes a StatementComment attached to a statement, located
/// by the addresses of the instructions that make it up. The Index is assigned
/// by the caller.
static llvm::Expected<model::StatementComment>
makeStatementComment(mlir::Operation *Op,
                     llvm::ArrayRef<llvm::StringRef> Lines) {
  SortedVector<MetaAddress> Addresses;
  if (Op != nullptr)
    Addresses = clift::getStatementExpressionAddresses(Op);
  if (Addresses.empty()) {
    return revng::createError("the statement cannot be identified by its "
                              "addresses");
  }

  std::string Body;
  for (llvm::StringRef Line : Lines) {
    if (not Body.empty())
      Body += "\n";
    Body += Line.str();
  }

  model::StatementComment Comment;
  Comment.Body() = std::move(Body);
  for (const MetaAddress &Address : Addresses)
    Comment.Location().insert(Address);
  return Comment;
}

/// A `RENAME:`/`RETYPE:` directive renames and/or retypes a local variable,
/// located by the addresses of the instructions that use it. It can only be
/// applied to a local variable declaration.
static llvm::Expected<model::LocalVariable>
makeLocalVariableEdit(mlir::Operation *Op,
                      const std::optional<std::string> &NewName,
                      const std::optional<std::string> &NewTypeName,
                      const ResolvedTypeMap &ResolvedTypes,
                      const AmbiguousLocations &Ambiguous) {
  auto LocalVariable = mlir::dyn_cast_or_null<clift::LocalVariableOp>(Op);
  if (not LocalVariable
      or not pipeline::locationFromString(rr::LocalVariable,
                                          LocalVariable.getHandle())) {
    return revng::createError("`RENAME`/`RETYPE` can only be applied to a "
                              "local variable declaration");
  }

  SortedVector<MetaAddress> Location = clift::getUserAddressSet(LocalVariable);
  if (Location.empty()) {
    return revng::createError("the local variable cannot be identified by "
                              "its addresses");
  }

  if (Ambiguous.Variables.contains(Location)) {
    std::string Name = LocalVariable.getName().str();
    return revng::createError("`" + Name + "` shares its addresses ("
                              + addressesToString(Location)
                              + ") with another local variable, so it cannot "
                                "be edited: rev.ng identifies a local variable "
                                "by the addresses of the instructions using "
                                "it");
  }

  model::LocalVariable Variable;
  Variable.Name() = NewName.has_value() ? *NewName :
                                          LocalVariable.getName().str();
  if (NewTypeName.has_value()) {
    auto Iterator = ResolvedTypes.find(*NewTypeName);
    if (Iterator == ResolvedTypes.end() or Iterator->second.isEmpty())
      return revng::createError("unknown type in `RETYPE`: " + *NewTypeName);
    Variable.Type() = Iterator->second.copy();
  }
  Variable.Location() = std::move(Location);
  return Variable;
}

/// A `RENAME:` directive on a label renames it. The `GotoLabel` is located by
/// the addresses of the instructions that use the label. It can only be applied
/// to a label statement.
static llvm::Expected<model::GotoLabel>
makeGotoLabelEdit(mlir::Operation *Op,
                  const std::optional<std::string> &NewName,
                  const AmbiguousLocations &Ambiguous) {
  // The labels closing a loop body and following a loop are emitted by the C
  // backend, not lifted, so there is nothing in the model to rename.
  if (Op == nullptr) {
    return revng::createError("the label is synthesized by the C backend and "
                              "has no counterpart in the model");
  }

  auto AssignLabel = mlir::dyn_cast<clift::AssignLabelOp>(Op);
  clift::MakeLabelOp Label = AssignLabel ? AssignLabel.getLabelOp() : nullptr;
  if (not Label
      or not pipeline::locationFromString(rr::GotoLabel, Label.getHandle())) {
    return revng::createError("`RENAME` can only be applied to a goto label");
  }

  SortedVector<MetaAddress> Location = //
    clift::getUserAddressSet(AssignLabel.getLabel());
  if (Location.empty()) {
    return revng::createError("the label cannot be identified by its "
                              "addresses");
  }

  if (Ambiguous.Labels.contains(Location)) {
    std::string Name = Label.getName().str();
    return revng::createError("`" + Name + "` shares its addresses ("
                              + addressesToString(Location)
                              + ") with another goto label, so it cannot be "
                                "edited: rev.ng identifies a goto label by the "
                                "addresses of the instructions using it");
  }

  model::GotoLabel Result;
  Result.Name() = NewName.has_value() ? *NewName : Label.getName().str();
  Result.Location() = std::move(Location);
  return Result;
}

namespace {

/// The edits a statement's leading comments produce: at most one comment and at
/// most one local variable rename/retype or one label rename.
struct StatementEdits {
  std::optional<model::StatementComment> Comment;
  std::optional<model::LocalVariable> Variable;
  std::optional<model::GotoLabel> Label;
};

} // namespace

static void reportDropped(llvm::Error Error,
                          llvm::StringRef Annotation,
                          StatementKind Kind) {
  // The error has to be consumed whether or not the logger is enabled.
  std::string Reason = consumeToString(std::move(Error));
  revng_log(Log,
            "Ignoring " << Annotation << " on " << describe(Kind) << ": "
                        << Reason);
}

/// Names the directives a statement carries, for the message reporting them
/// dropped.
static llvm::StringRef describeDirectives(bool HasName, bool HasTypeName) {
  if (HasName and HasTypeName)
    return "`RENAME`/`RETYPE`";
  return HasName ? "`RENAME`" : "`RETYPE`";
}

/// Classify a statement's leading comments and delegate each kind to its
/// builder.
///
/// Each annotation is applied on its own: one that cannot be is dropped and
/// reported (see reportDropped), leaving the others on the same statement, and
/// on every other statement, unaffected.
static StatementEdits
computeStatementEdits(llvm::ArrayRef<std::string> LeadingComments,
                      mlir::Operation *Op,
                      StatementKind Kind,
                      const ResolvedTypeMap &ResolvedTypes,
                      const AmbiguousLocations &Ambiguous) {
  // A `RENAME:`/`RETYPE:` line edits the statement's local variable (or, on a
  // label statement, renames the label); any other line is a plain comment
  // attached to the statement.
  llvm::SmallVector<llvm::StringRef> PlainComments;
  std::optional<std::string> NewName;
  std::optional<std::string> NewTypeName;
  for (llvm::StringRef Line : LeadingComments) {
    if (Line.consume_front("RENAME:"))
      NewName = Line.trim().str();
    else if (Line.consume_front("RETYPE:"))
      NewTypeName = Line.trim().str();
    else
      PlainComments.push_back(Line);
  }

  StatementEdits Edits;

  if (not PlainComments.empty()) {
    auto MaybeComment = makeStatementComment(Op, PlainComments);
    if (MaybeComment) {
      Edits.Comment = std::move(*MaybeComment);
    } else {
      reportDropped(MaybeComment.takeError(),
                    "the comment \"" + llvm::join(PlainComments, " ") + "\"",
                    Kind);
    }
  }

  // A label has only a name, so `RETYPE:` does not apply to it. Dropping the
  // directive alone leaves a `RENAME:` on the same label working.
  if (NewTypeName.has_value() and Kind == StatementKind::Label) {
    reportDropped(revng::createError("`RETYPE` cannot be applied to a label"),
                  "`RETYPE`",
                  Kind);
    NewTypeName.reset();
  }

  if (Kind == StatementKind::Label) {
    if (NewName.has_value()) {
      auto MaybeLabel = makeGotoLabelEdit(Op, NewName, Ambiguous);
      if (MaybeLabel) {
        Edits.Label = std::move(*MaybeLabel);
      } else {
        reportDropped(MaybeLabel.takeError(), "`RENAME`", Kind);
      }
    }
  } else if (NewName.has_value() or NewTypeName.has_value()) {
    auto MaybeVariable = makeLocalVariableEdit(Op,
                                               NewName,
                                               NewTypeName,
                                               ResolvedTypes,
                                               Ambiguous);
    if (MaybeVariable) {
      Edits.Variable = std::move(*MaybeVariable);
    } else {
      reportDropped(MaybeVariable.takeError(),
                    describeDirectives(NewName.has_value(),
                                       NewTypeName.has_value()),
                    Kind);
    }
  }

  return Edits;
}

//
// Analysis
//

namespace {

struct ImportConfiguration {
  MetaAddress Function;
  std::string CCode;
};

} // namespace

template<>
struct llvm::yaml::MappingTraits<ImportConfiguration> {
  static void mapping(IO &IO, ImportConfiguration &Fields) {
    IO.mapRequired("Function", Fields.Function);
    IO.mapRequired("CCode", Fields.CCode);
  }
};

namespace revng::pypeline::analyses {

llvm::Error EditCBody::run(Model &Model,
                           const Request &Incoming,
                           llvm::StringRef Configuration,
                           const CliftFunctionContainer &Clift,
                           const PTMLCContainer &TypeAndGlobalHeader,
                           const PTMLCContainer &HelperHeader) {
  auto MaybeConfiguration = fromString<ImportConfiguration>(Configuration);
  if (not MaybeConfiguration)
    return MaybeConfiguration.takeError();

  ImportConfiguration &ParsedConfiguration = *MaybeConfiguration;

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

  // Flatten the Clift function into an ordered list of statements, each with
  // the set of addresses identifying it.
  std::vector<CliftStatement> CliftStatements;
  flattenCliftRegion(Function.getBody(), CliftStatements);

  AmbiguousLocations Ambiguous = collectAmbiguousLocations(Function);

  // Parse the user's C code and flatten it in the same way.
  auto MaybeHeader = writeHeader(TypeAndGlobalHeader, HelperHeader);
  if (not MaybeHeader)
    return MaybeHeader.takeError();

  auto MaybeParsed = parseUserFunction(MaybeHeader->path(),
                                       ParsedConfiguration.CCode,
                                       Binary);
  if (not MaybeParsed)
    return MaybeParsed.takeError();
  std::vector<CStatement> &UserStatements = MaybeParsed->Statements;

  // The C code must match the Clift function statement by statement; only the
  // comments may differ.
  if (UserStatements.size() != CliftStatements.size()) {
    return revng::createError("the provided C code has a different number of "
                              "statements than the decompiled function");
  }

  for (const auto &[Parsed, Decompiled] :
       llvm::zip(UserStatements, CliftStatements)) {
    if (Parsed.Kind != Decompiled.Kind) {
      return revng::createError("the provided C code differs in structure from "
                                "the decompiled function; only comments may be "
                                "changed");
    }
  }

  // Build the new comments, variable edits and label edits before touching the
  // model. Nothing below can fail any more, but keeping the two phases apart
  // means the model is only ever written once the whole C has been accounted
  // for.
  std::vector<model::StatementComment> NewComments;
  std::vector<model::LocalVariable> NewVariables;
  std::vector<model::GotoLabel> NewLabels;

  for (const auto &[Parsed, Decompiled] :
       llvm::zip(UserStatements, CliftStatements)) {
    if (Parsed.LeadingComments.empty())
      continue;

    StatementEdits Edits = computeStatementEdits(Parsed.LeadingComments,
                                                 Decompiled.Op,
                                                 Decompiled.Kind,
                                                 MaybeParsed->ResolvedTypes,
                                                 Ambiguous);

    if (Edits.Comment.has_value()) {
      Edits.Comment->Index() = NewComments.size();
      NewComments.push_back(std::move(*Edits.Comment));
    }
    if (Edits.Variable.has_value())
      NewVariables.push_back(std::move(*Edits.Variable));
    if (Edits.Label.has_value())
      NewLabels.push_back(std::move(*Edits.Label));
  }

  // Replace the function's comments with the imported ones.
  model::Function &ModelFunction = *FunctionIterator;
  ModelFunction.Comments().clear();
  for (model::StatementComment &Comment : NewComments)
    ModelFunction.Comments().insert(std::move(Comment));

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
