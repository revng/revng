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
#include "ModelEdits.h"
#include "Statements.h"

using namespace llvm;
using namespace revng::editcbody;

static Logger Log("edit-c-body");

namespace rr = revng::ranks;

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

/// A `RENAME:`/`RETYPE:` directive renames and/or retypes a local variable. It
/// can only be applied to a local variable declaration.
static llvm::Expected<model::LocalVariable>
makeLocalVariableEditAt(mlir::Operation *Op,
                        const std::optional<std::string> &NewName,
                        const std::optional<std::string> &NewTypeName,
                        const std::optional<std::string> &NewComment,
                        const ResolvedTypeMap &ResolvedTypes,
                        const AmbiguousLocations &Ambiguous) {
  auto Variable = mlir::dyn_cast_or_null<clift::LocalVariableOp>(Op);
  return makeLocalVariableEdit(Variable,
                               NewName,
                               NewTypeName,
                               NewComment,
                               ResolvedTypes,
                               Ambiguous);
}

/// A `RENAME:` directive on a label renames it. It can only be applied to a
/// label statement.
static llvm::Expected<model::GotoLabel>
makeGotoLabelEditAt(mlir::Operation *Op,
                    const std::optional<std::string> &NewName,
                    const std::optional<std::string> &NewComment,
                    const AmbiguousLocations &Ambiguous) {
  // The labels closing a loop body and following a loop are emitted by the C
  // backend, not lifted, so there is nothing in the model to rename.
  if (Op == nullptr) {
    return revng::createError("the label is synthesized by the C backend and "
                              "has no counterpart in the model");
  }

  auto AssignLabel = mlir::dyn_cast<clift::AssignLabelOp>(Op);
  clift::MakeLabelOp Label = AssignLabel ? AssignLabel.getLabelOp() : nullptr;
  if (not Label)
    return revng::createError("`RENAME` can only be applied to a goto label");

  return makeGotoLabelEdit(Label, NewName, NewComment, Ambiguous);
}

/// Whether \p Op declares the stack frame variable, whose comment the model
/// keeps on the function rather than among its local variables.
static bool isStackFrameDeclaration(mlir::Operation *Op) {
  auto Variable = mlir::dyn_cast_or_null<clift::LocalVariableOp>(Op);
  if (not Variable)
    return false;

  return pipeline::locationFromString(rr::StackFrameVariable,
                                      Variable.getHandle())
    .has_value();
}

namespace {

/// The edits a statement's leading comments produce: at most one comment and at
/// most one local variable rename/retype/comment or one label rename/comment.
///
/// Where the comment goes depends on what it sits above. A declaration and a
/// label own a comment of their own, so it is folded into \ref Variable or
/// \ref Label; anywhere else it becomes a \ref Comment attached to the point in
/// the code. The stack frame is a declaration whose comment lives on the
/// function rather than in `LocalVariables`, so it travels on its own.
struct StatementEdits {
  std::optional<model::StatementComment> Comment;
  std::optional<model::LocalVariable> Variable;
  std::optional<model::GotoLabel> Label;
  std::optional<std::string> StackFrameComment;
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
  if (HasName)
    return "`RENAME`";
  return HasTypeName ? "`RETYPE`" : "the comment";
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

  // A comment above a declaration or a label is about the entity it introduces,
  // which is also where the decompiler writes one back, so it is recorded on
  // that entity. Everywhere else a comment is about the point in the code, and
  // stays a statement comment.
  bool CommentsAnEntity = Kind == StatementKind::LocalVariableDeclaration
                          or Kind == StatementKind::Label;

  std::optional<std::string> NewComment;
  if (not PlainComments.empty() and CommentsAnEntity)
    NewComment = llvm::join(PlainComments, "\n");

  // The stack frame is not in `LocalVariables`, so nothing locates it by
  // address and its comment is taken here rather than by the edit below.
  if (NewComment.has_value() and isStackFrameDeclaration(Op)) {
    Edits.StackFrameComment = std::move(*NewComment);
    NewComment.reset();
  }

  auto RecordStatementComment = [&] {
    auto MaybeComment = makeStatementComment(Op, PlainComments);
    if (MaybeComment) {
      Edits.Comment = std::move(*MaybeComment);
    } else {
      reportDropped(MaybeComment.takeError(),
                    "the comment \"" + llvm::join(PlainComments, " ") + "\"",
                    Kind);
    }
  };

  if (not PlainComments.empty() and not CommentsAnEntity)
    RecordStatementComment();

  // A label has only a name, so `RETYPE:` does not apply to it. Dropping the
  // directive alone leaves a `RENAME:` on the same label working.
  if (NewTypeName.has_value() and Kind == StatementKind::Label) {
    reportDropped(revng::createError("`RETYPE` cannot be applied to a label"),
                  "`RETYPE`",
                  Kind);
    NewTypeName.reset();
  }

  if (Kind == StatementKind::Label) {
    if (NewName.has_value() or NewComment.has_value()) {
      auto MaybeLabel = makeGotoLabelEditAt(Op, NewName, NewComment, Ambiguous);
      if (MaybeLabel) {
        Edits.Label = std::move(*MaybeLabel);
      } else {
        reportDropped(MaybeLabel.takeError(),
                      NewName.has_value() ? "`RENAME`" : "the comment",
                      Kind);

        // Nothing holds a comment the label could not take, so keep it where a
        // comment always used to go rather than losing it.
        if (NewComment.has_value())
          RecordStatementComment();
      }
    }
  } else if (NewName.has_value() or NewTypeName.has_value()
             or NewComment.has_value()) {
    auto MaybeVariable = makeLocalVariableEditAt(Op,
                                                 NewName,
                                                 NewTypeName,
                                                 NewComment,
                                                 ResolvedTypes,
                                                 Ambiguous);
    if (MaybeVariable) {
      Edits.Variable = std::move(*MaybeVariable);
    } else {
      reportDropped(MaybeVariable.takeError(),
                    describeDirectives(NewName.has_value(),
                                       NewTypeName.has_value()),
                    Kind);

      if (NewComment.has_value())
        RecordStatementComment();
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
  std::optional<std::string> NewStackFrameComment;

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
    if (Edits.StackFrameComment.has_value())
      NewStackFrameComment = std::move(*Edits.StackFrameComment);
  }

  // Replace the function's comments with the imported ones.
  model::Function &ModelFunction = *FunctionIterator;

  // A comment above the stack frame declaration replaces the one it carries.
  // A body carrying none leaves it alone, as it does for every other comment
  // belonging to an entity rather than to a point in the code.
  if (NewStackFrameComment.has_value())
    ModelFunction.StackFrame().Comment() = std::move(*NewStackFrameComment);

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
