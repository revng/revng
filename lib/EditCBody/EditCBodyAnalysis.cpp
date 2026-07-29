//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <map>
#include <optional>
#include <string>
#include <vector>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
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
    return revng::createError("a comment is attached to a statement that "
                              "cannot be identified by its addresses");
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
                      const ResolvedTypeMap &ResolvedTypes) {
  auto LocalVariable = mlir::dyn_cast_or_null<clift::LocalVariableOp>(Op);
  if (not LocalVariable
      or not pipeline::locationFromString(rr::LocalVariable,
                                          LocalVariable.getHandle())) {
    return revng::createError("`RENAME`/`RETYPE` can only be applied to a "
                              "local variable declaration");
  }

  SortedVector<MetaAddress> Location = //
    clift::getUserAddressSet(LocalVariable.getResult());
  if (Location.empty()) {
    return revng::createError("the local variable cannot be identified by "
                              "its addresses");
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
                  const std::optional<std::string> &NewName) {
  auto AssignLabel = mlir::dyn_cast_or_null<clift::AssignLabelOp>(Op);
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

/// Classify a statement's leading comments and delegate each kind to its
/// builder.
static llvm::Expected<StatementEdits>
computeStatementEdits(llvm::ArrayRef<std::string> LeadingComments,
                      mlir::Operation *Op,
                      StatementKind Kind,
                      const ResolvedTypeMap &ResolvedTypes) {
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
    if (not MaybeComment)
      return MaybeComment.takeError();
    Edits.Comment = std::move(*MaybeComment);
  }

  if (NewName.has_value() or NewTypeName.has_value()) {
    if (Kind == StatementKind::Label) {
      // A label has only a name; `RETYPE:` does not apply.
      if (NewTypeName.has_value())
        return revng::createError("`RETYPE` cannot be applied to a label");
      auto MaybeLabel = makeGotoLabelEdit(Op, NewName);
      if (not MaybeLabel)
        return MaybeLabel.takeError();
      Edits.Label = std::move(*MaybeLabel);
    } else {
      auto MaybeVariable = makeLocalVariableEdit(Op,
                                                 NewName,
                                                 NewTypeName,
                                                 ResolvedTypes);
      if (not MaybeVariable)
        return MaybeVariable.takeError();
      Edits.Variable = std::move(*MaybeVariable);
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
  // model, so that a failure leaves it untouched.
  std::vector<model::StatementComment> NewComments;
  std::vector<model::LocalVariable> NewVariables;
  std::vector<model::GotoLabel> NewLabels;

  for (const auto &[Parsed, Decompiled] :
       llvm::zip(UserStatements, CliftStatements)) {
    if (Parsed.LeadingComments.empty())
      continue;

    auto MaybeEdits = computeStatementEdits(Parsed.LeadingComments,
                                            Decompiled.Op,
                                            Decompiled.Kind,
                                            MaybeParsed->ResolvedTypes);
    if (not MaybeEdits)
      return MaybeEdits.takeError();

    if (MaybeEdits->Comment.has_value()) {
      MaybeEdits->Comment->Index() = NewComments.size();
      NewComments.push_back(std::move(*MaybeEdits->Comment));
    }
    if (MaybeEdits->Variable.has_value())
      NewVariables.push_back(std::move(*MaybeEdits->Variable));
    if (MaybeEdits->Label.has_value())
      NewLabels.push_back(std::move(*MaybeEdits->Label));
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
