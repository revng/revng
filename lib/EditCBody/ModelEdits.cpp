//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/Clift/CliftOpHelpers.h"
#include "revng/Clift/LocationAddresses.h"
#include "revng/Ranks/Location.h"
#include "revng/Ranks/Ranks.h"
#include "revng/Support/Assert.h"
#include "revng/Support/Error.h"

#include "ModelEdits.h"

using namespace llvm;

namespace rr = revng::ranks;

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

/// The comment the model already attached to \p Op, which the importer left on
/// it.
///
/// An edit replaces the model entry rather than amending it, so a directive the
/// edit does not carry has to be carried over from what is there, just as the
/// name is.
static std::string currentComment(mlir::Operation *Op) {
  if (auto Comment = Op->getAttrOfType<mlir::StringAttr>("clift.comment"))
    return Comment.getValue().str();
  return {};
}

namespace revng::editcbody {

AmbiguousLocations collectAmbiguousLocations(clift::FunctionOp Function) {
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

llvm::Expected<model::LocalVariable>
makeLocalVariableEdit(clift::LocalVariableOp Variable,
                      const std::optional<std::string> &NewName,
                      const std::optional<std::string> &NewTypeName,
                      const std::optional<std::string> &NewComment,
                      const ResolvedTypeMap &ResolvedTypes,
                      const AmbiguousLocations &Ambiguous) {
  if (not Variable) {
    return revng::createError("`RENAME`/`RETYPE` can only be applied to a "
                              "local variable declaration");
  }

  // The stack frame and the reserved variables are identified by their handle,
  // not by addresses, so an edit has nothing to locate them by.
  if (not pipeline::locationFromString(rr::LocalVariable,
                                       Variable.getHandle())) {
    return revng::createError("rev.ng does not identify this variable by the "
                              "addresses of the instructions using it, so it "
                              "cannot be edited");
  }

  SortedVector<MetaAddress> Location = //
    clift::getUserAddressSet(Variable.getResult());
  if (Location.empty()) {
    return revng::createError("the local variable cannot be identified by "
                              "its addresses");
  }

  if (Ambiguous.Variables.contains(Location)) {
    std::string Name = Variable.getName().str();
    return revng::createError("`" + Name + "` shares its addresses ("
                              + addressesToString(Location)
                              + ") with another local variable, so it cannot "
                                "be edited: rev.ng identifies a local variable "
                                "by the addresses of the instructions using "
                                "it");
  }

  model::LocalVariable Result;
  Result.Name() = NewName.has_value() ? *NewName : Variable.getName().str();
  Result.Comment() = NewComment.has_value() ? *NewComment :
                                              currentComment(Variable);
  if (NewTypeName.has_value()) {
    auto Iterator = ResolvedTypes.find(*NewTypeName);
    if (Iterator == ResolvedTypes.end() or Iterator->second.isEmpty())
      return revng::createError("unknown type: " + *NewTypeName);
    Result.Type() = Iterator->second.copy();
  }
  Result.Location() = std::move(Location);
  return Result;
}

llvm::Expected<model::GotoLabel>
makeGotoLabelEdit(clift::MakeLabelOp Label,
                  const std::optional<std::string> &NewName,
                  const std::optional<std::string> &NewComment,
                  const AmbiguousLocations &Ambiguous) {
  revng_assert(Label);

  // A label the C backend synthesizes has no counterpart in the model.
  if (not pipeline::locationFromString(rr::GotoLabel, Label.getHandle())) {
    return revng::createError("rev.ng does not identify this label by the "
                              "addresses of the instructions using it, so it "
                              "cannot be edited");
  }

  SortedVector<MetaAddress> Location = //
    clift::getUserAddressSet(Label.getResult());
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
  Result.Comment() = NewComment.has_value() ? *NewComment :
                                              currentComment(Label);
  Result.Location() = std::move(Location);
  return Result;
}

llvm::Expected<TemporaryFile>
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

} // namespace revng::editcbody
