/// \file RootKind.cpp
/// The kind associated to non isolated root.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <optional>

#include "llvm/Support/Casting.h"

#include "revng/Model/Binary.h"
#include "revng/Pipeline/AllRegistries.h"
#include "revng/Pipeline/Kind.h"
#include "revng/Pipeline/Target.h"
#include "revng/Pipes/Kinds.h"
#include "revng/Pipes/RootKind.h"
#include "revng/Support/FunctionTags.h"
#include "revng/TupleTree/Visits.h"

using namespace pipeline;
using namespace ::revng::kinds;

std::optional<Target>
RootKind::symbolToTarget(const llvm::Function &Symbol) const {
  if (Symbol.isDeclaration())
    return std::nullopt;
  if (FunctionTags::Root.isTagOf(&Symbol)
      and not FunctionTags::IsolatedRoot.isTagOf(&Symbol))
    return Target({}, *this);

  return std::nullopt;
}

std::optional<Target>
IsolatedRootKind::symbolToTarget(const llvm::Function &Symbol) const {
  if (Symbol.isDeclaration())
    return std::nullopt;
  if (FunctionTags::IsolatedRoot.isTagOf(&Symbol))
    return Target({}, *this);

  return std::nullopt;
}

void RootKind::getInvalidations(const Context &Ctx,
                                TargetsList &ToRemove,
                                const GlobalTupleTreeDiff &Base) const {

  const auto *Diff = Base.getAs<model::Binary>();
  if (not Diff)
    return;

  const TupleTreePath ToCheck = *stringAsPath<model::Binary>("/ExtraCodeAddre"
                                                             "ss"
                                                             "es");

  bool RootChanged = llvm::any_of(Diff->Changes, [&ToCheck](const auto &Entry) {
    const auto &[Path, Old, New] = Entry;
    return ToCheck.isPrefixOf(Path);
  });

  if (RootChanged)
    ToRemove.emplace_back(*this);
}

void RootKind::appendAllTargets(const Context &Ctx, TargetsList &Out) const {
  Out.push_back(Target(*this));
}

void IsolatedRootKind::appendAllTargets(const Context &Ctx,
                                        TargetsList &Out) const {
  Out.push_back(Target(*this));
}
