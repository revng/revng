/// \file RootKind.cpp
/// The kind associated to non isolated root.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <optional>

#include "llvm/Support/Casting.h"

#include "revng/Model/Binary.h"
#include "revng/Model/FunctionTags.h"
#include "revng/Pipeline/AllRegistries.h"
#include "revng/Pipeline/Kind.h"
#include "revng/Pipeline/Target.h"
#include "revng/Pipes/Kinds.h"
#include "revng/Pipes/RootKind.h"
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

void RootKind::appendAllTargets(const Context &Context,
                                TargetsList &Out) const {
  Out.push_back(Target(*this));
}

void IsolatedRootKind::appendAllTargets(const Context &Context,
                                        TargetsList &Out) const {
  Out.push_back(Target(*this));
}
