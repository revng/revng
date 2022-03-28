/// \file RootKind.cpp
/// \brief the kind associated to non isolated root.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <optional>

#include "revng/Pipeline/AllRegistries.h"
#include "revng/Pipeline/Kind.h"
#include "revng/Pipeline/Target.h"
#include "revng/Pipes/Kinds.h"
#include "revng/Pipes/RootKind.h"
#include "revng/Support/FunctionTags.h"

using namespace pipeline;
using namespace ::revng::pipes;

std::optional<Target>
RootKind::symbolToTarget(const llvm::Function &Symbol) const {
  if (FunctionTags::Root.isTagOf(&Symbol)
      and not FunctionTags::IsolatedRoot.isTagOf(&Symbol))
    return Target({}, *this);

  return std::nullopt;
}

std::optional<Target>
IsolatedRootKind::symbolToTarget(const llvm::Function &Symbol) const {
  if (FunctionTags::IsolatedRoot.isTagOf(&Symbol))
    return Target({}, *this);

  return std::nullopt;
}
