/// \file RootKind.cpp
/// \brief the kind associated to non isolated root.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <optional>

#include "revng/Pipeline/Kind.h"
#include "revng/Pipeline/Registry.h"
#include "revng/Pipeline/Target.h"
#include "revng/Pipes/Kinds.h"
#include "revng/Pipes/RootKind.h"
#include "revng/Support/FunctionTags.h"

using namespace pipeline;
using namespace revng::pipes;

RootKind::RootKind() : LLVMKind("Root", &RootRank) {
}

std::optional<Target>
RootKind::symbolToTarget(const llvm::Function &Symbol) const {
  if (FunctionTags::Root.isTagOf(&Symbol)
      and not FunctionTags::IsolatedRoot.isTagOf(&Symbol))
    return Target("root", *this);

  return std::nullopt;
}

RootKind revng::pipes::Root;

IsolatedRootKind::IsolatedRootKind() : LLVMKind("IsolatedRoot", Root) {
}

std::optional<Target>
IsolatedRootKind::symbolToTarget(const llvm::Function &Symbol) const {
  if (FunctionTags::IsolatedRoot.isTagOf(&Symbol))
    return Target("root", *this);

  return std::nullopt;
}

IsolatedRootKind revng::pipes::IsolatedRoot;

static RegisterKind K1(Root);
static RegisterKind K2(IsolatedRoot);
