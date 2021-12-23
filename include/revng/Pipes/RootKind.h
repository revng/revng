#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Pipeline/Context.h"
#include "revng/Pipeline/GenericLLVMPipe.h"
#include "revng/Pipeline/Kind.h"
#include "revng/Pipeline/LLVMContainer.h"
#include "revng/Pipeline/LLVMGlobalKindBase.h"

namespace revng::pipes {
class RootKind : public pipeline::LLVMKind {

public:
  RootKind();

  pipeline::TargetsList
  compactTargets(const pipeline::Context &Ctx,
                 pipeline::TargetsList::List &Targets) const final {
    return Targets;
  }

  std::optional<pipeline::Target>
  symbolToTarget(const llvm::Function &Symbol) const override;
};

extern RootKind Root;

class IsolatedRootKind : public pipeline::LLVMKind {
public:
  IsolatedRootKind();

  pipeline::TargetsList
  compactTargets(const pipeline::Context &Ctx,
                 pipeline::TargetsList::List &Targets) const final {
    return Targets;
  }

  std::optional<pipeline::Target>
  symbolToTarget(const llvm::Function &Symbol) const override;
};

extern IsolatedRootKind IsolatedRoot;

} // namespace revng::pipes
