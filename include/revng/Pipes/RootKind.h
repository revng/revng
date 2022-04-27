#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/STLExtras.h"

#include "revng/Pipeline/Context.h"
#include "revng/Pipeline/GenericLLVMPipe.h"
#include "revng/Pipeline/Kind.h"
#include "revng/Pipeline/LLVMContainer.h"
#include "revng/Pipeline/LLVMGlobalKindBase.h"
#include "revng/TupleTree/TupleTreeDiff.h"
#include "revng/TupleTree/TupleTreePath.h"

namespace revng::pipes {

class RootKind : public pipeline::LLVMKind {
public:
  using pipeline::LLVMKind::LLVMKind;

  pipeline::TargetsList
  compactTargets(const pipeline::Context &Ctx,
                 pipeline::TargetsList::List &Targets) const final {
    return Targets;
  }

  std::optional<pipeline::Target>
  symbolToTarget(const llvm::Function &Symbol) const override;

  void
  getInvalidations(pipeline::TargetsList &ToRemove,
                   const pipeline::GlobalTupleTreeDiff &Base) const override;
};

class IsolatedRootKind : public pipeline::LLVMKind {
public:
  using pipeline::LLVMKind::LLVMKind;

  pipeline::TargetsList
  compactTargets(const pipeline::Context &Ctx,
                 pipeline::TargetsList::List &Targets) const final {
    return Targets;
  }

  std::optional<pipeline::Target>
  symbolToTarget(const llvm::Function &Symbol) const override;
};

} // namespace revng::pipes
