#pragma once

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

#include "llvm/ADT/STLExtras.h"

#include "revng/Pipeline/Context.h"
#include "revng/Pipeline/GenericLLVMPipe.h"
#include "revng/Pipeline/Kind.h"
#include "revng/Pipeline/LLVMContainer.h"
#include "revng/Pipeline/LLVMKind.h"
#include "revng/TupleTree/TupleTreeDiff.h"
#include "revng/TupleTree/TupleTreePath.h"

namespace revng::kinds {

class RootKind : public pipeline::LLVMKind {
public:
  using pipeline::LLVMKind::LLVMKind;

  std::optional<pipeline::Target>
  symbolToTarget(const llvm::Function &Symbol) const override;

  void appendAllTargets(const pipeline::Context &Ctx,
                        pipeline::TargetsList &Out) const override;
};

class IsolatedRootKind : public pipeline::LLVMKind {
public:
  using pipeline::LLVMKind::LLVMKind;

  std::optional<pipeline::Target>
  symbolToTarget(const llvm::Function &Symbol) const override;

  void appendAllTargets(const pipeline::Context &Ctx,
                        pipeline::TargetsList &Out) const override;
};

} // namespace revng::kinds
