#pragma once

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/Model/Binary.h"
#include "revng/Model/LoadModelPass.h"
#include "revng/Pipeline/ContainerSet.h"
#include "revng/Pipeline/Context.h"
#include "revng/Pipeline/Contract.h"
#include "revng/Pipeline/GenericLLVMPipe.h"
#include "revng/Pipeline/LLVMContainer.h"
#include "revng/Pipeline/Loader.h"
#include "revng/Pipeline/Target.h"
#include "revng/TupleTree/TupleTree.h"

namespace revng {
constexpr static const char *ModelGlobalName = "model.yml";
using ModelGlobal = pipeline::TupleTreeGlobal<model::Binary>;

inline const TupleTree<model::Binary> &
getModelFromContext(const pipeline::Context &Ctx) {
  using Wrapper = ModelGlobal;
  const auto &Model = llvm::cantFail(Ctx.getGlobal<Wrapper>(ModelGlobalName));
  Model->get().cacheReferences();
  return Model->get();
}

inline const TupleTree<model::Binary> &
getModelFromContext(const pipeline::ExecutionContext &Ctx) {
  return getModelFromContext(Ctx.getContext());
}

inline TupleTree<model::Binary> &
getWritableModelFromContext(pipeline::Context &Ctx) {
  using Wrapper = ModelGlobal;
  const auto &Model = llvm::cantFail(Ctx.getGlobal<Wrapper>(ModelGlobalName));
  Model->get().evictCachedReferences();
  return Model->get();
}

inline TupleTree<model::Binary> &
getWritableModelFromContext(pipeline::ExecutionContext &Ctx) {
  return getWritableModelFromContext(Ctx.getContext());
}
} // namespace revng
