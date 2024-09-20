#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/Model/Binary.h"
#include "revng/Model/Generated/Early/TypeDefinition.h"
#include "revng/Model/LoadModelPass.h"
#include "revng/Pipeline/ContainerSet.h"
#include "revng/Pipeline/Context.h"
#include "revng/Pipeline/Contract.h"
#include "revng/Pipeline/GenericLLVMPipe.h"
#include "revng/Pipeline/LLVMContainer.h"
#include "revng/Pipeline/Loader.h"
#include "revng/Pipeline/Target.h"
#include "revng/Support/YAMLTraits.h"
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
getModelFromContext(const pipeline::ExecutionContext &EC) {
  return getModelFromContext(EC.getContext());
}

inline TupleTree<model::Binary> &
getWritableModelFromContext(pipeline::Context &Ctx) {
  using Wrapper = ModelGlobal;
  const auto &Model = llvm::cantFail(Ctx.getGlobal<Wrapper>(ModelGlobalName));
  Model->get().evictCachedReferences();
  return Model->get();
}

inline TupleTree<model::Binary> &
getWritableModelFromContext(pipeline::ExecutionContext &EC) {
  return getWritableModelFromContext(EC.getContext());
}

inline cppcoro::generator<const model::Function &>
getFunctionsAndCommit(pipeline::ExecutionContext &EC,
                      llvm::StringRef ContainerName) {
  const auto &Binary = revng::getModelFromContext(EC);
  auto Extractor =
    [&](const pipeline::Target &Target) -> const model::Function & {
    auto MetaAddress = MetaAddress::fromString(Target.getPathComponents()[0]);
    return Binary->Functions().at(MetaAddress);
  };
  for (const auto &F :
       EC.getAndCommit<model::Function>(Extractor, ContainerName))
    co_yield F;
}

inline cppcoro::generator<const model::TypeDefinition &>
getTypeDefinitionsAndCommit(pipeline::ExecutionContext &EC,
                            llvm::StringRef ContainerName) {
  using model::TypeDefinition;
  const auto &Binary = revng::getModelFromContext(EC);
  auto Extractor =
    [&](const pipeline::Target &Target) -> const TypeDefinition & {
    using KeyTuple = TypeDefinition::Key;
    auto Key = cantFail(deserialize<KeyTuple>(Target.getPathComponents()[0]));
    return *Binary->TypeDefinitions().at(Key);
  };
  for (const auto &T :
       EC.getAndCommit<TypeDefinition>(Extractor, ContainerName))
    co_yield T;
}

} // namespace revng
