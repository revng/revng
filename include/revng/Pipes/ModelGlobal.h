#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/Binary.h"
#include "revng/Model/LoadModelPass.h"
#include "revng/Pipeline/ContainerSet.h"
#include "revng/Pipeline/Context.h"
#include "revng/Pipeline/Contract.h"
#include "revng/Pipeline/GenericLLVMPipe.h"
#include "revng/Pipeline/LLVMContainer.h"
#include "revng/Pipeline/Loader.h"
#include "revng/Pipeline/SavableObject.h"
#include "revng/Pipeline/Target.h"
#include "revng/TupleTree/TupleTree.h"

namespace revng::pipes {

class ModelGlobal : public pipeline::SavableObject<ModelGlobal> {
private:
  TupleTree<model::Binary> Model;

public:
  constexpr static const char *Name = "model.yml";
  static const char ID;
  llvm::Error storeToDisk(llvm::StringRef Path) const final;
  llvm::Error loadFromDisk(llvm::StringRef Path) final;

  explicit ModelGlobal(TupleTree<model::Binary> Model) :
    Model(std::move(Model)) {}
  ModelGlobal() = default;

  const TupleTree<model::Binary> &getModelWrapper() const { return Model; }
  TupleTree<model::Binary> &getModel() { return Model; }
};

inline const model::Binary &getModelFromContext(const pipeline::Context &Ctx) {
  using Wrapper = ModelGlobal;
  const auto &Model = llvm::cantFail(Ctx.getGlobal<Wrapper>(Wrapper::Name));
  return *Model->getModel();
}

inline TupleTree<model::Binary> &
getWritableModelFromContext(pipeline::Context &Ctx) {
  using Wrapper = ModelGlobal;
  const auto &Model = llvm::cantFail(Ctx.getGlobal<Wrapper>(Wrapper::Name));
  return Model->getModel();
}

} // namespace revng::pipes
