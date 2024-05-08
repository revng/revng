/// \file LoadModelPass.cpp
/// Implementation of the immutable pass providing access to the model and
/// taking care of its deserialization on the IR.

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

// LLVM includes
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/Error.h"

// Local libraries includes
#include "revng/Model/LoadModelPass.h"
#include "revng/TupleTree/TupleTree.h"

using namespace llvm;

char LoadModelWrapperPass::ID;

AnalysisKey LoadModelAnalysis::Key;

template<typename T>
using RP = RegisterPass<T>;

static RP<LoadModelWrapperPass>
  X("load-model", "Deserialize the model", true, true);

using namespace llvm::cl;
opt<std::string> ModelPath("model-path", desc("<model path>"), init(""));

bool hasModel(const llvm::Module &M) {
  NamedMDNode *NamedMD = M.getNamedMetadata(ModelMetadataName);
  return NamedMD and NamedMD->getNumOperands();
}

TupleTree<model::Binary> loadModel(const llvm::Module &M) {
  revng_check(hasModel(M));

  NamedMDNode *NamedMD = M.getNamedMetadata(ModelMetadataName);
  auto *Tuple = cast<MDTuple>(NamedMD->getOperand(0));
  revng_check(Tuple->getNumOperands());

  Metadata *MD = Tuple->getOperand(0).get();
  StringRef YAMLString = cast<MDString>(MD)->getString();

  return std::move(TupleTree<model::Binary>::deserialize(YAMLString).get());
}

static TupleTree<model::Binary> extractModel(Module &M) {
  auto Result = loadModel(M);
  // Erase the named metadata in order to make sure no one is tempted to
  // deserialize it on its own
  NamedMDNode *NamedMD = M.getNamedMetadata(ModelMetadataName);
  NamedMD->eraseFromParent();

  return Result;
}

bool LoadModelWrapperPass::doInitialization(Module &M) {
  if (isModelExternal())
    return false;

  if (not ModelPath.empty()) {
    auto MaybeBinary = TupleTree<model::Binary>::fromFile(ModelPath);
    auto ExpectedBinary = llvm::errorOrToExpected(std::move(MaybeBinary));
    InternalModel = llvm::cantFail(std::move(ExpectedBinary));
  } else {

    InternalModel = extractModel(M);
  }

  return false;
}

bool LoadModelWrapperPass::doFinalization(Module &M) {
  if (isModelExternal() or not Wrapper.hasChanged())
    return false;

  // Check if the named metadata has reappeared. If not, the changes we made in
  // this pipeline would go lost
  NamedMDNode *NamedMD = M.getNamedMetadata(ModelMetadataName);
  revng_check(NamedMD != nullptr,
              "The model has changed, but -serialize-model has not been run");

  return false;
}

ModelWrapper LoadModelAnalysis::run(Module &M, ModuleAnalysisManager &) {
  if (not isModelExternal())
    InternalModel = loadModel(M);
  return Wrapper;
}

ModelWrapper LoadModelAnalysis::run(Function &F, FunctionAnalysisManager &) {
  if (not isModelExternal())
    InternalModel = loadModel(*F.getParent());
  return Wrapper;
}

const TupleTree<model::Binary> &ModelWrapper::getReadOnlyModel() const {
  auto Result = [](auto &Model) -> const TupleTree<model::Binary> & {
    using TupleTreeT = std::remove_pointer_t<std::decay_t<decltype(Model)>>;
    if constexpr (not std::is_const_v<TupleTreeT>)
      Model->cacheReferences();
    return *std::as_const(Model);
  };
  return std::visit(Result, TheBinary);
}

TupleTree<model::Binary> &ModelWrapper::getWriteableModel() {
  HasChanged = true;
  auto Result = [](auto &Model) -> TupleTree<model::Binary> & {
    using TupleTreeT = std::remove_pointer_t<std::decay_t<decltype(Model)>>;
    if constexpr (std::is_const_v<TupleTreeT>) {
      revng_abort("A writeable model has been requested, but the wrapper has "
                  "a reference to a const model");
    } else {
      Model->evictCachedReferences();
      return *Model;
    }
  };
  return std::visit(Result, TheBinary);
}
