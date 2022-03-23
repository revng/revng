#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <variant>

#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"

#include "revng/Model/Binary.h"
#include "revng/TupleTree/TupleTree.h"

extern llvm::cl::opt<std::string> ModelPath;

inline const char *ModelMetadataName = "revng.model";

TupleTree<model::Binary> loadModel(const llvm::Module &M);

bool hasModel(const llvm::Module &M);

class ModelWrapper {
private:
  using Storage = std::variant<TupleTree<model::Binary> *,
                               const TupleTree<model::Binary> *>;

private:
  Storage TheBinary;
  bool HasChanged = false;

public:
  ModelWrapper(TupleTree<model::Binary> &TheBinary) : TheBinary(&TheBinary) {}
  ModelWrapper(const TupleTree<model::Binary> &TheBinary) :
    TheBinary(&TheBinary) {}

public:
  static ModelWrapper createConst(const TupleTree<model::Binary> &TheBinary) {
    return ModelWrapper(TheBinary);
  }

public:
  const TupleTree<model::Binary> &getReadOnlyModel() const {
    return std::visit([](auto &Model)
                        -> const TupleTree<model::Binary> & { return *Model; },
                      TheBinary);
  }

  TupleTree<model::Binary> &getWriteableModel() {
    HasChanged = true;
    auto Result = [](auto &Model) -> TupleTree<model::Binary> & {
      using T = std::decay_t<decltype(Model)>;
      if constexpr (std::is_same_v<T, const TupleTree<model::Binary> *>) {
        revng_abort("A writeable model has been requested, but the wrapper has "
                    "a reference to a const model");
      } else if constexpr (std::is_same_v<T, TupleTree<model::Binary> *>) {
        return *Model;
      } else {
        revng_abort();
      }
    };
    return std::visit(Result, TheBinary);
  }

  bool hasChanged() const { return HasChanged; }

  template<typename IRUnitT, typename PreservedAnalysesT, typename InvalidatorT>
  bool invalidate(IRUnitT &, const PreservedAnalysesT &, InvalidatorT &) {
    return false;
  }
};

class LoadModelWrapperPass : public llvm::ImmutablePass {
public:
  static char ID;

private:
  TupleTree<model::Binary> InternalModel;
  bool External;
  ModelWrapper Wrapper;

public:
  LoadModelWrapperPass() :
    llvm::ImmutablePass(ID),
    External(false),
    Wrapper(ModelWrapper(InternalModel)) {}
  LoadModelWrapperPass(const ModelWrapper &Wrapper) :
    llvm::ImmutablePass(ID), External(true), Wrapper(Wrapper) {}

public:
  bool doInitialization(llvm::Module &M) override final;
  bool doFinalization(llvm::Module &M) override final;

public:
  ModelWrapper &get() { return Wrapper; }

  bool isModelExternal() const { return External; }
};

class LoadModelAnalysis : public llvm::AnalysisInfoMixin<LoadModelAnalysis> {
  friend llvm::AnalysisInfoMixin<LoadModelAnalysis>;

private:
  static llvm::AnalysisKey Key;

public:
  using Result = ModelWrapper;

private:
  TupleTree<model::Binary> InternalModel;
  bool External;
  ModelWrapper Wrapper;

private:
  LoadModelAnalysis() : External(false), Wrapper(ModelWrapper(InternalModel)) {}
  LoadModelAnalysis(const ModelWrapper &Wrapper) :
    External(true), Wrapper(Wrapper) {}

public:
  static LoadModelAnalysis fromModule() { return {}; }
  static LoadModelAnalysis fromModelWrapper(const ModelWrapper &Wrapper) {
    return { Wrapper };
  }

public:
  bool isModelExternal() const { return External; }

public:
  ModelWrapper run(llvm::Module &M, llvm::ModuleAnalysisManager &);
  ModelWrapper run(llvm::Function &F, llvm::FunctionAnalysisManager &);
};
