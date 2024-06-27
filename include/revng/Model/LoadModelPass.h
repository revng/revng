#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <variant>

#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"

#include "revng/Model/Binary.h"
#include "revng/TupleTree/TupleTree.h"

extern llvm::cl::opt<std::string> ModelPath;

inline const char *ModelMetadataName = "revng.model";

TupleTree<model::Binary> loadModel(const llvm::Module &M);

bool hasModel(const llvm::Module &M);

class ModelWrapper {
private:
  using ModelPointer = std::variant<TupleTree<model::Binary> *,
                                    const TupleTree<model::Binary> *>;

private:
  ModelPointer TheBinary;
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
  const TupleTree<model::Binary> &getReadOnlyModel() const;

  TupleTree<model::Binary> &getWriteableModel();

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
  ModelWrapper Wrapper;

public:
  LoadModelWrapperPass(const ModelWrapper &Wrapper) :
    llvm::ImmutablePass(ID), Wrapper(Wrapper) {}

public:
  bool doInitialization(llvm::Module &M) override final;
  bool doFinalization(llvm::Module &M) override final;

public:
  ModelWrapper &get() { return Wrapper; }
};

class LoadModelAnalysis : public llvm::AnalysisInfoMixin<LoadModelAnalysis> {
  friend llvm::AnalysisInfoMixin<LoadModelAnalysis>;

private:
  static llvm::AnalysisKey Key;

public:
  using Result = ModelWrapper;

private:
  ModelWrapper Wrapper;

private:
  LoadModelAnalysis(const ModelWrapper &Wrapper) : Wrapper(Wrapper) {}

public:
  static LoadModelAnalysis fromModelWrapper(const ModelWrapper &Wrapper) {
    return { Wrapper };
  }

public:
  ModelWrapper run(llvm::Module &M, llvm::ModuleAnalysisManager &);
  ModelWrapper run(llvm::Function &F, llvm::FunctionAnalysisManager &);
};
