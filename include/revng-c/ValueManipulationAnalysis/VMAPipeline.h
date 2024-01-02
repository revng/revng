#pragma once
//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

#include "llvm/ADT/SmallVector.h"

#include "revng/EarlyFunctionAnalysis/FunctionMetadataCache.h"
#include "revng/Model/Binary.h"
#include "revng/Model/QualifiedType.h"

namespace vma {
struct TypeFlowGraph;
}

namespace llvm {
class Function;
}

// --------------- Initializers

/// Assign the initial colors of nodes in \a TFG.
/// NOTE: Initializers are expected to be only incremental, i.e. the only add
/// information about nodes that are not already colored.
class VMAInitializer {
public:
  virtual ~VMAInitializer(){};
  virtual void initializeColors(vma::TypeFlowGraph *TFG) = 0;
};

/// Initializes the colors of \a TFG from model types
class TypeMapInitializer : public VMAInitializer {
public:
  using TypeMapT = std::map<const llvm::Value *, const model::QualifiedType>;

private:
  const TypeMapT &TypeMap;

public:
  TypeMapInitializer(const TypeMapT &TypeMap) : TypeMap(TypeMap) {}

public:
  void initializeColors(vma::TypeFlowGraph *TFG) override;
};

/// Initializes the colors of \a TFG by looking at the LLVM IR
class LLVMInitializer : public VMAInitializer {
public:
  void initializeColors(vma::TypeFlowGraph *TFG) override;
};

// --------------- Updaters

/// Use the types recovered in \a TFG to update an external repository of type
/// information
class VMAUpdater {
public:
  virtual ~VMAUpdater(){};
  virtual void updateWithResults(const vma::TypeFlowGraph *TFG) = 0;
};

/// Update the TypeMap with the types recovered in \a TFG
class TypeMapUpdater : public VMAUpdater {
private:
  using TypeMapT = std::map<const llvm::Value *, const model::QualifiedType>;

private:
  TypeMapT &TypeMap;
  const model::Binary *Model;

public:
  TypeMapUpdater(TypeMapT &TypeMap, const model::Binary *Model) :
    VMAUpdater(), TypeMap(TypeMap), Model(Model) {}

public:
  void updateWithResults(const vma::TypeFlowGraph *TFG) override;
};

// --------------- Pipeline

/// Engine for propagating type-flow information on a TypeFlowGraph
class VMAPipeline {
private:
  vma::TypeFlowGraph *TFG;

  const model::Binary &Model;

private:
  /// Initializers kickstart the VMA process by coloring some of the nodes of
  /// the TFG. They are expected to work incrementally, so the order in which
  /// initializers are added to this vector matters.
  llvm::SmallVector<std::unique_ptr<VMAInitializer>, 2> Initializers;
  /// The Updater finalizes the VMA process. It typically extracts information
  /// from the \a TFG and uses it to update an external type repository.
  std::unique_ptr<VMAUpdater> Updater;
  /// If this flag is set to true, the pipeline will try to assign exactly one
  /// color to each node.
  bool UseSolver;

private:
  void runSolver();

public:
  VMAPipeline(const model::Binary &Model) : Model(Model) {}

public:
  void addInitializer(std::unique_ptr<VMAInitializer> Initializer) {
    Initializers.push_back(std::move(Initializer));
  }

  void setUpdater(std::unique_ptr<VMAUpdater> Updater) {
    this->Updater = std::move(Updater);
  }

public:
  /// Also run the constraint-solving pass before finalizing.
  /// WARNING: The solver can be expensive.
  void enableSolver() { UseSolver = true; }
  void disableSolver() { UseSolver = false; }
  bool isSolverEnabled() { return UseSolver; }

public:
  void run(FunctionMetadataCache &Cache, const llvm::Function *F);
};
