#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Pass.h"

#include "revng/DataLayoutAnalysis/DLALayouts.h"
#include "revng/DataLayoutAnalysis/DLATypeSystem.h"
#include "revng/Model/Binary.h"

#include "../FuncOrCallInst.h"

namespace dla {

using LayoutTypePtrVec = std::vector<LayoutTypePtr>;

/// This class is used to print LLVM information when debugging the TS
///
/// Since nodes in the TypeSystem graph only have IDs, which are grouped into
/// equivalence classes, if we want to track each ID back to the original LLVM
/// Value when printing we need to define a special DebugPrinter, that knows
/// which Value is mapped to each ID.
class LLVMTSDebugPrinter : public TSDebugPrinter {
protected:
  const llvm::Module &M;
  const LayoutTypePtrVect &Values;

public:
  /// Build the Debug printer
  ///
  ///\param M The LLVM module from which the TS graph was built
  ///\param Values Ordered vector, Values are indexed with the ID of the
  ///              corresponding TypeSystemNode
  LLVMTSDebugPrinter(const llvm::Module &M, const LayoutTypePtrVect &Values) :
    M(M), Values(Values) {}

  LLVMTSDebugPrinter() = delete;

public:
  /// Print the `llvm::Value`s collapsed inside \a N
  void printNodeContent(const LayoutTypeSystem &TS,
                        const LayoutTypeSystemNode *N,
                        llvm::raw_fd_ostream &DotFile) const override;
};

/// This class builds a DLA type system from an LLVM module
class DLATypeSystemLLVMBuilder {
public:
  using VisitedMapT = std::map<LayoutTypePtr, LayoutTypeSystemNode *>;
  using PrototypesMapT = std::map<const model::TypeDefinition *,
                                  FuncOrCallInst>;

private:
  /// Separate class that add `Instance` edges
  class InstanceLinkAdder;

  /// The TypeSystem to build
  LayoutTypeSystem &TS;

  /// Ordered vector, each element is indexed with the ID of the
  /// corresponding Node
  LayoutTypePtrVect Values;

  /// Reverse map between `llvm::Value`s and Nodes
  VisitedMapT VisitedValues;
  /// Associate each indirect call's prototype in the model with the
  /// first `llvm::CallInst` found with that prototype,
  PrototypesMapT VisitedPrototypes;

private:
  LayoutTypeSystemNode *getLayoutType(const llvm::Value *V, unsigned Id);

  LayoutTypeSystemNode *getLayoutType(const llvm::Value *V) {
    return getLayoutType(V, std::numeric_limits<unsigned>::max());
  };

  std::pair<LayoutTypeSystemNode *, bool>
  getOrCreateLayoutType(const llvm::Value *V, unsigned Id);

  std::pair<LayoutTypeSystemNode *, bool>
  getOrCreateLayoutType(const llvm::Value *V) {
    return getOrCreateLayoutType(V, std::numeric_limits<unsigned>::max());
  }

  llvm::SmallVector<LayoutTypeSystemNode *, 2>
  getLayoutTypes(const llvm::Value &V);

  llvm::SmallVector<std::pair<LayoutTypeSystemNode *, bool>, 2>
  getOrCreateLayoutTypes(const llvm::Value &V);

private:
  bool createInterproceduralTypes(llvm::Module &M, const model::Binary &Model);
  bool createIntraproceduralTypes(llvm::Module &M,
                                  llvm::ModulePass *MP,
                                  const model::Binary &Model);

  /// Collect LayoutTypePtrs and place them in the right position
  void createValuesList();

public:
  LayoutTypePtrVect &getValues() { return Values; }

  /// Print a `.csv` with the mapping between nodes and `llvm::Value`s
  ///
  /// The mapping is reconstructed on-the-fly, therefore is expensive. The
  /// generated .csv uses *semicolons* as separators.
  void debug_function dumpValuesMapping(const llvm::StringRef Name) const;

public:
  DLATypeSystemLLVMBuilder(LayoutTypeSystem &TS) : TS(TS){};

  /// Create a DLATypeSystem graph for a given LLVM module
  ///
  /// LayoutTypePtrs represent elements of the LLVM IR that are thought to be
  /// possible pointers. The builder's job is to:
  /// 1. Identify such LayoutTypePtrs
  /// 2. Create a Node for each of them in the DLATypeSystem graph (TS)
  /// 3. Keep an ordered vector of LayoutTypePtrs, where each element's index
  /// corresponds to the ID of the corresponding LayoutTypeSystemNode generated
  void buildFromLLVMModule(llvm::Module &M,
                           llvm::ModulePass *MP,
                           const model::Binary &Model);

  /// Given an indirect Call instruction, check if it shares the model
  /// prototype with another function. If it does, connect the nodes
  /// corresponding to the types of the return values and arguments of this Call
  /// with the types of the return values and arguments of the function sharing
  /// the same prototype, using equality links.
  bool connectToFuncsWithSamePrototype(const llvm::CallInst *Call,
                                       const model::Binary &Model);
};

} // namespace dla
