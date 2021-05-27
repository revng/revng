#pragma once

//
// Copyright (c) rev.ng Srls. See LICENSE.md for details.
//

#include "llvm/Pass.h"

#include "revng/Support/Assert.h"

#include "revng-c/Decompiler/DLALayouts.h"

#include "DLATypeSystem.h"

namespace dla {

///\brief This class is used to print LLVM information when debugging the TS
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
  ///\brief Build the Debug printer
  ///
  ///\param M The LLVM module from which the TS graph was built
  ///\param Values Ordered vector, Values are indexed with the ID of the
  ///              corresponding TypeSystemNode
  LLVMTSDebugPrinter(const llvm::Module &M, const LayoutTypePtrVect &Values) :
    M(M), Values(Values) {}

  LLVMTSDebugPrinter() = delete;

public:
  ///\brief Print the `llvm::Value`s collapsed inside \a N
  void printNodeContent(const LayoutTypeSystem &TS,
                        const LayoutTypeSystemNode *N,
                        llvm::raw_fd_ostream &DotFile) const override;

  ///\brief Print the instruction that originated a given \a AccessSize of \a N
  ///
  /// Information on the `load`/`store`s related to a given set of `Value`s is
  /// reconstructed on-the-fly, therefore this function is expensive.
  void printAccessDetails(const LayoutTypeSystem &TS,
                          const LayoutTypeSystemNode *N,
                          const uint64_t AccessSize,
                          llvm::raw_fd_ostream &DotFile) const override;
};

///\brief This class builds a DLA type system from an LLVM module
class DLATypeSystemLLVMBuilder {
public:
  using VisitedMapT = std::map<LayoutTypePtr, LayoutTypeSystemNode *>;

private:
  ///\brief Separate class that add `Instance` edges
  class InstanceLinkAdder;

  ///\brief The TypeSystem to build
  LayoutTypeSystem &TS;

  ///\brief Ordered vector, each element is indexed with the ID of the
  /// corresponding Node
  LayoutTypePtrVect Values;

  ///\brief Reverse map between `llvm::Value`s and Nodes
  VisitedMapT VisitedMap;

private:
  void assertGetLayoutTypePreConditions(const llvm::Value *V, unsigned Id);
  void assertGetLayoutTypePreConditions(const llvm::Value &V);

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
  bool createInterproceduralTypes(llvm::Module &M);
  bool createIntraproceduralTypes(llvm::Module &M, llvm::ModulePass *MP);

  ///\brief Collect LayoutTypePtrs and place them in the right position
  void createValuesList();

public:
  LayoutTypePtrVect &getValues() { return Values; }

  ///\brief Print a `.csv` with the mapping between nodes and `llvm::Value`s
  ///
  /// The mapping is reconstructed on-the-fly, therefore is expensive. The
  /// generated .csv uses _semicolons_ as separators.
  void dumpValuesMapping(const llvm::StringRef Name);

public:
  DLATypeSystemLLVMBuilder(LayoutTypeSystem &TS) : TS(TS){};

  ///\brief Create a DLATypeSystem graph for a given LLVM module
  ///
  /// LayoutTypePtrs represent elements of the LLVM IR that are thought to be
  /// possible pointers. The builder's job is to:
  /// 1. Identify such LayoutTypePtrs
  /// 2. Create a Node for each of them in the DLATypeSystem graph (TS)
  /// 3. Keep an ordered vector of LayoutTypePtrs, where each element's index
  /// corresponds to the ID of the corresponding LayoutTypeSystemNode generated
  void buildFromLLVMModule(llvm::Module &M, llvm::ModulePass *MP);
};

} // namespace dla