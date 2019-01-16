#ifndef INTERPROCEDURAL_H
#define INTERPROCEDURAL_H

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <map>
#include <set>
#include <vector>

// LLVM includes
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/User.h"
#include "llvm/Support/Casting.h"

// Local libraries includes
#include "revng/Support/Debug.h"
#include "revng/Support/IRHelpers.h"

// Local includes
#include "Intraprocedural.h"

/// \brief Logger for messages concerning the interprocedural analysis
extern Logger<> SaInterpLog;

namespace StackAnalysis {

/// \brief Class to collect the (partial) results of all the analyses
///
/// Tracks register arguments and return values of functions and function calls,
/// the size of the stack at each call site, the type of each basic block
/// (branch) and the type of each function.
///
/// The information stored here is "raw", as opposed to that stored in
/// `FunctionsSummary` which, e.g., merges the information available on a
/// function with the information coming from all of the call sites targeting
/// it.
class ResultsPool {
  friend struct ClobberedRegistersAnalysis;

public:
  using BasicBlock = llvm::BasicBlock;

  template<typename K, typename V>
  using map = std::map<K, V>;

  using BasicBlockTypeMap = map<BasicBlock *, BranchType::Values>;
  using StackSizeMap = map<FunctionCall, llvm::Optional<int32_t>>;

private:
  using FunctionSlot = std::pair<BasicBlock *, int32_t>;
  using FunctionCallSlot = std::pair<CallSite, int32_t>;

private:
  // TODO: maps with the same keys could be merged

  // Data about functions
  map<FunctionSlot, FunctionRegisterArgument> FunctionRegisterArguments;
  map<FunctionSlot, FunctionReturnValue> FunctionReturnValues;

  // Data about function calls
  using FCS = FunctionCallSlot;
  map<FCS, FunctionCallRegisterArgument> FunctionCallRegisterArguments;
  map<FCS, FunctionCallReturnValue> FunctionCallReturnValues;

  /// \brief Height of the stack at each call site
  map<CallSite, llvm::Optional<int32_t>> CallSites;

  /// \brief Classification of each branch
  map<Branch, BranchType::Values> BranchesType;

  /// \brief Classification of each function
  map<BasicBlock *, FunctionType::Values> FunctionTypes;

  map<BasicBlock *, std::set<int32_t>> LocallyWrittenRegisters;
  map<BasicBlock *, std::set<int32_t>> ExplicitlyCalleeSavedRegisters;
  map<BasicBlock *, std::vector<FunctionCall>> FunctionCalls;

public:
  /// \brief Register a function for which a summary is not available
  void registerFunction(llvm::BasicBlock *Function, FunctionType::Values Type) {
    FunctionTypes[Function] = Type;
  }

  void registerFunction(llvm::BasicBlock *Entry,
                        FunctionType::Values Type,
                        const IntraproceduralFunctionSummary &Summary) {
    registerFunction(Entry, Type);
    mergeCallSites(Entry, Summary.FrameSizeAtCallSite);
    mergeBranches(Entry, Summary.BranchesType);
    if (Type == FunctionType::Regular or Type == FunctionType::NoReturn
        or Type == FunctionType::IndirectTailCall)
      mergeFunction(Entry, Summary);
  }

  /// \brief Merge data about \p Function in \p Summary into the results pool
  ///
  /// \param Function the entry basic block of the function we're currently
  ///        considering
  /// \param Summary the final summary of the function from which the data about
  ///        registers and return values will be fetched
  void mergeFunction(llvm::BasicBlock *Function,
                     const IntraproceduralFunctionSummary &Summary);

  /// \brief Merge data about the classification of a set of branches in \p
  ///        Function
  void
  mergeBranches(llvm::BasicBlock *Function, const BasicBlockTypeMap &Branches);

  /// \brief Merge information about the height of the stack at the call sites
  ///        of \p Function
  void mergeCallSites(llvm::BasicBlock *Function, const StackSizeMap &ToImport);

  /// \brief Finalized the data stored in this object and produce a
  ///        FunctionsSummary
  FunctionsSummary finalize(llvm::Module *M);

  void dump(const llvm::Module *M) const debug_function { dump(M, dbg); }

  template<typename T>
  void dump(const llvm::Module *M, T &Output) const {
    Output << "FunctionRegisterArguments:\n";
    for (auto &P : FunctionRegisterArguments) {
      Output << getName(P.first.first) << " ";
      ASSlot::create(ASID::cpuID(), P.first.second).dump(M, Output);
      Output << ": ";
      P.second.dump(Output);
      Output << "\n";
    }
    Output << "\n";

    Output << "FunctionReturnValues:\n";
    for (auto &P : FunctionReturnValues) {
      Output << getName(P.first.first) << " ";
      ASSlot::create(ASID::cpuID(), P.first.second).dump(M, Output);
      Output << ": ";
      P.second.dump(Output);
      Output << "\n";
    }
    Output << "\n";

    Output << "FunctionCallRegisterArguments:\n";
    for (auto &P : FunctionCallRegisterArguments) {
      P.first.first.dump(Output);
      Output << " ";
      ASSlot::create(ASID::cpuID(), P.first.second).dump(M, Output);
      Output << ": ";
      P.second.dump(Output);
      Output << "\n";
    }
    Output << "\n";

    Output << "FunctionCallReturnValues:\n";
    for (auto &P : FunctionCallReturnValues) {
      P.first.first.dump(Output);
      Output << " ";
      ASSlot::create(ASID::cpuID(), P.first.second).dump(M, Output);
      Output << ": ";
      P.second.dump(Output);
      Output << "\n";
    }
    Output << "\n";
  }

  /// \brief Build a set of all the `BasicBlock`s that have been visited so far
  std::set<llvm::BasicBlock *> visitedBlocks() const {
    std::set<llvm::BasicBlock *> Result;
    for (auto &P : BranchesType)
      Result.insert(P.first.branch()->getParent());
    return Result;
  }
};

/// \brief Interprocedural part of the stack analysis
class InterproceduralAnalysis {
private:
  using Analysis = Intraprocedural::Analysis;

private:
  Cache &TheCache;
  GeneratedCodeBasicInfo &GCBI;
  std::vector<Analysis> InProgress;
  std::set<llvm::BasicBlock *> InProgressFunctions; ///< For recursion detection
  bool AnalyzeABI;

public:
  InterproceduralAnalysis(Cache &TheCache,
                          GeneratedCodeBasicInfo &GCBI,
                          bool AnalyzeABI) :
    TheCache(TheCache),
    GCBI(GCBI),
    AnalyzeABI(AnalyzeABI) {}

  void run(llvm::BasicBlock *Entry, ResultsPool &Results);

private:
  void push(llvm::BasicBlock *Entry);

  void popUntil(const Analysis *WI) {
    while (&InProgress.back() != WI)
      pop();
  }

  const Analysis *getRecursionRoot(llvm::BasicBlock *Entry) const {
    for (const Analysis &WI : InProgress)
      if (WI.entry() == Entry)
        return &WI;

    return nullptr;
  }

  void pop() {
    InProgressFunctions.erase(InProgress.back().entry());
    InProgress.pop_back();
  }
};

} // namespace StackAnalysis

#endif // INTERPROCEDURAL_H
