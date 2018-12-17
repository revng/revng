#ifndef INTRAPROCEDURAL_H
#define INTRAPROCEDURAL_H

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <array>
#include <map>
#include <set>
#include <utility>
#include <vector>

// LLVM includes
#include "llvm/ADT/SmallVector.h"

// Local libraries includes
#include "revng/BasicAnalyses/GeneratedCodeBasicInfo.h"
#include "revng/Support/Debug.h"
#include "revng/Support/IRHelpers.h"
#include "revng/Support/MonotoneFramework.h"

// Local includes
#include "ABIIR.h"
#include "Element.h"
#include "FunctionABI.h"
#include "IntraproceduralFunctionSummary.h"

template<typename T>
inline bool compareOptional(llvm::Optional<T> LHS, llvm::Optional<T> RHS) {
  return LHS.hasValue() == RHS.hasValue() && (!LHS.hasValue() || *LHS == *RHS);
}

namespace StackAnalysis {

class Cache;

/// \brief Copy of \p I, i.e., a container of classes with no copy constructor,
///        but having a .copy() method
template<typename T>
inline T copyContainer(const T &I) {
  T Result;
  Result.reserve(I.size());
  for (auto &V : I)
    Result.push_back(V.copy());

  return Result;
}

namespace Intraprocedural {

/// \brief Result of the transfer function
///
/// This class represents the result of the transfer function, it might simply
/// represent the result of the transfer functions starting from the initial
/// state or more sophisticated situations, e.g., function calls that have to be
/// handled by the intraprocedural part of the analysis.
class Interrupt {
public:
  using vector = llvm::SmallVector<llvm::BasicBlock *, 2>;
  using iterator = typename vector::iterator;
  using const_iterator = typename vector::const_iterator;
  using iterator_range = typename llvm::iterator_range<iterator>;
  using const_iterator_range = typename llvm::iterator_range<const_iterator>;

private:
  bool ResultExtracted;

  BranchType::Values Type;

  Element Result;
  vector RelatedBasicBlocks;
  IntraproceduralFunctionSummary Summary;

private:
  Interrupt() :
    ResultExtracted(false),
    Type(BranchType::Invalid),
    Result(Element::bottom()),
    Summary(IntraproceduralFunctionSummary::bottom()) {}

  Interrupt(Element Result, BranchType::Values Type) :
    ResultExtracted(false),
    Type(Type),
    Result(std::move(Result)),
    Summary(IntraproceduralFunctionSummary::bottom()) {}

  Interrupt(BranchType::Values Type, IntraproceduralFunctionSummary Summary) :
    ResultExtracted(false),
    Type(Type),
    Result(Element::bottom()),
    Summary(std::move(Summary)) {}

  Interrupt(Element Result, BranchType::Values Type, vector Successors) :
    ResultExtracted(false),
    Type(Type),
    Result(std::move(Result)),
    RelatedBasicBlocks(Successors),
    Summary(IntraproceduralFunctionSummary::bottom()) {}

  Interrupt(BranchType::Values Type, vector Successors) :
    ResultExtracted(false),
    Type(Type),
    Result(Element::bottom()),
    RelatedBasicBlocks(Successors),
    Summary(IntraproceduralFunctionSummary::bottom()) {}

public:
  static Interrupt createInvalid() { return Interrupt(); };

  static Interrupt createWithSuccessor(Element Result,
                                       BranchType::Values Type,
                                       llvm::BasicBlock *Successor) {
    revng_assert(Type == BranchType::FakeFunctionCall
                 || Type == BranchType::HandledCall
                 || Type == BranchType::IndirectCall
                 || Type == BranchType::FakeFunctionReturn);

    return Interrupt(std::move(Result), Type, { Successor });
  }

  static Interrupt createWithSuccessors(Element Result,
                                        BranchType::Values Type,
                                        vector Successors) {
    revng_assert(Type == BranchType::InstructionLocalCFG
                 || Type == BranchType::FunctionLocalCFG);
    revng_assert(Successors.size() > 0);

    return Interrupt(std::move(Result), Type, Successors);
  }

  static Interrupt create(Element Result, BranchType::Values Type) {
    using namespace BranchType;
    revng_assert(Type == Return || Type == IndirectTailCall
                 || Type == FakeFunction || Type == LongJmp || Type == Killer
                 || Type == Unreachable || Type == NoReturnFunction);

    return Interrupt(std::move(Result), Type);
  }

  static Interrupt createUnhandledCall(llvm::BasicBlock *Callee) {
    return Interrupt(BranchType::UnhandledCall, { Callee });
  }

  static Interrupt createSummary(IntraproceduralFunctionSummary Summary) {
    return Interrupt(BranchType::FunctionSummary, std::move(Summary));
  }

  static Interrupt
  createNoReturnFunction(IntraproceduralFunctionSummary Summary) {
    return Interrupt(BranchType::NoReturnFunction, std::move(Summary));
  }

  static Interrupt
  createIndirectTailCallFunction(IntraproceduralFunctionSummary Summary) {
    return Interrupt(BranchType::IndirectTailCallFunction, std::move(Summary));
  }

public:
  /// \brief True if this result has successors
  bool hasSuccessors() const {
    switch (Type) {
    case BranchType::InstructionLocalCFG:
    case BranchType::FunctionLocalCFG:
    case BranchType::FakeFunctionCall:
    case BranchType::FakeFunctionReturn:
    case BranchType::HandledCall:
    case BranchType::IndirectCall:
      return true;
    case BranchType::UnhandledCall:
    case BranchType::Return:
    case BranchType::IndirectTailCall:
    case BranchType::FakeFunction:
    case BranchType::LongJmp:
    case BranchType::Killer:
    case BranchType::Unreachable:
      return false;
    case BranchType::Invalid:
    case BranchType::FunctionSummary:
    case BranchType::NoReturnFunction:
    case BranchType::IndirectTailCallFunction:
      revng_abort();
    }

    revng_abort();
  }

  /// \brief Is this a regular function terminator?
  ///
  /// \note This function doesn't tell you if this basic block has no
  ///       successors, but if this is a proper return whose final result should
  ///       be considered in the computation of the final result of the function
  ///       as a whole. Abnormal exits (such as `noreturn` function calls)
  ///       return false.
  bool isFinalState() const {
    switch (Type) {
    case BranchType::Return:
    case BranchType::IndirectTailCall:
      // TODO: an IndirectTailCall, *can be* an IndirectTail call, but it might
      //       also just be a longjmp. So it's an assumption that this is a
      //       final state.
      return true;
    case BranchType::InstructionLocalCFG:
    case BranchType::FunctionLocalCFG:
    case BranchType::FakeFunctionCall:
    case BranchType::FakeFunctionReturn:
    case BranchType::HandledCall:
    case BranchType::IndirectCall:
    case BranchType::UnhandledCall:
    case BranchType::FakeFunction:
    case BranchType::LongJmp:
    case BranchType::Killer:
    case BranchType::Unreachable:
      return false;
    case BranchType::Invalid:
    case BranchType::FunctionSummary:
    case BranchType::NoReturnFunction:
    case BranchType::IndirectTailCallFunction:
      revng_abort();
    }

    revng_abort();
  }

  BranchType::Values type() const { return Type; }

  bool isReturn() const { return Type == BranchType::Return; }

  bool requiresInterproceduralHandling() const {
    switch (Type) {
    case BranchType::FakeFunction:
    case BranchType::UnhandledCall:
    case BranchType::FunctionSummary:
    case BranchType::NoReturnFunction:
    case BranchType::IndirectTailCallFunction:
      return true;
    case BranchType::InstructionLocalCFG:
    case BranchType::FunctionLocalCFG:
    case BranchType::FakeFunctionCall:
    case BranchType::FakeFunctionReturn:
    case BranchType::HandledCall:
    case BranchType::IndirectCall:
    case BranchType::Return:
    case BranchType::IndirectTailCall:
    case BranchType::LongJmp:
    case BranchType::Killer:
    case BranchType::Unreachable:
      return false;
    case BranchType::Invalid:
      revng_abort();
    }

    revng_abort();
  }

  Element &&extractResult() {
    revng_assert(Type != BranchType::FunctionSummary
                 and Type != BranchType::UnhandledCall);

    revng_assert(not ResultExtracted);
    ResultExtracted = true;
    return std::move(Result);
  }

  llvm::BasicBlock *getCallee() const {
    revng_assert(Type == BranchType::UnhandledCall);
    revng_assert(RelatedBasicBlocks.size() == 1);
    return RelatedBasicBlocks[0];
  }

  const IntraproceduralFunctionSummary &getFunctionSummary() {
    // TODO: is it OK for fake functions to have summaries?
    revng_assert(Type == BranchType::FunctionSummary
                 || Type == BranchType::FakeFunction
                 || Type == BranchType::NoReturnFunction
                 || Type == BranchType::IndirectTailCallFunction);
    return Summary;
  }

  const_iterator begin() { return RelatedBasicBlocks.begin(); }
  const_iterator end() { return RelatedBasicBlocks.end(); }
  size_t size() const {
    revng_assert(hasSuccessors());
    return RelatedBasicBlocks.size();
  }

  void dump(const llvm::Module *M) const debug_function { dump(M, dbg); }

  template<typename T>
  void dump(const llvm::Module *M, T &Output) const {
    Output << "Interrupt reason: " << BranchType::getName(Type) << "\n";
    switch (Type) {
    case BranchType::InstructionLocalCFG:
    case BranchType::FunctionLocalCFG:
    case BranchType::FakeFunctionCall:
    case BranchType::FakeFunctionReturn:
    case BranchType::HandledCall:
    case BranchType::IndirectCall:
      Output << "Successors:";
      for (llvm::BasicBlock *BB : RelatedBasicBlocks)
        Output << " " << getName(BB);
      Output << "\n";
      Output << "Result:\n";
      Result.dump(M);
      break;
    case BranchType::UnhandledCall:
      Output << "Unhandled call to " << getName(RelatedBasicBlocks[0]) << "\n";
      break;
    case BranchType::FunctionSummary:
      Output << "Summary:\n";
      Summary.dump(M);

      break;
    case BranchType::Return:
    case BranchType::NoReturnFunction:
    case BranchType::IndirectTailCall:
    case BranchType::FakeFunction:
    case BranchType::LongJmp:
    case BranchType::Killer:
      Output << "Result:\n";
      Result.dump(M);
      break;
    case BranchType::Invalid:
    case BranchType::IndirectTailCallFunction:
    case BranchType::Unreachable:
      revng_abort();
    }
  }
};

/// \brief Intraprocedural part of the stack analysis
class Analysis : public MonotoneFramework<llvm::BasicBlock *,
                                          Element,
                                          Interrupt,
                                          Analysis,
                                          Interrupt::const_iterator_range,
                                          BreadthFirst,
                                          true> {
private:
  // Label: llvm::BasicBlock *
  // LatticeElement: Element
  // Interrupt: Interrupt
  // D (derived class): Analysis
  // SuccessorsRange: Interrupt::const_iterator_range
  // Visit: BreadthFirst
  // DynamicGraph: true
  using Base = MonotoneFramework<llvm::BasicBlock *,
                                 Element,
                                 Interrupt,
                                 Analysis,
                                 Interrupt::const_iterator_range,
                                 BreadthFirst,
                                 true>;

private:
  llvm::BasicBlock *Entry; ///< The entry point of the current function
  const llvm::Module *M;
  const Cache *TheCache; ///< Reference to the Cache (for query purposes)
  ASSlot ReturnAddressSlot; ///< Slot that contains the return address
  GeneratedCodeBasicInfo *GCBI;
  Element InitialState; ///< Empty Element with stack pointer initialized
  int32_t SPIndex; ///< Offset of the stack pointer CSV
  int32_t PCIndex; ///< Offset of the PC CSV
  ABIFunction TheABIIR; ///< The ABI IR
  int32_t CSVCount; ///< Number of CSVs, used to distinguish from alloca

  /// \brief Set of return addresses from fake function calls
  std::set<uint64_t> FakeReturnAddresses;

  /// \brief Branches list and classification
  std::map<llvm::BasicBlock *, BranchType::Values> BranchesType;

  std::map<llvm::Instruction *, Value> VariableContent; ///< Content of allocas

  /// This flag is set if the last time we interrupted the analysis was due to
  /// an unhandled function call, which should then result in a cache hit
  bool CacheMustHit;

  /// \brief Set of functions currently being analyzed, for recursion detection
  ///        purposes
  const std::set<llvm::BasicBlock *> &InProgressFunctions;

  /// \brief Record all call sites and the associated stack size
  std::map<FunctionCall, llvm::Optional<int32_t>> FrameSizeAtCallSite;

  /// \brief Called functions that have been found incoherent with the caller
  std::set<llvm::BasicBlock *> IncoherentFunctions;

  bool AnalyzeABI;

  std::map<const llvm::User *, int32_t> CPUIndices;

public:
  Analysis(llvm::BasicBlock *Entry,
           const Cache &TheCache,
           GeneratedCodeBasicInfo *GCBI,
           const std::set<llvm::BasicBlock *> &InProgressFunctions,
           bool AnalyzeABI) :
    Base(Entry),
    Entry(Entry),
    M(getModule(Entry)),
    TheCache(&TheCache),
    ReturnAddressSlot(ASSlot::invalid()),
    GCBI(GCBI),
    InitialState(Element::bottom()),
    TheABIIR(Entry),
    InProgressFunctions(InProgressFunctions),
    AnalyzeABI(AnalyzeABI) {

    registerExtremal(Entry);
    initialize();
  }

  bool isCSV(ASSlot Slot) const {
    return Slot.addressSpace() == ASID::cpuID() && Slot.offset() < CSVCount;
  }

  void assertLowerThanOrEqual(const Element &A, const Element &B) const {
    ::StackAnalysis::assertLowerThanOrEqual(A, B, getModule(Entry));
  }

  llvm::Optional<Element> handleEdge(const Element &Original,
                                     llvm::BasicBlock *Source,
                                     llvm::BasicBlock *Destination) const {
    return llvm::Optional<Element>();
  }

  llvm::BasicBlock *entry() const { return Entry; }

  void resetCacheMustHit() { CacheMustHit = false; }

  bool cacheMustHit() const { return CacheMustHit; }

  /// \brief Reset the analysis with a new intial state
  void initialize();

  /// \brief Return the stack size of \p Result, if available
  llvm::Optional<int32_t> stackSize(Element &Result) const {
    Value StackPointer = Value::fromSlot(ASID::cpuID(), SPIndex);
    ASID StackID = ASID::stackID();

    // Save the value of the stack pointer for later
    Value OldStackPointer = Result.load(StackPointer);

    llvm::Optional<int32_t> CallerStackSize;
    if (const ASSlot *OldStackPointerSlot = OldStackPointer.directContent()) {
      if (OldStackPointerSlot->addressSpace() != StackID)
        return llvm::Optional<int32_t>();
      return -OldStackPointerSlot->offset();
    } else {
      return llvm::Optional<int32_t>();
    }
  }

  /// \brief Register the stack size at call site \p TheFunctionCall
  ///
  /// \return false if the stack size is different from the one that was
  ///         previously recorded, if any.
  bool registerStackSizeAtCallSite(FunctionCall TheFunctionCall,
                                   llvm::Optional<int32_t> StackSize) {
    auto It = FrameSizeAtCallSite.find(TheFunctionCall);
    if (It != FrameSizeAtCallSite.end()) {
      if (not compareOptional(It->second, StackSize)) {
        It->second = llvm::Optional<int32_t>();
        return false;
      }
    } else {
      FrameSizeAtCallSite[TheFunctionCall] = StackSize;
    }

    return true;
  }

  /// \brief If available, return the registered size of the call site
  ///        \p Location
  llvm::Optional<int32_t> frameSizeAt(FunctionCall Location) const {
    auto It = FrameSizeAtCallSite.find(Location);
    revng_assert(It != FrameSizeAtCallSite.end(),
                 "Location has never been registered");
    return It->second;
  }

  /// \brief The almighty transfer function
  Interrupt transfer(llvm::BasicBlock *BB);

  /// \brief The extremal value, i.e., the context of the analysis
  Element extremalValue(llvm::BasicBlock *) const {
    return InitialState.copy();
  }

  void dumpFinalState() const {
    if (SaLog.isEnabled()) {
      SaLog << "FinalResult:\n";
      FinalResult.dump(getModule(Entry), SaLog);
      SaLog << DoLog;
    }
  }

  Interrupt::const_iterator_range
  successors(llvm::BasicBlock *, Interrupt &I) const {
    return llvm::make_range(I.begin(), I.end());
  }

  size_t successor_size(llvm::BasicBlock *, Interrupt &I) const {
    if (I.hasSuccessors())
      return I.size();
    else
      return 0;
  }

  Interrupt createSummaryInterrupt() {
    return Interrupt::createSummary(createSummary());
  }

  Interrupt createNoReturnInterrupt() {
    if (hasIndirectTailCall())
      return Interrupt::createIndirectTailCallFunction(createSummary());
    else
      return Interrupt::createNoReturnFunction(createSummary());
  }

  /// \brief Return the set of functions called by this function in an
  ///        incoherent way
  ///
  /// This function returns the set of basic blocks representing functions
  /// called by the current function for which the information obtained about a
  /// call site isn't compatible with the information obtained by analysing the
  /// callee.
  const std::set<llvm::BasicBlock *> &incoherentFunctions() const {
    return IncoherentFunctions;
  }

private:
  /// \brief Creates a summary for the current analysis ready to be wrapped in
  ///        an Interrupt
  IntraproceduralFunctionSummary createSummary();

  /// \brief Check whether the ABI analysis results for a slot of the function
  ///        and a call site are compatible
  bool isCoherent(const FunctionABI &CallerSummary,
                  const FunctionABI &CalleeSummary,
                  FunctionCall TheFunctionCall,
                  IntraproceduralFunctionSummary::LocalSlot Slot) const;

  /// \brief Populate IncoherentFunctions
  void
  findIncoherentFunctions(const IntraproceduralFunctionSummary &ABISummary);

  /// \brief Part of the transfer function handling terminator instructions
  Interrupt handleTerminator(llvm::TerminatorInst *T,
                             Element &Result,
                             ABIIRBasicBlock &ABIBB);

  /// \brief Part of the transfer function handling function calls
  Interrupt handleCall(llvm::Instruction *Caller,
                       llvm::BasicBlock *Callee,
                       uint64_t ReturnAddress,
                       llvm::BasicBlock *ReturnFromCall,
                       Element &Result,
                       ABIIRBasicBlock &ABIBB);

  /// \return true if at least a branch is an indirect tail call
  bool hasIndirectTailCall() const {
    for (auto &P : BranchesType)
      if (P.second == BranchType::IndirectTailCall)
        return true;
    return false;
  }
};

} // namespace Intraprocedural

} // namespace StackAnalysis

#endif // INTRAPROCEDURAL_H
