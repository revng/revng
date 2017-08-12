/// \file stackanalysis.cpp
/// \brief

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// TODO: reduce copy of Element (check arguments and return values)

// Standard includes
#include <array>
#include <map>
#include <queue>
#include <sstream>
#include <vector>

// Boost includes
#include <boost/algorithm/cxx11/any_of.hpp>

// Local includes
#include "datastructures.h"
#include "debug.h"
#include "ir-helpers.h"
#include "jumptargetmanager.h"
#include "revamb.h"
#include "stackanalysis.h"
#include "stackanalysis_impl.h"

// We don't put using namespace llvm since it would create conflicts with the
// local definition of Value
using llvm::AllocaInst;
using llvm::BlockAddress;
using llvm::CallInst;
using llvm::cast;
using llvm::Constant;
using llvm::ConstantInt;
using llvm::DataLayout;
using llvm::dyn_cast;
using llvm::dyn_cast_or_null;
using llvm::Function;
using llvm::isa;
using llvm::LoadInst;
using llvm::Metadata;
using llvm::MDNode;
using llvm::MDString;
using llvm::MDTuple;
using llvm::RegisterPass;
using llvm::SmallBitVector;
using llvm::SmallVector;
using llvm::StoreInst;
using llvm::StringRef;
using llvm::TerminatorInst;
using llvm::UnreachableInst;
using llvm::User;

using namespace StackAnalysis;

/// \brief Given an array of GlobalVariable/AllocaInst, returns a unique index
///        for each one of them
///
/// Get an index for each GlobalVariabe defined in \p M and each AllocaInst in
/// the root function.
///
/// \tparam N the number of elements to search
///
/// \return a std::array of integers representing the index of the corresponding
///         input GlobalVariable/AllocaInst, or -1 if the request object
///         couldn't be found
template<unsigned N>
static inline std::array<int32_t, N>
getCPUIndex(const Module *M, std::array<const User *, N> Search) {
  std::array<int32_t, N> Result;

  // Initialize results to -1
  for (int32_t &Index : Result)
    Index = -1;

  // Go through global variables first
  int32_t I = 0;
  for (const GlobalVariable &GV : M->globals()) {

    int32_t J = 0;
    for (const User *U : Search) {
      if (U == &GV) {
        assert(Result[J] == -1);
        Result[J] = I;
        break;
      }
      J++;
    }

    I++;
  }

  // Look for AllocaInst at the beginning of the root function
  const BasicBlock *Entry = &*M->getFunction("root")->begin();
  auto It = Entry->begin();
  while (It != Entry->end() && isa<AllocaInst>(&*It)) {

    int32_t J = 0;
    for (const User *U : Search) {
      if (U == &*It) {
        assert(Result[J] == -1);
        Result[J] = I;
        break;
      }
      J++;
    }

    I++;
    It++;
  }

  return Result;
}

/// \brief CRTP base class for an element of the lattice
///
/// \note This class is more for reference. It's unused.
///
/// \tparam D the derived class.
template<typename D>
class ElementBase {
public:

  /// \brief The partial ordering relation
  bool lowerThanOrEqual(const ElementBase &RHS) const {
    const D &This = *static_cast<const D *>(this);
    const D &Other = static_cast<const D &>(RHS);
    return This.lowerThanOrEqual(Other);
  }

  /// \brief The opposite of the partial ordering operation
  bool greaterThan(const ElementBase &RHS) const {
    return !this->lowerThanOrEqual(RHS);
  }

  /// \brief The combination operator
  // TODO: assert monotonicity
  ElementBase &combine(const ElementBase &RHS) {
    return static_cast<const D *>(this)->combine(static_cast<const D &>(RHS));
  }

};

/// \brief CRTP base class for implementing a monotone framework
///
/// This class provides the base structure to implement an analysis based on a
/// monotone framework. It also provides an implementation of the MFP solution.
///
/// \tparam Label the type identifying a "label" in the monotone framework,
///         typically an instruction or a basic block.
/// \tparam LatticeElement the type representing an element of the lattice.
/// \tparam Interrupt the type describing why the analysis has been interrupted.
/// \tparam D the derived class.
template<typename Label,
         typename LatticeElement,
         typename Interrupt,
         typename D>
class MonotoneFramework {
public:
  using LabelRange = std::vector<Label>;

  /// \brief The transfer function
  ///
  /// Starting from the initial state at \p BB provides a new lattice element or
  /// a reason why the analysis should be interrupted.
  Interrupt transfer(Label L) {
    return static_cast<D *>(this)->transfer(L);
  }

  /// \brief Return a list of all the extremal labels
  ///
  /// An extremal node is typically the entry or the exit nodes of the function,
  /// depending on whether the analysis being implemented is a forward or
  /// backward analysis.
  LabelRange extremalLabels() const {
    return static_cast<const D *>(this)->extremalLabels();
  }

  /// \brief Return the element of the lattice associated with an extremal label
  LatticeElement extremalValue() {
    return static_cast<D *>(this)->extremalValue();
  }

  /// \brief Initialize/reset the analysis
  ///
  /// Call this method before invoking run or if you want to reset the state of
  /// the analysis.
  void initialize() {
    FirstFinalState = true;
    State.clear();
    WorkList.clear();
    Extremals.clear();

    for (Label ExtremalLabel : extremalLabels()) {
      WorkList.insert(ExtremalLabel);
      State[ExtremalLabel] = extremalValue();
    }

  }

  /// \brief Resolve the data flow analysis problem using the MFP solution
  Interrupt run() {
    while (!WorkList.empty()) {
      Label ToAnalyze = WorkList.head();
      ToVisit.erase(ToAnalyze);

      // Run the transfer function
      Interrupt Result = transfer(ToAnalyze);

      // Check if we should continue or if we should yield control to the
      // caller, i.e., the interprocedural part of the analysis, if present.
      if (Result.requiresInterproceduralHandling())
        return Result;

      // OK, we can handle this result by ourselves: get the result and pop an
      // element from the work list
      const LatticeElement &NewLattice = Result.getResult();
      WorkList.pop();

      // Are we in a final state?
      if (Result.isFinalState()) {

        // If so, accumulate the result in FinalResult
        if (FirstFinalState) {
          FinalResult = NewLattice;
          FirstFinalState = false;
        } else {
          FinalResult.combine(NewLattice);
        }

        DBG("sa", {
            dbg << "FinalResult:\n";
            FinalResult.dump(getModule(ToAnalyze));
            dbg << "\n";
          });

      }

      // If it has successors, check if we have to re-enqueue them
      if (Result.hasSuccessors()) {
        for (Label Successor : Result) {
          auto It = State.find(Successor);
          if (It == State.end()) {
            // We have never seen this Label, register it in the analysis state
            State[Successor] = NewLattice;
            WorkList.insert(Successor);
          } else if (NewLattice.greaterThan(It->second)) {
            // We have already seen this Label but the result of the transfer
            // function is larger than its previous initial state: re-enqueue it
            State[Successor].combine(NewLattice);
            WorkList.insert(Successor);
          }
        }
      }

    }

    // The work list is empty, return to the caller the final result
    assert(ToVisit.empty());
    if (FirstFinalState) {
      return Interrupt::createNoReturnFunction();
    } else {
      return Interrupt::createSummary(FinalResult);
    }
  }

  /// \brief Registers \p L to be visited before the end of the analysis
  ///
  /// If \p L has already been visited at least once before, it's simply
  /// enqueued in the WorkList, otherwise is registered to be visited at least
  /// once before the end of the analysis.
  ///
  /// This function is required when you want to visit a basic block only if
  /// it's part of the current function, or fail otherwise.
  void registerToVisit(Label L) {
    if (State.count(L) == 0)
      ToVisit.insert(L);
    else
      WorkList.insert(L);
  }

public:
  bool FirstFinalState;
  LatticeElement FinalResult;
  UniquedQueue<Label> WorkList;
  std::set<Label> Extremals;
  std::map<Label, LatticeElement> State;
  std::set<Label> ToVisit;
};

/// \brief Class that allows querying mutiple maps at the same time through a
///        "fallback" mechanism
///
/// Given an index, it will be searched in all the registered maps, and the
/// first matching result will be returned.
///
/// \tparam M type of the map container
template<typename M>
class LayeredMap {
public:
  using iterator = typename M::const_iterator;
  using key_type = typename M::key_type;
  using mapped_type = typename M::mapped_type;

public:

  LayeredMap(mapped_type &Default) : Default(Default) { }

  /// \brief Register a new map
  void add(const M &Map) { Maps.push_back(&Map); }

  const mapped_type &operator[](key_type &Key) const {
    auto It = find(Key);
    if (It != end())
      return It->second;

    return Default;
  }

  iterator find(const key_type &Key) const {
    for (const M *Map : Maps) {
      iterator It = Map->find(Key);
      if (It != Map->end())
        return It;
    }

    return end();
  }

  iterator end() const {
    return Maps[0]->end();
  }

  bool contains(const key_type &Key) const {
    return find(Key) != end();
  }

private:
  SmallVector<const M *, 2> Maps;
  mapped_type &Default;
};

/// \brief Class to keep track of the Value associated to each instruction in a
///        basic block
class BasicBlockState {
public:
  using ContentMap = std::map<Instruction *, Value>;

public:
  BasicBlockState(ContentMap &VariableContent,
                  const DataLayout &DL,
                  ASID Last) :
    Default(),
    VariableContent(VariableContent),
    Content(Default),
    DL(DL),
    Last(Last) {

    Content.add(InstructionContent);
    Content.add(VariableContent);
  }

  /// \brief Gets the Value associated to \p V
  ///
  /// This function handles a couple of type of llvm::Values:
  ///
  /// * AllocaInst/GlobalVariables: represent a part of the CPU state, the
  ///   result will be an ASSlot relative to the CPU address space.
  /// * Constant: represent an absolute address, the result will be an ASSlot
  ///   relative to the GLB address space with an offset equal to the actual
  ///   value of the constant.
  /// * Instruction: represents the result of a (previously analyzed)
  ///   Instruction. It can be any Value.
  Value get(llvm::Value *V) const {
    if (auto *CSV = dyn_cast<AllocaInst>(V)) {

      int32_t Index;
      Index = getCPUIndex<1>(getModule(CSV),
                             { static_cast<const User *>(CSV) })[0];
      assert(Index != -1);
      return Value(ASSlot(ASID::cpuID(), Index));

    } else if (auto *CSV = dyn_cast<GlobalVariable>(V)) {

      int32_t Index;
      Index = getCPUIndex<1>(CSV->getParent(),
                             { static_cast<const User *>(CSV) })[0];
      assert(Index != -1);
      return Value(ASSlot(ASID::cpuID(), Index));

    } else if (auto *C = dyn_cast<Constant>(V)) {
      return Value(ASSlot(ASID::globalID(), getZExtValue(C, DL)));
    }

    Instruction *I = cast<Instruction>(V);
    assert(Content.contains(I));
    return Content[I];
  }

  /// \brief Register the value of instruction \p I
  void set(Instruction *I, Value V) {
    assert(I != nullptr);
    DBG("sa-verbose", {
        dbg << "Set " << getName(I) << " to ";
        V.dump(getModule(I));
        dbg << "\n";
      });

    assert(InstructionContent.count(I) == 0);
    InstructionContent[I] = V;
  }

  /// \brief Handle automatically an otherwise un-handleable instruction
  ///
  /// This is a fallback handling of instruction not otherwise manually
  /// handled. The resulting Value will be a combination of the Values of all
  /// its operands.
  void handleGenericInstruction(Instruction *I) {
    assert(I->getNumOperands() > 0);
    Value Result = get(I->getOperand(0));

    DBG("sa-verbose", {
        dbg << "Merging operands: ";
        Result.dump(getModule(I));
      });
    for (llvm::Value *Operand : skip(1, I->operands())) {
      Value OperandValue = get(Operand);
      DBG("sa-verbose", {
          dbg << ", ";
          OperandValue.dump(getModule(I));
        });
      Result.combine(OperandValue);
    }
    DBG("sa-verbose", dbg << "\n");

    set(I, Result);
  }

  std::set<BasicBlock *> save() {
    std::set<BasicBlock *> Result;
    BasicBlock *BB = nullptr;

    for (auto &P : InstructionContent) {
      Instruction *I = P.first;
      Value &NewValue = P.second;

      assert(BB == nullptr || BB == I->getParent());
      BB = I->getParent();

      if (I->isUsedOutsideOfBlock(BB)) {
        bool Changed = false;

        // Has this instruction ever been registered?
        auto It = VariableContent.find(I);
        if (It == VariableContent.end()) {
          VariableContent[I] = NewValue;
          Changed = true;
        } else {
          // If not, are we saying something new?
          Value &OldValue = It->second;
          if (NewValue.greaterThan(OldValue, Last)) {
            OldValue = NewValue;
            Changed = true;
          }
        }

        if (Changed) {
          for (User *U : I->users()) {
            if (auto *UserI = dyn_cast<Instruction>(U)) {
              BasicBlock *UserBB = UserI->getParent();
              if (UserBB != BB)
                Result.insert(UserBB);
            }
          }
        }

      }
    }

    return Result;
  }

private:
  Value Default;
  ContentMap InstructionContent;
  ContentMap &VariableContent;
  LayeredMap<ContentMap> Content;
  const DataLayout &DL;
  ASID Last;
};

/// \brief Compare two bit vectors
static inline bool
lowerThanOrEqual(const SmallBitVector &LHS, const SmallBitVector &RHS) {
  assert(LHS.size() == RHS.size());

  auto T = LHS;
  T &= RHS;
  return T == LHS;
}

static inline bool
greaterThan(const SmallBitVector &LHS, const SmallBitVector &RHS) {
  return !(lowerThanOrEqual(LHS, RHS));
}

/// \brief Cache for the result of the analysis of a function
///
/// This cache keeps track of three pieces of information:
///
/// * the result of the analysis of a function: for each a function, a set of
///   <Context, Result> pairs are stored.
/// * the set of "fake" functions.
/// * the association between each function and its return register.
class Cache {
private:
  struct FunctionSummary {
    SmallVector<Element, 1> MatchingContexts;
    LoadStoreLog Summary;
    bool Temporary;
  };

  struct FunctionInfo {
    ASIndexer Indexer;
    std::vector<FunctionSummary> Summaries;
  };

public:
  using ReturnAddressesMap = std::map<BasicBlock *, GlobalVariable *>;

public:
  Cache() : Default(nullptr) { }

  ASIndexer &getIndexer(BasicBlock *Entry) {
    return Results[Entry].Indexer;
  }

  const ASIndexer &getIndexer(BasicBlock *Entry) const {
    auto It = Results.find(Entry);
    assert(It != Results.end());
    return It->second.Indexer;
  }

  bool isFakeFunction(BasicBlock *Function) const {
    return FakeFunctions.count(Function) != 0;
  }

  void markAsFake(BasicBlock *Function) { FakeFunctions.insert(Function); }

  bool isNoReturnFunction(BasicBlock *Function) const {
    return NoReturnFunctions.count(Function) != 0;
  }

  void markAsNoReturn(BasicBlock *Function) {
    NoReturnFunctions.insert(Function);
  }

  /// \brief Query the cache for the result of the analysis for a specific
  ///        function with a specific context
  ///
  /// \return the matching result, if available.
  Optional<LoadStoreLog>
  get(BasicBlock *Function, Element Query) const {
    auto It = Results.find(Function);
    if (It != Results.end())
      for (const FunctionSummary &Summary : It->second.Summaries)
        if (matches(&getIndexer(Function), Summary, Query))
          return Summary.Summary;

    return Optional<LoadStoreLog>();
  }

  /// \brief Insert in the cache an entry for the function \p Function
  ///
  /// \return true if this function-context was already present with a different
  ///         result.
  // TODO: Result for reference
  bool update(BasicBlock *Function,
              SmallVector<Element, 1> Contexts,
              LoadStoreLog Result,
              bool Temporary=false) {
    ASIndexer *Indexer = &getIndexer(Function);

    DBG("sa", {
        dbg << "Cache.update(" << getName(Function) << "), with contexts\n";
        for (Element &Context : Contexts) {
          Context.dump(getModule(Function));
          dbg << "\n";
        }
        dbg << "with value\n";
        Result.dump(getModule(Function));
        dbg << "\n";
      });

    auto It = Results.find(Function);
    if (It == Results.end()) {
      for (Element &Context : Contexts)
        Context.prune(&Result, Indexer);
      Results[Function].Summaries.push_back({ Contexts, Result, Temporary });
      return false;
    } else {
      bool Changed = false;
      auto &Summaries = It->second.Summaries;

      // For each input context
      for (auto ContextIt = Contexts.begin();
           ContextIt != Contexts.end();
           /**/) {
        Element &Context = *ContextIt;
        bool Remove = false;

        // We have three situations:
        //
        // 1. the context doesn't match any context in an existing summary: we
        //    will insert it in a new summary
        // 2. the context matches a context in an existing summary and the
        //    associated summary is smaller than the current one: we leave the
        //    context where it is and do not add it to the new summary
        // 3. the context matches a conext in an existing summary and the
        //    associated summary is larger than the current one: we drop the
        //    context from the existing summary and we will insert it in a new
        //    summary

        // For each summary
        for (auto SummaryIt = Summaries.begin();
             SummaryIt != Summaries.end();
             /**/) {
          FunctionSummary &Summary = *SummaryIt;

          // Prepare the comparison among the current context and the contexts
          // of this summary
          Element Pruned = Context;
          Pruned.prune(&Summary.Summary, Indexer);
          auto Predicate = [&Pruned] (Element &Candidate) {
            return Pruned.equal(Candidate);
          };

          // Is the summary temporary or do we have an increase in precision?
          if (Summary.Temporary
              || (Result.Store.lowerThanOrEqual(Summary.Summary.Store)
                  && !Summary.Summary.Store.lowerThanOrEqual(Result.Store))) {
            unsigned OldSize = Summary.MatchingContexts.size();
            // Remove all the matching contexts for this summary (the new result
            // is more precise)
            erase_if(Summary.MatchingContexts, Predicate);
            unsigned NewSize = Summary.MatchingContexts.size();

            // If we dropped at least one element, inform the the caller that we
            // changed the summary associated with a previously known context
            if (OldSize != NewSize)
              Changed = true;

            // If there are not matching contexts left, drop this summary
            // altogether
            if (NewSize == 0) {
              SummaryIt = Summaries.erase(SummaryIt);
              continue;
            }

          } else {
            // If the current input contexts matches at least one of the
            // contexts associated with the summary, drop it
            if (std::any_of(Summary.MatchingContexts.begin(),
                            Summary.MatchingContexts.end(),
                            Predicate)) {
              Remove = true;
              break;
            }
          }

          SummaryIt++;
        }

        if (Remove)
          ContextIt = Contexts.erase(ContextIt);
        else
          ContextIt++;

      }

      // Do we have any contexts left?
      if (Contexts.size() == 0) {
        assert(!Changed);
        return false;
      }

      // At this point we should only have a set of contexts which is worth to
      // insert into the cache

      // Prune all of the contexts and create a new summary
      for (Element &Context : Contexts)
        Context.prune(&Result, Indexer);
      It->second.Summaries.push_back({ Contexts, Result, Temporary });

      return Changed;
    }
  }

  /// \brief Get the link register for the function identified by \p Function
  GlobalVariable *getLinkRegister(BasicBlock *Function) const {
    if (Default == nullptr)
      return nullptr;

    if (LinkRegisters.size() == 0) {
      return Default;
    } else {
      auto It = LinkRegisters.find(Function);
      if (It == LinkRegisters.end())
        return Default;
      else
        return It->second;
    }
  }

  /// \brief Initialize the link registers map, providing a default too
  void setLinkRegisters(GlobalVariable *Default,
                        ReturnAddressesMap LinkRegisters) {
    this->Default = Default;
    this->LinkRegisters = LinkRegisters;
  }

private:
  // TODO: inline?
  bool matches(const ASIndexer *Indexer,
               const FunctionSummary &Summary,
               const Element &Query) const {
    Element Pruned = Query;
    Pruned.prune(&Summary.Summary, Indexer);

    for (const Element &Candidate : Summary.MatchingContexts)
      if (Pruned.equal(Candidate))
        return true;

    return false;
  }

private:
  /// Map from function entry point to a vector of context and associated
  /// analysis results
  std::map<BasicBlock *, FunctionInfo> Results;
  ReturnAddressesMap LinkRegisters;
  std::set<BasicBlock *> FakeFunctions;
  std::set<BasicBlock *> NoReturnFunctions;
  GlobalVariable *Default;
};

/// \brief Result of the transfer function
///
/// This class represents the result of the transfer function, it might simply
/// represent the result of the transfer functions starting from the initial
/// state or more sophisticated situations, e.g., function calls that have to be
/// handled by the intraprocedural part of the analysis.
class AnalysisInterrupt {
public:
  enum BranchType {
    InstructionLocalCFG, ///< Branch due to instruction-level CFG (e.g.,
                         ///  conditional move)
    FunctionLocalCFG, ///< Branch due to function-local CFG (a regular branch)
    FakeFunctionCall, ///< A call to a fake function
    FakeFunctionReturn, ///< A return from a fake function
    HandledCall, ///< A function call for which the cache was able to produce a
                 ///  summary
    IndirectCall, ///< A function call for which the target is unknown
    UnhandledCall, ///< A function call for which the cache was not able to
                   ///  produce a summary (interprocedural part of the analysis
                   ///  is required)
    Return, ///< A proper function return
    IndirectTailCall, ///< A branch representing an indirect tail call
    FakeFunction, ///< This function is fake, inform the interprocedural part of
                  ///  the analysis
    LongJmp, ///< A branch representing a longjmp or similar constructs
    Killer, ///< A killer basic block (killer syscall or endless loop)
    Unreachable, ///< The basic block ends with an unreachable instruction
    FunctionSummary, ///< The analysis of the function is finished and a summary
                     ///  is available
    NoReturnFunction ///< This is a function for which we couldn't find any
                     ///  return statement
  };

  using vector = SmallVector<BasicBlock *, 2>;

private:
  AnalysisInterrupt(ElementProxy Result, BranchType Type) :
    Type(Type), Result(Result) { }

  AnalysisInterrupt(ElementProxy Result,
                    BranchType Type,
                    SmallVector<BasicBlock *, 2> Successors) :
    Type(Type), Result(Result), RelatedBasicBlocks(Successors) { }

public:
  static AnalysisInterrupt
  createWithSuccessor(ElementProxy Result,
                      BranchType Type,
                      BasicBlock * Successor) {
    assert(Type == FakeFunctionCall
           || Type == HandledCall
           || Type == IndirectCall
           || Type == FakeFunctionReturn);

    return AnalysisInterrupt(Result, Type, { Successor });
  }

  static AnalysisInterrupt
  createWithSuccessors(ElementProxy Result,
                       BranchType Type,
                       SmallVector<BasicBlock *, 2> Successors) {
    assert(Type == InstructionLocalCFG || Type == FunctionLocalCFG);
    assert(Successors.size() > 0);

    return AnalysisInterrupt(Result, Type, Successors);
  }

  static AnalysisInterrupt createNoReturnFunction() {
    return create(ElementProxy(), NoReturnFunction);
  }

  static AnalysisInterrupt create(ElementProxy Result, BranchType Type) {
    assert(Type == Return
           || Type == IndirectTailCall
           || Type == FakeFunction
           || Type == LongJmp
           || Type == Killer
           || Type == Unreachable
           || Type == NoReturnFunction);

    return AnalysisInterrupt(Result, Type);
  }

  static AnalysisInterrupt
  createUnhandledCall(ElementProxy Context, BasicBlock *Callee) {
    return AnalysisInterrupt(Context, UnhandledCall, { Callee });
  }

  static AnalysisInterrupt createSummary(ElementProxy Result) {
    return AnalysisInterrupt(Result, FunctionSummary, { });
  }

public:
  bool hasSuccessors() const {
    switch (Type) {
    case InstructionLocalCFG:
    case FunctionLocalCFG:
    case FakeFunctionCall:
    case FakeFunctionReturn:
    case HandledCall:
    case IndirectCall:
      return true;
    case UnhandledCall:
    case Return:
    case IndirectTailCall:
    case FakeFunction:
    case LongJmp:
    case Killer:
    case Unreachable:
      return false;
    case FunctionSummary:
    case NoReturnFunction:
    default:
      abort();
    }
  }

  bool isFinalState() const {
    switch (Type) {
    case Return:
    case IndirectTailCall:
      // TODO: an IndirectTailCall, *can be* an IndirectTail call, but it might
      //       also just be a longjmp. So it's an assumption that this is a
      //       final state.
      return true;
    case InstructionLocalCFG:
    case FunctionLocalCFG:
    case FakeFunctionCall:
    case FakeFunctionReturn:
    case HandledCall:
    case IndirectCall:
    case UnhandledCall:
    case FakeFunction:
    case LongJmp:
    case Killer:
    case Unreachable:
      return false;
    case FunctionSummary:
    case NoReturnFunction:
    default:
      abort();
    }
  }

  BranchType type() const { return Type; }

  bool requiresInterproceduralHandling() const {
    switch (Type) {
    case FakeFunction:
    case UnhandledCall:
    case FunctionSummary:
    case NoReturnFunction:
      return true;
    case InstructionLocalCFG:
    case FunctionLocalCFG:
    case FakeFunctionCall:
    case FakeFunctionReturn:
    case HandledCall:
    case IndirectCall:
    case Return:
    case IndirectTailCall:
    case LongJmp:
    case Killer:
    case Unreachable:
      return false;
    default:
      abort();
    }
  }

  const ElementProxy &getResult() const {
    assert(Type != FunctionSummary && Type != UnhandledCall);
    return Result;
  }

  BasicBlock *getCallee() const {
    assert(Type == UnhandledCall);
    assert(RelatedBasicBlocks.size() == 1);
    return RelatedBasicBlocks[0];
  }

  const Element &getCallContext() const {
    assert(Type == UnhandledCall);
    return Result.actual();
  }

  const ElementProxy &getFunctionSummary() const {
    assert(Type == FunctionSummary);
    return Result;
  }

  vector::const_iterator begin() {
    assert(hasSuccessors());
    return RelatedBasicBlocks.begin();
  }

  vector::const_iterator end()  {
    assert(hasSuccessors());
    return RelatedBasicBlocks.end();
  }

  static StringRef getTypeName(BranchType Type) {
    switch (Type) {
    case InstructionLocalCFG:
      return "InstructionLocalCFG";
    case FunctionLocalCFG:
      return "FunctionLocalCFG";
    case FakeFunctionCall:
      return "FakeFunctionCall";
    case FakeFunctionReturn:
      return "FakeFunctionReturn";
    case HandledCall:
      return "HandledCall";
    case IndirectCall:
      return "IndirectCall";
    case UnhandledCall:
      return "UnhandledCall";
    case Return:
      return "Return";
    case IndirectTailCall:
      return "IndirectTailCall";
    case FakeFunction:
      return "FakeFunction";
    case LongJmp:
      return "LongJmp";
    case Killer:
      return "Killer";
    case FunctionSummary:
      return "FunctionSummary";
    case NoReturnFunction:
      return "NoReturnFunction";
    default:
      abort();
    }
  }

  void dump(const Module *M) const {
    dbg << "Interrupt reason: " << getTypeName(Type).str() << "\n";
    switch (Type) {
    case InstructionLocalCFG:
    case FunctionLocalCFG:
    case FakeFunctionCall:
    case FakeFunctionReturn:
    case HandledCall:
    case IndirectCall:
      dbg << "Successors:";
      for (BasicBlock *BB : RelatedBasicBlocks)
        dbg << " " << getName(BB);
      dbg << "\n";
      dbg << "Result:\n";
      Result.dump(M);
      break;
    case UnhandledCall:
      dbg << "Unhandled call to " << getName(RelatedBasicBlocks[0])
          << " with context:\n";
      Result.dump(M);
      break;
    case Return:
    case FunctionSummary:
    case NoReturnFunction:
    case IndirectTailCall:
    case FakeFunction:
    case LongJmp:
    case Killer:
      dbg << "Result:\n";
      Result.dump(M);
      break;
    default:
      abort();
    }
  }

private:
  BranchType Type;
  ElementProxy Result;
  SmallVector<BasicBlock *, 2> RelatedBasicBlocks;
};

/// \brief Intraprocedural part of the stack analysis
class IntraproceduralAnalysis :
  public MonotoneFramework<BasicBlock *,
                           ElementProxy,
                           AnalysisInterrupt,
                           IntraproceduralAnalysis> {
public:
  IntraproceduralAnalysis(BasicBlock *Entry,
                          Cache &TheCache,
                          GeneratedCodeBasicInfo *GCBI,
                          Element InitialState) :
    Entry(Entry),
    TheCache(&TheCache),
    GCBI(GCBI),
    InitialState(InitialState),
    Indexer(&TheCache.getIndexer(Entry)) {

    const Module *M = getModule(Entry);

    DBG("sa", {
        dbg << "Creating IntraproceduralAnalysis for " << getName(Entry);
        dbg << " with the follwing context\n";
        this->InitialState.dump(M);
        dbg << "\n";
      });

    TerminatorInst *T = Entry->getTerminator();
    assert(T != nullptr);

    // Obtain the link register used to call this function
    LinkRegister = TheCache.getLinkRegister(Entry);

    // Get the register indices for for the stack pointer, the program counter
    // andn the link register
    int32_t LinkRegisterIndex;
    auto Indices = getCPUIndex<3>(getModule(Entry),
                                  {
                                    static_cast<const User *>(GCBI->pcReg()),
                                    static_cast<const User *>(LinkRegister),
                                    static_cast<const User *>(GCBI->spReg())
                                  });
    PCIndex = Indices[0];
    LinkRegisterIndex = Indices[1];
    SPIndex = Indices[2];

    assert(PCIndex != -1
           && ((LinkRegisterIndex == -1) ^ (LinkRegister != nullptr)));

    // Set the stack pointer to SP0+0
    Value StackPointer(ASSlot(ASID::cpuID(), SPIndex));
    ASID StackID = ASID::lastStackID();
    this->InitialState.store(StackPointer, ASSlot(StackID, 0));

    Top = this->InitialState.flatten();

    if (LinkRegister == nullptr) {
      auto StackID = ASID::lastStackID();
      ReturnValue = this->InitialState.load(ASSlot(StackID, 0));
    } else {
      ReturnValue = this->InitialState.load(ASSlot(ASID::cpuID(),
                                                   LinkRegisterIndex));
    }

    DBG("sa", {
        dbg << "The return address is in ";
        if (LinkRegister != nullptr)
          dbg << LinkRegister->getName().str();
        else
          dbg << "the top of the stack";
        dbg << " and has value ";
        ReturnValue.dump(M);
        dbg << "\n";
      });

    DBG("sa-verbose", {
        dbg << "ReturnValue: ";
        ReturnValue.dump(M);
        dbg << "\n";
      });

    initialize();

  }

  AnalysisInterrupt transfer(BasicBlock *BB);

  /// \brief There's only on extremal label: the entry node
  LabelRange extremalLabels() const { return { Entry }; }

  /// \brief The extremal value, i.e., the context of the analysis
  ElementProxy extremalValue() {
    return ElementProxy(InitialState, Indexer);
  }

  using BranchType = AnalysisInterrupt::BranchType;
  using BranchesTypeVector = std::vector<std::pair<BasicBlock *, BranchType>>;

  BranchesTypeVector getBranchesType() const {
    BranchesTypeVector Result;
    std::copy(BranchesType.begin(),
              BranchesType.end(),
              std::back_inserter(Result));
    return Result;
  }

private:
  /// \brief Part of the transfer function handling terminator instructions
  AnalysisInterrupt
  handleTerminator(TerminatorInst *T, ElementProxy &Result);

  AnalysisInterrupt handleCall(BasicBlock *Callee,
                               uint64_t ReturnAddress,
                               BasicBlock *ReturnFromCall,
                               ElementProxy &Result);

private:
  BasicBlock *Entry;
  Cache *TheCache;
  GlobalVariable *LinkRegister;
  GeneratedCodeBasicInfo *GCBI;
  std::set<uint64_t> FakeReturnAddresses;
  Element InitialState;
  Value ReturnValue;
  ASSet Top;
  int32_t SPIndex;
  int32_t PCIndex;
  ASIndexer *Indexer;
  std::map<BasicBlock *, AnalysisInterrupt::BranchType> BranchesType;
  std::map<Instruction *, Value> VariableContent;
};

/// \brief Is the basic block part of the function-local CFG?
///
/// Returns true if the basic block starts with a new instruction marker (a call
/// to newpc) or it's a dispatcher-related basic block
static bool isInstructionLocal(BasicBlock *BB) {
  assert(!BB->empty());
  if (isCallTo(&*BB->begin(), "newpc"))
    return false;

  return BB->getTerminator()->getMetadata("revamb.block.type") == nullptr;
}

AnalysisInterrupt
IntraproceduralAnalysis::handleCall(BasicBlock *Callee,
                                    uint64_t ReturnAddress,
                                    BasicBlock *ReturnFromCall,
                                    ElementProxy &Result) {

  using SAI = AnalysisInterrupt;

  int32_t PCRegSize = GCBI->pcRegSize();
  Value StackPointer(ASSlot(ASID::cpuID(), SPIndex));
  Value PC(ASSlot(ASID::cpuID(), PCIndex));
  const Module *M = getModule(Entry);

  bool IsIndirect = Callee == nullptr;
  bool IsIndirectTailCall = IsIndirect && ReturnFromCall == nullptr;

  // Save the value of the stack pointer for later
  Value OldStackPointer = Result.load(StackPointer);

  //
  // Prepare the context for querying the cache
  //
  Element CallContext = Result.actual();

  // We have to transform the context into the actual state of this
  // function. Shift all the address space indices by one position
  CallContext.shiftAddressSpaces(+1);

  // We now have to initialize some fields of the new state. Specifically,
  // we have to set the stack pointer to SP0+0 and copy and flip the
  // caller's address space into the new SP0 address space.

  ASID StackID = ASID::lastStackID();
  ASID CallerStackID = CallContext.getCallerStack(StackID);

  // If we know the current stack frame size, copy the arguments
  // TODO: report failure somehow
  if (const ASSlot *OldStackPointerSlot = OldStackPointer.getASSlot()) {
    assert(OldStackPointerSlot->addressSpace() == StackID);

    int32_t CallerStackSize = -OldStackPointerSlot->offset();
    assert(CallerStackSize >= 0);

    CallContext.copyStackArguments(CallerStackID, StackID, CallerStackSize);
  }

  // Set the stack pointer to SP0+0
  CallContext.store(StackPointer, ASSlot(StackID, 0));

  // The context is now ready

  LoadStoreLog CallSummary;

  // Is it an indirect function call?
  if (IsIndirect) {
    // We have an indirect call, we assume top except for the stack pointer
    // and the program counter being restored as expected
    ElementProxy IndirectCallResult(CallContext, Result.indexer());

    // Set to top
    IndirectCallResult.top();

    CallSummary = std::move(IndirectCallResult.log());
  } else {
    // We have a direct call
    assert(Callee != nullptr);

    // It's a direct function call, lookup the <Callee, Context> pair in the
    // cache
    Optional<LoadStoreLog> CacheEntry = TheCache->get(Callee, CallContext);

    if (CacheEntry) {
      // We have a match in the cache
      CallSummary = std::move(CacheEntry.getValue());
    } else {
      // We don't have a match in the cache. Ask interprocedural analysis to
      // analyze this function call with the current context
      return SAI::createUnhandledCall(ElementProxy(CallContext, nullptr),
                                      Callee);
    }

  }

  DBG("sa", {
      dbg << "The summary result for a call to " << getName(Callee)
          << " with the following context\n";
      CallContext.dump(M);
      dbg << "is\n";
      CallSummary.dump(M);
      dbg << "\n";
    });

  ASIndexer &OldIndexer = TheCache->getIndexer(Callee);
  ASIndexer NewIndexer = CallSummary.moveToCaller(CallContext, OldIndexer);

  // Use the result and proceed from the return address
  Result.applyStores(CallSummary.Store);
  Result.applyLoads(NewIndexer, CallSummary.Load);

  // Restore the stack pointer
  GlobalVariable *CalleeLinkRegister = TheCache->getLinkRegister(Callee);
  if (CalleeLinkRegister == nullptr) {
    // Increase the stack pointer of the size of the PC reg
    OldStackPointer.add(PCRegSize);
  }
  Result.store(StackPointer, OldStackPointer);

  // Restore the PC
  // TODO: handle return address from indirect tail calls
  ASSlot ReturnAddressSlot(ASID::globalID(), ReturnAddress);
  Result.store(PC, Value(ReturnAddressSlot));

  if (IsIndirectTailCall) {
    return SAI::create(Result, SAI::IndirectTailCall);
  } else {
    assert(ReturnFromCall != nullptr);
    auto Reason = IsIndirect ? SAI::IndirectCall : SAI::HandledCall;
    return SAI::createWithSuccessor(Result, Reason, ReturnFromCall);
  }

}

AnalysisInterrupt
IntraproceduralAnalysis::handleTerminator(TerminatorInst *T,
                                          ElementProxy &Result) {
  assert(!isa<UnreachableInst>(T));

  using SAI = AnalysisInterrupt;

  int32_t PCRegSize = GCBI->pcRegSize();
  Value StackPointer(ASSlot(ASID::cpuID(), SPIndex));
  Value PC(ASSlot(ASID::cpuID(), PCIndex));

  // 0. Check if it's a direct killer basic block
  // TODO: we should move the metadata enums and functions to get their names to
  //       GCBI
  auto NoReturnMD = T->getMetadata("noreturn");
  if (auto *NoreturnTuple = dyn_cast_or_null<MDTuple>(NoReturnMD)) {
    QuickMetadata QMD(getContext(T));
    auto NoReturnType = QMD.extract<StringRef>(NoreturnTuple, 0);
    if (NoReturnType != "LeadsToKiller")
      return SAI::create(Result, SAI::Killer);
  }

  // 1. Check if we're dealing with instruction-local control flow (e.g., the if
  //    generated due to a conditional branch)
  // 2. Check if it's an indirect branch, which means that "anypc" is among
  //    successors
  bool IsInstructionLocal = false;
  bool IsIndirect = false;
  bool IsUnresolvedIndirect = false;

  for (BasicBlock *Successor : T->successors()) {
    // If at least one successor is not a jump target, the branch is instruction
    // local
    IsInstructionLocal = IsInstructionLocal || isInstructionLocal(Successor);

    BlockType SuccessorType = GCBI->getType(Successor->getTerminator());

    IsIndirect = IsIndirect
      || SuccessorType == AnyPCBlock
      || SuccessorType == UnexpectedPCBlock
      || SuccessorType == DispatcherBlock;

    IsUnresolvedIndirect = IsUnresolvedIndirect
      || SuccessorType == AnyPCBlock
      || SuccessorType == DispatcherBlock;
  }

  if (IsInstructionLocal) {
    SmallVector<BasicBlock *, 2> Successors;
    std::copy(T->successors().begin(),
              T->successors().end(),
              std::back_inserter(Successors));
    return SAI::createWithSuccessors(Result,
                                     SAI::InstructionLocalCFG,
                                     Successors);
  }

  // 3. Check if this a function call (although the callee might not be a proper
  //    function)
  bool IsFunctionCall = false;
  BasicBlock *Callee = nullptr;
  BasicBlock *ReturnFromCall = nullptr;
  uint64_t ReturnAddress = 0;

  auto It = T->getIterator();
  if (It != T->getParent()->begin()) {
    It--;
    // TODO: we'd probably need something like prevNonMarker
    if (CallInst *Call = getCallTo(&*It, "function_call")) {
      IsFunctionCall = true;
      auto *Arg0 = Call->getArgOperand(0);
      auto *Arg1 = Call->getArgOperand(1);
      auto *Arg2 = Call->getArgOperand(2);

      if (auto *CalleeBlockAddress = dyn_cast<BlockAddress>(Arg0))
        Callee = CalleeBlockAddress->getBasicBlock();

      auto *ReturnBlockAddress = cast<BlockAddress>(Arg1);
      ReturnFromCall = ReturnBlockAddress->getBasicBlock();

      ReturnAddress = getLimitedValue(Arg2);
    }
  }

  // 4. Check if the stack pointer is in its original position
  bool IsInitialStack = false;
  bool IsPostReturnStack = false;

  // Get the current value of the stack pointer
  // TODO: we should evaluate the approximation introduced here appropriately
  Value StackPointerValue = Result.load(StackPointer);

  const ASSlot *StackPointerSlot = StackPointerValue.getASSlot();

  if (StackPointerSlot != nullptr
      && StackPointerSlot->addressSpace() == ASID::lastStackID()) {
    // Check if we're exactly at same position as at the entry point
    IsInitialStack = StackPointerSlot->offset() == 0;

    if (LinkRegister == nullptr) {
      // The return address is pushed on the stack: the post return stack should
      // be the original one plus the size of the PC register
      IsPostReturnStack = StackPointerSlot->offset() == PCRegSize;
    } else {
      // We have an actual link register: the post return stack should be the
      // same as the initial one
      IsPostReturnStack = IsInitialStack;
    }

  }

  // 5. Are we jumping to the return address? Are we jumping to the return
  //    address from a fake function?
  bool IsReturn = false;
  bool IsReturnFromFake = false;
  uint64_t FakeFunctionReturnAddress = 0;

  if (IsIndirect) {
    // Get the current value being stored in the program counter
    Value ProgramCounter(ASSlot(ASID::cpuID(), PCIndex));
    Value ProgramCounterValue = Result.load(ProgramCounter);

    // Do we have a named value as content?
    if (const TaggedValue *PCContent = ProgramCounterValue.directContent()) {
      const Tag *T = PCContent->tag();

      // It's a return if the PC has a value with a name matching the name of
      // the initial value of the link register or if the initial value of the
      // link register had no indirect content and the direct content is an
      // ASSlot
      if (const TaggedValue *Return = ReturnValue.directContent()) {
        const Tag *ReturnTag = Return->tag();
        IsReturn = ReturnTag != nullptr
          && T != nullptr
          && *ReturnTag == *T;

        if (const ASSlot *ReturnASSlot = ReturnValue.getASSlot()) {
          if (const ASSlot *PCASSlot = PCContent->getASSlot()) {
            IsReturn |= *ReturnASSlot == *PCASSlot;
          }
        }

      }

      if (!IsReturn) {
        if (const ASSlot *ASSlotValue = PCContent->getASSlot()) {
          if (ASSlotValue->addressSpace() == ASID::globalID()) {
            uint32_t Offset = ASSlotValue->offset();
            FakeFunctionReturnAddress = Offset;
            IsReturnFromFake = FakeReturnAddresses.count(Offset) != 0;
          }
        }
      }

    }
  }

  // 6. Using the collected information classify the branch type

  // Are we returning to the return address?
  if (IsReturn) {

    // Is the stack in the position it should be?
    if (IsPostReturnStack) {
      // This looks like an actual return
      return SAI::create(Result, SAI::Return);
    } else {
      // We have a return instruction, but the stack is not in the position we'd
      // expect, mark this function as a fake function.
      return SAI::create(Result, SAI::FakeFunction);
    }

  }

  if (IsFunctionCall) {

    // Is it a call to a fake function?
    if (!IsIndirect) {
      if (TheCache->isFakeFunction(Callee)) {
        // Assume normal control flow (i.e., inline)
        FakeReturnAddresses.insert(ReturnAddress);
        return SAI::createWithSuccessor(Result, SAI::FakeFunctionCall, Callee);
      } else if (TheCache->isNoReturnFunction(Callee)) {
        return SAI::create(Result, SAI::Killer);
      }
    }

    return handleCall(Callee, ReturnAddress, ReturnFromCall, Result);
  }

  // Is it an indirect jump?
  if (IsIndirect) {

    // Is it targeting an address that we registered as a return from fake
    // function call?
    if (IsReturnFromFake) {
      // Continue from there
      BasicBlock *ReturnBB = GCBI->getBlockAt(FakeFunctionReturnAddress);
      return SAI::createWithSuccessor(Result,
                                      SAI::FakeFunctionReturn,
                                      ReturnBB);
    }

    // Check if it's a real indirect jump, i.e. we're not 100% of the targets
    if (IsUnresolvedIndirect) {
      if (IsInitialStack) {
        // If the stack is not in a valid position, we consider it an indirect
        // tail call
        return handleCall(nullptr, 0, nullptr, Result);
      } else {
        // We have an indirect jump with a stack not ready to return, It'a a
        // longjmp
        return SAI::create(Result, SAI::LongJmp);
      }
    }

  }

  // The branch is direct and has nothing else special, consider it
  // function-local
  SmallVector<BasicBlock *, 2> Successors;
  for (BasicBlock *Successor : T->successors())
    if (GCBI->getType(Successor) != UnexpectedPCBlock)
      Successors.push_back(Successor);

  return SAI::createWithSuccessors(Result, SAI::FunctionLocalCFG, Successors);
}

AnalysisInterrupt IntraproceduralAnalysis::transfer(BasicBlock *BB) {
  ElementProxy Result = State[BB];
  const Module *M = getModule(BB);

  DBG("sa", {
      dbg << "Analyzing basic block " << getName(BB);
      dbg << "\n";
      Result.actual().dump(M);
      dbg << "\n";
    });

  // TODO: prune all the info about dead instructions
  BasicBlockState BBState(VariableContent,
                          M->getDataLayout(),
                          Result.actual().last());

  for (Instruction &I : *BB) {

    DBG("sa-verbose", {
        dbg << "NewInstruction: ";
        I.dump();
      });

    switch (I.getOpcode()) {
    case Instruction::Load:
      {
        auto *Load = cast<LoadInst>(&I);
        const Value &AddressValue = BBState.get(Load->getPointerOperand());
        BBState.set(&I, Result.load(AddressValue));
        break;
      }
    case Instruction::Store:
      {
        auto *Store = cast<StoreInst>(&I);
        Value Address = BBState.get(Store->getPointerOperand());
        Value StoredValue = BBState.get(Store->getValueOperand());
        Result.store(Address, StoredValue);
        break;
      }
    case Instruction::And:
      {
        // If we're masking an address with a mask that is at most as strict as
        // the one for instruction alignment, ignore the operation. Note that
        // this works if the address is pointing to code, but not necessarily if
        // it's pointing to data.
        Value FirstOperand = BBState.get(I.getOperand(0));
        if (auto *SecondOperand = dyn_cast<ConstantInt>(I.getOperand(1))) {
          uint64_t Mask = getLimitedValue(SecondOperand);

          uint64_t SignificantPCBits;
          if (GCBI->pcRegSize() == 4) {
            SignificantPCBits = std::numeric_limits<uint32_t>::max();
          } else {
            assert(GCBI->pcRegSize() == 8);
            SignificantPCBits = std::numeric_limits<uint64_t>::max();
          }
          uint64_t AlignmentMask = GCBI->instructionAlignment() - 1;
          SignificantPCBits = SignificantPCBits & ~AlignmentMask;

          if ((SignificantPCBits & Mask) == SignificantPCBits) {
            BBState.set(&I, FirstOperand);
            break;
          }

        }

        BBState.handleGenericInstruction(&I);
        break;
      }
    case Instruction::Add:
    case Instruction::Sub:
      {
        int Sign = I.getOpcode() == Instruction::Add ? +1 : -1;

        // If the second operand is constant we can handle it
        Value FirstOperand = BBState.get(I.getOperand(0));
        if (auto *Addend = dyn_cast<ConstantInt>(I.getOperand(1))) {
          if (FirstOperand.add(Sign * getLimitedValue(Addend))) {
            BBState.set(&I, FirstOperand);
            break;
          }
        }

        BBState.handleGenericInstruction(&I);
        break;
      }
    case Instruction::Call:
      // TODO: handle calls to helpers, we should detect where they could write
      if (!cast<CallInst>(&I)->getFunctionType()->getReturnType()->isVoidTy())
        BBState.set(&I, Top);

      if (!isMarker(&I))
        handleCall(nullptr, 0, nullptr, Result);
      break;
    case Instruction::Br:
    case Instruction::Switch:
      {
        auto *T = cast<TerminatorInst>(&I);

        AnalysisInterrupt BBResult = handleTerminator(T, Result);

        BranchesType[BB] = BBResult.type();

        std::set<BasicBlock *> ToReanalyze = BBState.save();
        for (BasicBlock *BB : ToReanalyze)
          registerToVisit(BB);

        DBG("sa", {
            dbg << "Basic block terminated: " << getName(BB) << "\n";
            BBResult.dump(M);
            dbg << "\n";
          });

        return BBResult;
        break;
      }
    case Instruction::Unreachable:
      {
        using SAI = AnalysisInterrupt;
        return SAI::create(Result, SAI::Unreachable);
      }
    default:
      BBState.handleGenericInstruction(&I);
      break;
    }

    assert(Result.verify());
  }

  assert(false);
}

/// \brief Interprocedural part of the stack analysis
class InterproceduralAnalysis {
private:
  /// \brief Entry of the stack of the functions being currently analyzed
  ///
  /// For each function currently being analyzed, it keeps track of the original
  /// context which was used to enqueue it, its possibly enlarged version (e.g.,
  /// due to recursion), the function entry point and the state of the
  /// intraprocedural analysis so it can be resumed.
  struct WorkItem {
    WorkItem(Element Context,
             BasicBlock *Function,
             Cache &TheCache,
             GeneratedCodeBasicInfo &GCBI) :
      Context(Context),
      MatchingContexts({ Context }),
      Function(Function),
      State(Function, TheCache, &GCBI, Context) { }

    Element Context;
    SmallVector<Element, 1> MatchingContexts;
    BasicBlock *Function;
    IntraproceduralAnalysis State;

    bool verify(LoadStoreLog *Log=nullptr, ASIndexer *Indexer=nullptr) const {
      // Check that the combination of the matching context is the context we
      // used to compute the analysis
      Element Combined = MatchingContexts[0];
      auto Next = skip(1, MatchingContexts);
      for (const Element &MatchingContext : Next)
        Combined.combine(MatchingContext);

      if (!Combined.equal(Context))
        return false;

      if (Log != nullptr) {
        assert(Indexer != nullptr);

        // Check that the combination of the pruned matching context is the
        // pruned context we used to compute the analysis
        auto Prune = [Indexer, Log] (Element L) {
          L.prune(Log, Indexer);
          return L;
        };
        Element PrunedContext = Prune(Context);
        Combined = Prune(MatchingContexts[0]);
        for (const Element &Context : skip(1, MatchingContexts)) {
          Combined.combine(Prune(Context));
        }
        if (!Combined.equal(PrunedContext))
          return false;
      }

      return true;
    }

  };

public:
  InterproceduralAnalysis(Cache &TheCache,
                          GeneratedCodeBasicInfo &GCBI,
                          const Module *M) :
    TheCache(TheCache), GCBI(GCBI) { }

  class Result {
  private:
    enum FunctionType {
      Regular,
      Fake,
      NoReturn
    };

  public:
    using BranchesTypeVector = IntraproceduralAnalysis::BranchesTypeVector;

  private:
    Result(BranchesTypeVector BranchesType,
           std::set<ASSlot> CalleeSaved,
           FunctionType Type) :
      BranchesType(BranchesType),
      CalleeSaved(CalleeSaved),
      Type(Type) { }

  public:
    static Result createFake(BranchesTypeVector BranchesType) {
      return Result(BranchesType, { }, Fake);
    }

    static Result createNoReturn(BranchesTypeVector BranchesType) {
      return Result(BranchesType, { }, NoReturn);
    }

    static Result createRegular(BranchesTypeVector BranchesType,
                                std::set<ASSlot> CalleeSaved) {
      return Result(BranchesType, CalleeSaved, Regular);
    }

    const BranchesTypeVector &branchesType() const {
      return BranchesType;
    }

    const std::set<ASSlot> &calleeSaved() const {
      assert(isRegular());
      return CalleeSaved;
    }

    bool isRegular() const { return Type == Regular; }
    bool isFake() const { return Type == Fake; }
    bool isNoReturn() const { return Type == NoReturn; }

  private:
    BranchesTypeVector BranchesType;
    std::set<ASSlot> CalleeSaved;
    FunctionType Type;
  };

  Result run(BasicBlock *Entry) {
    using Result = InterproceduralAnalysis::Result;

    assert(InProgress.size() == 0);

    const Module *M = getModule(Entry);

    // Prepare the context of the function being analyzed: spread all the
    // initial address spaces (GLB, CPU and RST) with an ASSet of GLB and RST.
    ASSet InitialContent;
    InitialContent.add(ASID::globalID());
    InitialContent.add(ASID::restOfTheStackID());

    ASSet Address;
    Address.add(ASID::globalID());
    Address.add(ASID::cpuID());
    Address.add(ASID::restOfTheStackID());

    auto Indices = getCPUIndex<1>(M,
                                  { static_cast<const User *>(GCBI.spReg()) });
    unsigned SPIndex = Indices[0];
    Value StackPointer(ASSlot(ASID::cpuID(), SPIndex));

    Element InitialContext;
    InitialContext.store(Value(Address), Value(InitialContent));
    ASID StackID = ASID::lastStackID();

    // Set the stack pointer to SP0+0
    InitialContext.store(StackPointer, ASSlot(StackID, 0));

    newFunction(Entry, InitialContext);

    while (true) {
      assert(!InProgress.empty());
      // Get an element from the queue (but don't pop)
      WorkItem &Current = InProgress.back();

      DBG("sa-interp", {
          dbg << "Analyzing " << getName(Current.Function) << "\n";
        });

      // Run/continue the intraprocedural analysis
      AnalysisInterrupt Result = Current.State.run();

      assert(Result.requiresInterproceduralHandling());

      switch(Result.type()) {
      case AnalysisInterrupt::FakeFunction:
        DBG("sa-interp", {
            dbg << getName(Current.Function) << " is fake\n";
          });

        // It's a fake function, mark it as so and resume from the caller
        TheCache.markAsFake(Current.Function);

        if (InProgress.size() == 1)
          return Result::createFake(Current.State.getBranchesType());

        InProgress.pop_back();
        break;
      case AnalysisInterrupt::NoReturnFunction:
        DBG("sa-interp", {
            dbg << getName(Current.Function) << " does not return\n";
          });

        // It's a fake function, mark it as so and resume from the caller
        TheCache.markAsNoReturn(Current.Function);

        if (InProgress.size() == 1)
          return Result::createNoReturn(Current.State.getBranchesType());

        InProgress.pop_back();
        break;
      case AnalysisInterrupt::UnhandledCall:
        {
          const Element &CallContext = Result.getCallContext();
          BasicBlock *Callee = Result.getCallee();
          assert(Callee != nullptr);

          DBG("sa-interp", {
              dbg << getName(Current.Function)
                  << " performs an unhandled call to "
                  << getName(Callee) << "\n";
            });

          // Check if it's a recursive call
          auto RecursionRootIt = InProgress.begin();
          while (RecursionRootIt->Function != Callee
                 && RecursionRootIt != InProgress.end()) {
            RecursionRootIt++;
          }

          // Is it recursive?
          if (RecursionRootIt != InProgress.end()) {
            WorkItem &RecursionRoot = *RecursionRootIt;
            DBG("sa-interp", {
                dbg << getName(Current.Function) << " is recursive\n";
              });

            // Is the context of the recursive function call contained in the
            // context of the root of the recursion?
            if (CallContext.lowerThanOrEqual(RecursionRoot.Context)) {
              // OK, we can proceed. We have to inject in the cache a temporary
              // top entry for CallContext. If necessary, we also have to
              // register this CallContext as compatible with RecursionRoot so
              // that the cache can keep track of this.

              // Register CallContext in RecursionRoot: the result of this
              // analysis has to match also this context in the cache
              assert(RecursionRoot.verify());
              RecursionRoot.MatchingContexts.push_back(CallContext);
              assert(RecursionRoot.verify());

              DBG("sa-interp", {
                  dbg << "The context of the recursive call is large enough\n";
                });

              // Inject in the cache the <Result.Callee, RootContext, top>
              IntraproceduralAnalysis Analysis(Callee,
                                               TheCache,
                                               &GCBI,
                                               CallContext);
              auto Top = Analysis.extremalValue();
              Top.top();
              assert(Top.verify());
              TheCache.update(Callee,
                              RecursionRoot.MatchingContexts,
                              Top.log(),
                              true);

              // At this point the intraprocedural analysis will resume
              // employing bottom for the recursive call. Then, once Analysis is
              // done, the interprocedural part will detect that the result
              // associated with it has changed (hopefully the result won't be
              // bottom) and will run the analysis again until we're stable.

            } else {
              // The context is not contained in the root of recursion.  Give up
              // on everything, go back to the root of recursion and re-compute
              // it with a wider context (the original one merged with the one
              // of the recursive call)

              DBG("sa-interp", {
                  dbg << "The context of the recursive call is ";
                  dbg << "NOT large enough\n";
                });

              // Merge the context of the recursive function call with the one
              // from the root of the recursion
              Element NewContext = CallContext;
              NewContext.combine(RecursionRoot.Context);

              DBG("sa", {
                  dbg << "Call context\n";
                  CallContext.dump(M);
                  dbg << "Recursion root context\n";
                  RecursionRoot.Context.dump(M);
                  dbg << "Result\n";
                  NewContext.dump(M);
                });

              assert(CallContext.lowerThanOrEqual(NewContext));
              assert(RecursionRoot.Context.lowerThanOrEqual(NewContext));

              // Pop everything from the top of tstack up until the root of the
              // recursion excluded
              InProgress.erase(++RecursionRootIt, InProgress.end());
              assert(&RecursionRoot == &InProgress.back());

              // Update the context associated to RecursionRoot with the widened
              // one and register the context of this call as a mathcing context
              assert(RecursionRoot.verify());
              RecursionRoot.Context = NewContext;
              RecursionRoot.MatchingContexts.push_back(CallContext);
              assert(RecursionRoot.verify());

              // Reset the intraprocedural analysis from where we will resume
              // our analysis, using the widened context
              RecursionRoot.State = IntraproceduralAnalysis(Callee,
                                                            TheCache,
                                                            &GCBI,
                                                            NewContext);
            }
          } else {
            // Just a regular (uncached) function call, push it on the stack
            newFunction(Callee, CallContext);
          }

          break;
        }
      case AnalysisInterrupt::FunctionSummary:
        {
          ElementProxy Summary = Result.getFunctionSummary();
          LoadStoreLog Log = Summary.log();
          auto &Indexer = TheCache.getIndexer(Current.Function);

          DBG("sa-interp", {
              dbg << "We have a summary for "
                  << getName(Current.Function) << "\n";
            });

          assert(Current.verify(&Log, &Indexer));

          bool Changed = TheCache.update(Current.Function,
                                         Current.MatchingContexts,
                                         Log);

          for (Element &Context : Current.MatchingContexts) {
            if (!TheCache.get(Current.Function, Context).hasValue()) {
              dbg << "Couldn't find an entry we just inserted for function "
                  << getName(Current.Function)
                  << " in the cache using context:\n";
              Context.dump(M);
              dbg << "\n";
              abort();
            }
          }

          DBG("sa", {
              dbg << "Log for " << getName(Current.Function) << "\n";
              Log.dump(getModule(Entry));
              dbg << "Load log:\n";
              Indexer.dump(Log.Load);
            });

          if (Changed) {
            DBG("sa-interp", {
                dbg << "We have an improvement, let's reanalyze\n";
              });

            // Something changed, reset and re-run the analysis
            Current.State.initialize();
          } else {
            DBG("sa-interp", {
                dbg << "No improvement over the last analysis, we're OK\n";
              });

            // Is this the last iteration?
            if (InProgress.size() == 1) {
              return Result::createRegular(Current.State.getBranchesType(),
                                           Summary.computeCalleeSavedSlots());
            }

            // We're done here, let's go up one position in the stack
            InProgress.pop_back();
          }

          break;
        }
      default:
        assert(false);
      }
    }
  }

private:
  void newFunction(BasicBlock *Entry, Element Context) {
    InProgress.emplace_back(Context, Entry, TheCache, GCBI);
  }

private:
  Cache &TheCache;
  GeneratedCodeBasicInfo &GCBI;
  std::vector<WorkItem> InProgress;
};

namespace StackAnalysis {

char StackAnalysis::ID = 0;
static RegisterPass<StackAnalysis> X("sa",
                                     "Stack Analysis Pass",
                                     true,
                                     true);

bool StackAnalysis::runOnFunction(Function &F) {
  std::vector<BasicBlock *> Functions;

  for (BasicBlock &BB : F) {
    TerminatorInst *T = BB.getTerminator();
    assert(T != nullptr);
    MDNode *Node = T->getMetadata("revamb.jt.reasons");
    if (auto *Tuple = dyn_cast_or_null<MDTuple>(Node)) {
      for (Metadata *ReasonMD : Tuple->operands()) {
        if (cast<MDString>(ReasonMD)->getString() == "Callee"
            || cast<MDString>(ReasonMD)->getString() == "UnusedGlobalData") {
          Functions.push_back(&BB);
          break;
        }
      }
    }
  }

  Cache TheCache;

  //
  // For each function call identify where the return address is being stored
  //
  const Module *M = F.getParent();
  Function *FunctionCallFunction = M->getFunction("function_call");

  if (!FunctionCallFunction->user_empty()) {
    std::map<GlobalVariable *, unsigned> LinkRegisterStats;
    std::map<BasicBlock *, GlobalVariable *> LinkRegistersMap;
    for (User *U : FunctionCallFunction->users()) {
      if (auto *Call = dyn_cast<CallInst>(U)) {
        assert(isCallTo(Call, "function_call"));
        auto *LinkRegister = dyn_cast<GlobalVariable>(Call->getArgOperand(3));
        LinkRegisterStats[LinkRegister]++;

        // The callee might be unknown
        if (auto *BBA = dyn_cast<BlockAddress>(Call->getArgOperand(0))) {
          BasicBlock *Callee = BBA->getBasicBlock();
          assert(LinkRegistersMap.count(Callee) == 0
                 || LinkRegistersMap[Callee] == LinkRegister);
          LinkRegistersMap[Callee] = LinkRegister;
        }

      }
    }

    // Identify a default storage for the return address (the most common one)
    GlobalVariable *Default = nullptr;
    if (LinkRegisterStats.size() == 1) {
      Default = LinkRegisterStats.begin()->first;
    } else {
      std::pair<GlobalVariable *, unsigned> Max = { nullptr, 0 };
      for (auto &P : LinkRegisterStats) {
        if (P.second > Max.second)
          Max = P;
      }
      assert(Max.first != nullptr && Max.second != 0);
      Default = Max.first;
    }
    TheCache.setLinkRegisters(Default, std::move(LinkRegistersMap));
  }

  // Run the analysis
  std::stringstream Output;
  for (BasicBlock *BB : Functions) {
    Output << "Function " << getName(BB);

    auto &GCBI = getAnalysis<GeneratedCodeBasicInfo>();
    InterproceduralAnalysis SA(TheCache, GCBI, M);
    InterproceduralAnalysis::Result Result = SA.run(BB);

    if (Result.isFake())
      Output << " is fake";
    else if (Result.isNoReturn())
      Output << " does not return";
    Output << "\n";
    Output << "  BranchesType:";

    auto Sorted = Result.branchesType();
    using P = std::pair<BasicBlock *, AnalysisInterrupt::BranchType>;
    std::sort(Sorted.begin(),
              Sorted.end(),
              [] (const P &LHS, const P &RHS) {
                return LHS.first->getName() < RHS.first->getName();
              });
    for (const P &Pair : Sorted) {
      Output << " " << getName(Pair.first) << " (";
      Output << AnalysisInterrupt::getTypeName(Pair.second).str() << ")";
    }
    Output << "\n";

    if (Result.isRegular()) {
      Output << "  Saved slots:";
      for (ASSlot Slot : Result.calleeSaved()) {
        Output << " ";
        Slot.dump(M, Output);
      }
      Output << "\n";
    }

    Output << "\n";
  }

  TextRepresentation = Output.str();

  return false;
}

}
