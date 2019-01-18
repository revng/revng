/// \file interprocedural.cpp
/// \brief Implementation of the interprocedural portion of the stack analysis

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <chrono>

// Local libraries includes
#include "revng/Support/Statistics.h"

// Local includes
#include "Cache.h"
#include "InterproceduralAnalysis.h"

using llvm::BasicBlock;
using llvm::GlobalVariable;
using llvm::Instruction;
using llvm::Module;
using llvm::Optional;
using llvm::StringRef;

using time_point = std::chrono::steady_clock::time_point;
using StringIntCounter = CounterMap<std::string, uint64_t>;

Logger<> SaInterpLog("sa-interp");

/// \brief Logger for counting how much time is spent on a function
static StringIntCounter FunctionAnalysisTime("FunctionAnalysisTime");

/// \brief Logger for counting how many times a function is analyzed
static StringIntCounter FunctionAnalysisCount("FunctionAnalysisCount");

template<typename T>
static uint64_t nanoseconds(T Span) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(Span).count();
}

namespace StackAnalysis {

void InterproceduralAnalysis::push(BasicBlock *Entry) {
  InProgressFunctions.insert(Entry);
  InProgress.emplace_back(Entry,
                          TheCache,
                          &GCBI,
                          InProgressFunctions,
                          AnalyzeABI);
  FunctionAnalysisCount.push(Entry->getName().str());
}

void InterproceduralAnalysis::run(BasicBlock *Entry, ResultsPool &Results) {
  using IFS = IntraproceduralFunctionSummary;

  revng_assert(InProgress.size() == 0);

  Optional<const IFS *> Cached = TheCache.get(Entry);

  // Has this function been analyzed already? If so, only now we register it in
  // the ResultsPool.
  if (Cached) {
    FunctionType::Values Type;
    if (TheCache.isFakeFunction(Entry))
      Type = FunctionType::Fake;
    else if (TheCache.isIndirectTailCall(Entry))
      Type = FunctionType::IndirectTailCall;
    else if (TheCache.isNoReturnFunction(Entry))
      Type = FunctionType::NoReturn;
    else
      Type = FunctionType::Regular;

    // Regular functions need to be composed by at least a basic block
    const IFS &Summary = **Cached;
    if (Type == FunctionType::Regular)
      revng_assert(Summary.BranchesType.size() != 0);

    Results.registerFunction(Entry, Type, Summary);

    // We're done here
    return;
  }

  // Setup logger: each time we start a new intraprocedural analysis we indent
  // the output
  SaInterpLog.setIndentation(0);
  revng_log(SaInterpLog, "Running interprocedural analysis on " << Entry);

  // Push the request function in the worklist
  push(Entry);

  FunctionType::Values Type = FunctionType::Invalid;
  auto Result = Intraprocedural::Interrupt::createInvalid();

  // Loop over the worklist
  do {
    Type = FunctionType::Invalid;

    // Get an element from the stack (but don't pop yet)
    Analysis &Current = InProgress.back();

    SaInterpLog.setIndentation(InProgress.size());
    revng_log(SaInterpLog,
              "Analyzing function "
                << Current.entry() << " (size: " << Current.size()
                << " BBs, MustHit: " << Current.cacheMustHit() << ")");
    SaInterpLog.indent();

    time_point Begin = std::chrono::steady_clock::now();

    // Run/continue the intraprocedural analysis
    Result = Current.run();

    time_point End = std::chrono::steady_clock::now();
    FunctionAnalysisTime.push(Current.entry()->getName().str(),
                              nanoseconds(End - Begin));

    revng_assert(Result.requiresInterproceduralHandling());

    switch (Result.type()) {
    case BranchType::FakeFunction:
      revng_log(SaInterpLog, Current.entry() << " is fake");

      // It's a fake function, mark it as so and resume from the caller
      TheCache.markAsFake(Current.entry());

      // Set the function type in case this is the last in the worklist
      Type = FunctionType::Fake;

      // If it was recursive, pop until the recurions root (excluded, for now)
      if (const auto *Root = getRecursionRoot(Current.entry()))
        popUntil(Root);

      // Go back to the caller of the fake function.
      pop();

      // We basically evicted a function call which was supposed to hit the
      // cache. Reset the flag.
      Current.resetCacheMustHit();

      break;

    case BranchType::UnhandledCall: {
      BasicBlock *Callee = Result.getCallee();
      revng_assert(Callee != nullptr);

      revng_log(SaInterpLog,
                Current.entry() << " performs an unhandled call to " << Callee);

      // Is it recursive?
      if (getRecursionRoot(Callee) != nullptr) {
        if (SaInterpLog.isEnabled()) {
          SaInterpLog << Callee << " is recursive. Call stack:";
          for (const Analysis &WorkItem : InProgress)
            SaInterpLog << " " << WorkItem.entry();
          SaInterpLog << DoLog;
        }

        // We now have to inject in the cache a temporary entry.
        TheCache.update(Callee, IFS::bottom());

        // Ensure it was inserted in the cache
        revng_assert(TheCache.get(Callee));

        // At this point the intraprocedural analysis will resume employing
        // bottom for the recursive call. Then, once the analysis is done, the
        // interprocedural part will detect that the result associated with it
        // has changed (hopefully the result won't be bottom) and will run the
        // analysis again until we're stable.
      } else {
        // Just a regular (uncached) function call, push it on the stack
        push(Callee);
      }

      break;
    }

    case BranchType::IndirectTailCallFunction:
    case BranchType::NoReturnFunction:
    case BranchType::FunctionSummary: {
      const IFS &Summary = Result.getFunctionSummary();

      revng_log(SaInterpLog, "We have a summary for " << Current.entry());

      bool MustReanalyze = false;

      // Are there function calls that lead to a contradiction?
      const std::set<BasicBlock *> &Offending = Current.incoherentFunctions();
      if (Offending.size() != 0) {
        // If so, mark the called function as fake and re-analyze the caller
        for (BasicBlock *Entry : Offending) {
          revng_log(SaInterpLog,
                    Entry << " leads to contradiction, marking it as fake");

          revng_assert(Current.entry() != Entry);
          TheCache.markAsFake(Entry);
        }

        MustReanalyze = true;
      } else {
        // OK, no contradictions

        // TODO: we should probably move the cleanup earlier on
        // Perform some maintainance before recording in the cache
        IFS SummaryForCache = Summary.copy();
        SummaryForCache.FinalState.cleanup();

        // Let's register the result in the cache and check if we got any
        // changes w.r.t. to the last time we analyzed this function
        MustReanalyze = TheCache.update(Current.entry(), SummaryForCache);

        revng_assert(TheCache.get(Current.entry()));
      }

      if (SaLog.isEnabled()) {
        revng_log(SaLog, "FinalState for " << getName(Current.entry()));
        Summary.dump(getModule(Entry), SaLog);
      }

      if (MustReanalyze) {
        revng_log(SaInterpLog, "Something has changed, let's reanalyze");

        // Go back to the root of the recursion, if any
        if (const auto *Root = getRecursionRoot(Current.entry()))
          popUntil(Root);

        // Something changed, reset and re-run the analysis
        Current.initialize();

      } else {

        revng_log(SaInterpLog,
                  "No improvement over the last analysis, we're OK");

        switch (Result.type()) {
        case BranchType::IndirectTailCallFunction:
          revng_log(SaInterpLog,
                    Current.entry() << " ends with an indirect tail call");
          TheCache.markAsIndirectTailCall(Current.entry());
          Type = FunctionType::IndirectTailCall;
          break;

        case BranchType::NoReturnFunction:
          revng_log(SaInterpLog, Current.entry() << " doesn't return");
          TheCache.markAsNoReturn(Current.entry());
          Type = FunctionType::NoReturn;
          break;

        case BranchType::FunctionSummary:
          Type = FunctionType::Regular;
          break;

        default:
          revng_abort();
        }

        // We're done here, let's go up one position in the stack
        pop();
      }

    } break;

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
    case BranchType::Invalid:
      revng_abort("Unexpected branch type in interprocedural analysis");
    }

  } while (InProgress.size() > 0);

  revng_assert(Type != FunctionType::Invalid);
  Results.registerFunction(Entry, Type, Result.getFunctionSummary());
}

void ResultsPool::mergeFunction(BasicBlock *Function,
                                const IntraproceduralFunctionSummary &Summary) {
  using FRA = FunctionRegisterArgument;
  using FRV = FunctionReturnValue;
  using FCRA = FunctionCallRegisterArgument;
  using FCRV = FunctionCallReturnValue;

  const Module *M = getModule(Function);
  size_t CSVCount = std::distance(M->global_begin(), M->global_end());

  LocallyWrittenRegisters[Function] = Summary.WrittenRegisters;

  // Merge results from the arguments analyses
  const FunctionABI &ABI = Summary.ABI;
  auto &Slots = Summary.LocalSlots;
  for (auto &Slot : Slots) {
    int32_t Offset = Slot.first.offset();
    FunctionSlot Key = { Function, Offset };

    revng_assert(Slot.first.addressSpace() == ASID::cpuID()
                 and Offset <= static_cast<int32_t>(CSVCount));

    switch (Slot.second) {
    case LocalSlotType::UsedRegister:
    case LocalSlotType::ForwardedArgument:
    case LocalSlotType::ForwardedReturnValue:
      ABI.applyResults(FunctionRegisterArguments[Key], Offset);
      ABI.applyResults(FunctionReturnValues[Key], Offset);

      // Handle forwarded arguments/return values (push rax; pop rdx)
      if (Slot.second == LocalSlotType::ForwardedArgument
          && FunctionRegisterArguments[Key].value() == FRA::Yes) {
        FunctionRegisterArguments[Key] = FRA::maybe();
      }

      if (Slot.second == LocalSlotType::ForwardedReturnValue
          && (FunctionReturnValues[Key].value() == FRV::Yes
              || FunctionReturnValues[Key].value() == FRV::YesCandidate)) {
        FunctionReturnValues[Key] = FRV::maybe();
      }

      break;

    case LocalSlotType::ExplicitlyCalleeSavedRegister:
      FunctionRegisterArguments[Key] = FRA::no();
      FunctionReturnValues[Key] = FRV::no();
      ExplicitlyCalleeSavedRegisters[Function].insert(Slot.first.offset());
      break;
    }

    for (auto &P : CallSites) {
      CallSite Call = P.first;

      if (!Call.belongsTo(Function))
        continue;

      FunctionCallSlot K = { Call, Offset };
      Instruction *I = Call.callInstruction();
      FunctionCall TheCall = { getFunctionCallCallee(I->getParent()), I };

      switch (Slot.second) {
      case LocalSlotType::UsedRegister:
      case LocalSlotType::ForwardedArgument:
      case LocalSlotType::ForwardedReturnValue:

        ABI.applyResults(FunctionCallRegisterArguments[K], TheCall, Offset);
        ABI.applyResults(FunctionCallReturnValues[K], TheCall, Offset);

        // Handle forwarded arguments/return values (push rax; pop rdx)
        if (Slot.second == LocalSlotType::ForwardedArgument
            && FunctionCallRegisterArguments[K].value() == FCRA::Yes) {
          FunctionCallRegisterArguments[K] = FCRA::maybe();
        }

        if (Slot.second == LocalSlotType::ForwardedReturnValue
            && FunctionCallReturnValues[K].value() == FCRV::Yes) {
          FunctionCallReturnValues[K] = FCRV::maybe();
        }

        break;

      case LocalSlotType::ExplicitlyCalleeSavedRegister:
        FunctionCallRegisterArguments[K] = FunctionCallRegisterArgument::no();
        FunctionCallReturnValues[K] = FunctionCallReturnValue::no();
        break;
      }
    }
  }
}

void ResultsPool::mergeBranches(BasicBlock *Function,
                                const BasicBlockTypeMap &Branches) {
  // Merge information about the branches type
  for (auto &P : Branches)
    BranchesType[{ Function, P.first->getTerminator() }] = P.second;
}

void ResultsPool::mergeCallSites(BasicBlock *Entry,
                                 const StackSizeMap &ToImport) {
  for (auto &P : ToImport) {
    FunctionCalls[Entry].push_back(P.first);
    CallSite Call = { Entry, P.first.callInstruction() };
    auto It = CallSites.find(Call);
    if (It != CallSites.end()) {
      if (!compareOptional(It->second, P.second)) {
        revng_abort("This call site has a stack at a different height than"
                    " previously recorded");
      }
    } else {
      CallSites[Call] = P.second;
    }
  }
}

/// \brief Helper class for computing the set of registers clobbered by each
///        function
///
/// This class basically takes each function, computes the set of all the
/// registers written by it and all of the functions in the transitive closure
/// of the callee set and remove the registers that are explicitly callee saved
/// (ECS).
///
/// This process is repeated multiple times to handle with increasing precision
/// recursive and indirect function calls, which are initially ignored. After
/// the first iteration, the recursive function calls use the result from the
/// previous iteration. On the other hand, indirect function calls are
/// considered clobbering all the registers except those that are ECS in the
/// majority of the functions.
struct ClobberedRegistersAnalysis {
  using ClobberedMap = std::map<llvm::BasicBlock *, std::set<int32_t>>;

  /// \brief Struct representing the result of a single iteration
  struct IterationResult {
  public:
    /// \brief Description of a register
    struct ECSVote {
    public:
      unsigned ECS; ///< Number of functions in which this register is ECS
      unsigned Total; ///< Number of functions writing this register

    public:
      bool operator==(const ECSVote &Other) const {
        return std::tie(ECS, Total) == std::tie(Other.ECS, Other.Total);
      }

      bool isECS() const { return ECS > Total / 2; }
    };

  public:
    ClobberedMap Clobbered;
    std::map<int32_t, ECSVote> ECSVotes;

  public:
    bool operator==(const IterationResult &Other) const {
      using std::tie;
      return tie(Clobbered, ECSVotes) == tie(Other.Clobbered, Other.ECSVotes);
    }

    bool operator!=(const IterationResult &Other) const {
      return not(*this == Other);
    }
  };

  /// \brief Compute an iteration
  static IterationResult
  recompute(ResultsPool &This, const ClobberedMap &InitialState) {
    // The results pool
    IterationResult Result;
    auto &Clobbered = Result.Clobbered;
    auto &ECSVotes = Result.ECSVotes;

    // Loop over all the functions
    for (auto &P : This.FunctionTypes) {
      BasicBlock *Function = P.first;

      // Have we handled this already?
      if (Clobbered.count(Function) != 0)
        continue;

      using iterator = typename std::vector<FunctionCall>::const_iterator;

      struct State {
      public:
        BasicBlock *Function;
        iterator CallIt;
        iterator EndCallIt;

      public:
        State(BasicBlock *Function, iterator CallIt, iterator EndCallIt) :
          Function(Function),
          CallIt(CallIt),
          EndCallIt(EndCallIt) {}
      };

      // Worklist
      std::vector<State> WorkList;

      // Set of currently in-progress functions, used to detect recursive
      // function calls
      std::set<BasicBlock *> InProgress;

      // Initialize the worklist
      const auto &FunctionCallsList = This.FunctionCalls[Function];
      WorkList.emplace_back(Function,
                            FunctionCallsList.begin(),
                            FunctionCallsList.end());
      InProgress.insert(Function);

      // Loop over the worklist
      while (not WorkList.empty()) {
        // Peek but don't pop
        State &Current = WorkList.back();
        BasicBlock *Function = Current.Function;

        // Get a reference to the results for the current function
        std::set<int32_t> &CurrentClobbered = Clobbered[Function];

        // Loop over the unprocessed function calls
        while (Current.CallIt != Current.EndCallIt) {
          // Get the callee
          BasicBlock *Callee = Current.CallIt->callee();

          if ((Callee == nullptr) or (InProgress.count(Callee) != 0)) {
            // Indirect or recursive function call, use result from last
            // iteration
            auto It = InitialState.find(Callee);
            if (It != InitialState.end())
              CurrentClobbered.insert(It->second.begin(), It->second.end());

          } else {
            // Do we already handle this callee?
            auto ClobberedIt = Clobbered.find(Callee);
            if (ClobberedIt == Clobbered.end()) {
              // No, push it on the worklist
              const auto &FunctionCallsList = This.FunctionCalls[Callee];
              WorkList.emplace_back(Callee,
                                    FunctionCallsList.begin(),
                                    FunctionCallsList.end());
              InProgress.insert(Callee);

              // Early exit so we can proceed from the callee
              break;
            }

            // OK, we already processed this callee

            // Merge in the clobbered set all those clobbered by the callee
            CurrentClobbered.insert(ClobberedIt->second.begin(),
                                    ClobberedIt->second.begin());
          }

          // Proceed to the next call site
          Current.CallIt++;
        }

        // Are we done?
        if (Current.CallIt == Current.EndCallIt) {
          // Oh, we're done

          {
            // Add all the locally written registers
            auto It = This.LocallyWrittenRegisters.find(Function);
            if (It != This.LocallyWrittenRegisters.end())
              CurrentClobbered.insert(It->second.begin(), It->second.end());
          }

          // Increase the counter associated to each written register
          for (int32_t Index : CurrentClobbered)
            ECSVotes[Index].Total++;

          {
            // Erase from the clobbered registers all the callee-saved, if any
            auto It = This.ExplicitlyCalleeSavedRegisters.find(Function);
            if (It != This.ExplicitlyCalleeSavedRegisters.end()) {
              // Do not use CurrentClobbered.erase(BeginIt, EndIt);
              for (int32_t Index : It->second)
                CurrentClobbered.erase(Index);

              // Increase the counter associated to each register
              for (int32_t Index : It->second)
                ECSVotes[Index].ECS++;
            }
          }

          // Pop from the worklist
          WorkList.pop_back();
          InProgress.erase(Function);
        }
      }
    }

    return Result;
  }

  /// \brief Repeat the analysis until a fixed point is reached
  static ClobberedMap run(ResultsPool &This) {
    IterationResult LastResult;
    IterationResult NewResult;

    do {
      LastResult = std::move(NewResult);

      // Use as initial state the previous iteration's state
      NewResult = recompute(This, LastResult.Clobbered);

      // Perform a majority vote on the result to associate to indirect function
      // calls
      std::set<int32_t> IndirectCallClobbered;
      for (auto &P : NewResult.ECSVotes) {

        // When a register is written, is it usually an ECS?
        if (not P.second.isECS()) {
          // No, consider it clobbered by indirect function calls
          IndirectCallClobbered.insert(P.first);
        }
      }

      // Assert the new set of registers clobbered by indirect function calls
      // contains at least all of the registers clobbered in the previous
      // iteration. If this is not the case, the algorithm might not converge.
      for (int32_t Clobbered : LastResult.Clobbered[nullptr])
        revng_assert(IndirectCallClobbered.count(Clobbered) != 0);

      // Save the results of the vote as the result associated with nullptr
      NewResult.Clobbered[nullptr] = std::move(IndirectCallClobbered);

    } while (LastResult != NewResult);

    return std::move(NewResult.Clobbered);
  }
};

FunctionsSummary ResultsPool::finalize(Module *M) {
  ASID CPU = ASID::cpuID();

  std::vector<GlobalVariable *> IndexToCSV;
  IndexToCSV.push_back(nullptr);
  for (GlobalVariable &CSV : M->globals())
    IndexToCSV.push_back(&CSV);

  FunctionsSummary Result;

  // Set function types
  for (auto &P : FunctionTypes)
    Result.Functions[P.first].Type = P.second;

  // Compute the set of registers clobbered by each function
  ClobberedRegistersAnalysis::ClobberedMap Clobbered;
  Clobbered = ClobberedRegistersAnalysis::run(*this);
  for (auto &P : Clobbered) {
    auto &Function = Result.Functions[P.first];
    for (int32_t Offset : P.second)
      Function.ClobberedRegisters.insert(IndexToCSV.at(Offset));
  }

  // Register block types
  for (auto &P : BranchesType) {
    BasicBlock *BB = P.first.branch()->getParent();
    Result.Functions[P.first.entry()].BasicBlocks[BB] = P.second;
  }

  // Initialize the information about functions in Result
  for (auto &P : FunctionRegisterArguments) {
    auto &Function = Result.Functions[P.first.first];
    GlobalVariable *CSV = IndexToCSV.at(P.first.second);
    Function.RegisterSlots[CSV].Argument = P.second;
  }

  for (auto &P : FunctionReturnValues) {
    auto &Function = Result.Functions[P.first.first];
    GlobalVariable *CSV = IndexToCSV.at(P.first.second);
    Function.RegisterSlots[CSV].ReturnValue = P.second;
  }

  // Collect, for each call site, all the slots
  std::map<CallSite, std::set<ASSlot>> FunctionCallSlots;
  for (auto &P : FunctionCallRegisterArguments) {
    auto Slot = ASSlot::create(CPU, P.first.second);
    FunctionCallSlots[P.first.first].insert(Slot);
  }
  for (auto &P : FunctionCallReturnValues) {
    auto Slot = ASSlot::create(CPU, P.first.second);
    FunctionCallSlots[P.first.first].insert(Slot);
  }

  //
  // Merge information about a function and all the call sites targeting it
  //

  // For each function call
  for (auto &P : FunctionCallSlots) {
    CallSite Call = P.first;
    revng_assert(CallSites.count(Call) != 0);
    Instruction *I = Call.callInstruction();
    BasicBlock *Callee = getFunctionCallCallee(I->getParent());
    bool UnknownCallee = (Callee == nullptr);

    // Register this call site among the call sites of the caller
    auto &CallerCallSites = Result.Functions[Call.caller()].CallSites;
    CallerCallSites.emplace_back(I, Callee);
    auto &NewCallSite = CallerCallSites.back();

    // For each slot in this function call
    for (ASSlot Slot : P.second) {
      int32_t Offset = Slot.offset();
      FunctionCallSlot FCS{ Call, Offset };

      revng_assert(Slot.addressSpace() == CPU);
      FunctionSlot TheFunctionSlot{ Callee, Offset };

      // Take the result of the analyses for the current call site
      auto &FCRA = FunctionCallRegisterArguments;
      revng_assert(FCRA.count(FCS) != 0);
      const auto &CallerRegisterArgument = FCRA[FCS];

      auto &FCRV = FunctionCallReturnValues;
      revng_assert(FCRV.count(FCS) != 0);
      const auto &CallerReturnValue = FCRV[FCS];

      GlobalVariable *CSV = IndexToCSV.at(Offset);
      NewCallSite.RegisterSlots[CSV].Argument = CallerRegisterArgument;
      NewCallSite.RegisterSlots[CSV].ReturnValue = CallerReturnValue;

      // Check if the register is used at all in this function
      auto &FRA = FunctionRegisterArguments;
      auto &FRV = FunctionReturnValues;
      auto &Register = Result.Functions[Callee].RegisterSlots[CSV];
      bool CalleeHasSlot = FRA.count(TheFunctionSlot) != 0;

      if (not UnknownCallee and CalleeHasSlot) {
        // The register is used and the callee is available, collect the
        // information on the called function
        const auto &Argument = FRA[TheFunctionSlot];
        const auto &ReturnValue = FRV[TheFunctionSlot];

        // Update the information on the caller side incorporating those
        // coming from the called function
        NewCallSite.RegisterSlots[CSV].Argument.combine(Argument);
        NewCallSite.RegisterSlots[CSV].ReturnValue.combine(ReturnValue);

        // Update the information on the called function incorporating the
        // information coming from the current call site
        Register.Argument.combine(CallerRegisterArgument);
        Register.ReturnValue.combine(CallerReturnValue);
      } else {
        // TODO: Closed World Assumption
        auto &CallSiteRegister = NewCallSite.RegisterSlots[CSV];

        // Either the callee is unknown or does not use this slot
        CallSiteRegister.Argument.notAvailable();
        CallSiteRegister.ReturnValue.notAvailable();

        Register.Argument.combine(CallSiteRegister.Argument);
        Register.ReturnValue.combine(CallSiteRegister.ReturnValue);
      }
    }
  }

  return Result;
}

} // namespace StackAnalysis
