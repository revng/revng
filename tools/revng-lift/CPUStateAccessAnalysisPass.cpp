/// \file cpustateaccessanalysis.cpp
/// \brief This file performs an analysis for reconstructing the access
///        patterns to the CPU State Variables (CSV).

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <sstream>
#include <stack>
#include <string>
#include <vector>

// LLVM includes
#include "llvm/ADT/SmallSet.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

// Local libraries includes
#include "revng/Support/Debug.h"
#include "revng/Support/IRHelpers.h"

// Local includes
#include "CPUStateAccessAnalysisPass.h"
#include "VariableManager.h"

namespace llvm {
class DataLayout;
}

using namespace llvm;
using ConstFunctionPtrSet = std::set<const Function *>;
using InstrPtrSet = std::set<Instruction *>;
using CallPtrSet = std::set<CallInst *>;
using ConstValuePtrSet = std::set<const Value *>;

/// \brief Logger for forwardTaintAnalysis
static auto TaintLog = Logger<>("cpustate-taint-analysis");
/// \brief Logger for the creation of WorkItem
static auto CSVAccessLog = Logger<>("cpustate-access-analysis");
/// \brief Logger for fixing the accesses to CPUState
static auto FixAccessLog = Logger<>("cpustate-fix-access");

static uint64_t NumUnknown = 0;
static std::map<std::string, uint64_t> FunToNumUnknown;
static std::map<std::string, std::set<std::string>> FunToUnknowns;

void writeToLog(Logger<true> &L, const CSVOffsets &O, int /*Ignore*/) {
  L << "Kind: " << CSVOffsets::toString(O.OffsetKind);
  L << " Offsets = { ";
  for (const auto &Offset : O)
    L << Offset << ' ';
  L << "}";
}

/// \brief Computes the set of Functions reachable from a given Function through
///        direct calls.
///
/// \param RootFunction is a pointer to the Function from which is the
///        starting point for computing reachability.
/// \return set of pointers to the reachable Functions
///
/// This function can probably be implemented using the call graph utilities
/// already present in LLVM, but when I tried it I spent a day doing it and I
/// gave up because of bugs in the functions coming from LLVM. Then I
/// implemented my own version, i.e. this function
static ConstFunctionPtrSet
computeDirectlyReachableFunctions(const Function *RootFunction) {
  std::map<const Function *, ConstFunctionPtrSet> CallGraph;
  const Module &M = *RootFunction->getParent();

  // Initialize empty CallGraph
  for (const Function &F : M)
    CallGraph[&F] = {};

  for (const Function &F : M) {
    for (const Use &U : F.uses()) {
      const User *TheUser = U.getUser();
      if (const auto *TheCall = dyn_cast<const CallInst>(TheUser)) {
        const Function *Caller = TheCall->getParent()->getParent();
        const Function *Callee = getCallee(TheCall);
        if (Callee == &F) {
          CallGraph[Caller].insert(Callee);
        }
      } else if (const auto *CExpr = dyn_cast<const ConstantExpr>(TheUser)) {
        revng_assert(CExpr->getOpcode() == Instruction::BitCast);
        SmallSet<const ConstantExpr *, 10> CurBitCasts;
        SmallSet<const ConstantExpr *, 10> NextBitCasts;
        CurBitCasts.insert(CExpr);
        while (not CurBitCasts.empty()) {
          NextBitCasts.clear();
          for (const ConstantExpr *BitCast : CurBitCasts) {
            for (const User *BitCastUser : BitCast->users()) {
              const auto *TheCall = dyn_cast<const CallInst>(BitCastUser);
              const auto *NewCExpr = dyn_cast<const ConstantExpr>(BitCastUser);
              if (TheCall) {
                const Function *Caller = TheCall->getParent()->getParent();
                const Function *Callee = getCallee(TheCall);
                if (Callee == &F) {
                  CallGraph[Caller].insert(Callee);
                }
              } else if (NewCExpr) {
                revng_assert(NewCExpr->getOpcode() == Instruction::BitCast);
                NextBitCasts.insert(NewCExpr);
              }
            }
          }
          std::swap(CurBitCasts, NextBitCasts);
        }
      }
    }
  }

  ConstFunctionPtrSet ReachableFunctions = { RootFunction };
  ConstFunctionPtrSet CurrentChildren = { RootFunction };
  ConstFunctionPtrSet NextChildren;
  while (not CurrentChildren.empty()) {
    NextChildren.clear();
    for (const Function *F : CurrentChildren) {
      for (const Function *Callee : CallGraph.at(F)) {
        bool NewInsertion = ReachableFunctions.insert(Callee).second;
        if (NewInsertion)
          NextChildren.insert(Callee);
      }
    }
    std::swap(CurrentChildren, NextChildren);
  }

  return ReachableFunctions;
}

struct TaintResults {

  // A set of Instructions that access the CSV to load data. They can be
  // LoadInst or CallInst to Instrinsic::memcpy for which the size is known.
  InstrPtrSet TaintedLoads;

  // A set of Instructions that access the CSV to store data. They can be
  // LoadInst or CallInst to Instrinsic::memcpy for which the size is known.
  InstrPtrSet TaintedStores;

  // A set of Values that are tainted during the analysis.
  ConstValuePtrSet TaintedValues;

  // A set of CallInst that are considered illegal. These include indirect
  // calls, calls to functions without body, and calls to Intrinsinc::memcpy
  // with unknown size. They are considered illegal because we have no way of
  // knowing how they access the CSV.
  CallPtrSet IllegalCalls;

  bool empty() const noexcept {
    return TaintedLoads.empty() and TaintedStores.empty()
           and IllegalCalls.empty();
  }
};

/// \brief Interprocedural forward taint analysis.
//
/// \param CPUStatePtr is a pointer to the CPU State Variable, which is a global
///        variable. This variable is the Value that taints all the others.
/// \param ReachableFunctions is a set of functions that are reachable from
///        the root function. The analysis is restricted to those functions.
/// \return a TaintResults containing information on:
///         1) a set of instructions that access the CSV to load data;
///         2) a set of instructions that access the CSV to store data;
///         3) a set of illegal calls, for which it is impossible to understand
///            if and how they will access the CSV.
static TaintResults
forwardTaintAnalysis(GlobalVariable *CPUStatePtr,
                     const ConstFunctionPtrSet &ReachableFunctions) {
  //
  // Interprocedural Forward Taint Analysis
  //

  // This analysis aims to understand all the Values that are affected by the
  // CPUStatePtr pointer, that points to the CPUStateVariables (CSVs).
  // The main idea is that we explore the use chains depth first, propagating
  // interprocedurally when we find a Use whose User is a CallInstr.
  //
  // The function is structured as follows:
  // 1. Iterate on the users of `CPUStatePtr`
  //   2. For each user of `CPUStatePtr`, consider its next user,
  //      building a WorkList of `Use`s in exploration
  //   3. If we find unexplored uses keep pushing them on the WorkList. If we
  //      find a Load or a Store we taint it and don't push anything on the
  //      WorkList. If it's not a Load or Store we mark it as tainted separately
  //      3(a) If the next `Use` to explore is a `CallInst` the taint is
  //      propagated interprocedurally to the callee, through the arguments
  //      (propagation from caller to callee)
  //      3(b) If the next `Use` to explore is a `RetInst` the taint is
  //      propagated interprocedurally to the Function.
  //      (propagation from the callee to all call sites)
  //   4. If we didn't push anything on the WorkList we can start exploring the
  //      other `Use`s of the item that is currently on top of the WorkList
  //   5. If we didn't push anything on the WorkList we can start popping `Use`s
  //      from the WorkList, until we reach a `Value` that still has unexplored
  //      `Use`s
  //      5(a) If we're popping an argument of a function this means that we've
  //      finished analyzing the uses of that argument. We have to make sure
  //      that, if the taint reached the return instructions in the function,
  //      the taint is propagated to the call sites.
  //   6. After popping the top of the WorkList in 5., if the new top of the
  //      WorkList still has unexplored uses start to explore them.

  TaintResults Results;
  std::set<std::pair<const Function *, const Argument *>>
    FunctionArgTaintsReturn;

  revng_assert(CPUStatePtr != nullptr);
  revng_assert(CPUStatePtr->getType()->isPointerTy());

  struct CallSiteInfo {
    const CallInst *CallSite;
    const Argument *Arg;
    const unsigned ArgNo;
    CallSiteInfo(const CallInst *C, const Argument *A, const unsigned N) :
      CallSite(C),
      Arg(A),
      ArgNo(N) {}
  };

  Module *M = CPUStatePtr->getParent();
  if (TaintLog.isEnabled()) {
    TaintLog << "MODULE:" << DoLog;
    TaintLog << dumpToString(M) << DoLog;
  }

  // 1. Iterate on the users of `CPUStatePtr`
  for (const User *U : CPUStatePtr->users()) {

    // During the analysis we keep two stacks.
    // ToTaintWorkList is a stack representing the Values currently enqued that
    // must be tainted and for which we still have to analyze the uses.
    // CallSites is a stack representing the CallInst that we entered for
    // interprocedural propagation.
    std::stack<const Use *> ToTaintWorkList;
    std::stack<CallSiteInfo> CallSiteInfos;

    // Sanity check for the uses of CPUStatePtr.
    // They must all be direct Loads from CPUStatePtr.
    const auto *Load = cast<const LoadInst>(U);
    revng_assert(Load->getPointerOperand() == CPUStatePtr);

    // Push the first use on the WorkList
    const Function *F = Load->getParent()->getParent();
    if (Load->getNumUses() != 0
        and ReachableFunctions.find(F) != ReachableFunctions.end()) {
      if (TaintLog.isEnabled()) {
        TaintLog << "Tainted origin: " << Load << DoLog;
        TaintLog << dumpToString(Load) << DoLog;
        TaintLog.indent();
      }
      ToTaintWorkList.push(&*Load->use_begin());
    }

    //   2. For each user of `CPUStatePtr`, consider its next user,
    //      building a WorkList of `Use`s in exploration
    while (not ToTaintWorkList.empty()) {
      const Use *TheUse = ToTaintWorkList.top();
      revng_assert(TheUse != nullptr);
      auto *TheUser = cast<Instruction>(TheUse->getUser());
      const auto OpCode = TheUser->getOpcode();
      if (TaintLog.isEnabled()) {
        TaintLog << "Inst: " << TheUser << DoLog;
        TaintLog << dumpToString(TheUser) << DoLog;
      }

      const size_t Size = ToTaintWorkList.size();

      // 3. If we find unexplored uses keep pushing them on the WorkList. If we
      // find a Load or a Store we taint it and don't push anything on the
      // WorkList. If it's not a Load or Store we mark it as tainted separately

      // This switch explores the use-chains depth-first, pushing unexplored
      // uses on the ToTaintWorkList if necessary.
      unsigned OperandNo = TheUse->getOperandNo();
      switch (OpCode) {
      case Instruction::Load: {
        TaintLog << "LOAD" << DoLog;

        revng_assert(OperandNo == LoadInst::getPointerOperandIndex());
        auto *L = cast<LoadInst>(TheUser);
        if (TheUse->get() == L->getPointerOperand()) {
          if (TaintLog.isEnabled()) {
            TaintLog << "TAINT: " << TheUser << DoLog;
            TaintLog << dumpToString(TheUser) << DoLog;
          }
          Results.TaintedLoads.insert(TheUser);
        }
      } break;
      case Instruction::Store: {
        TaintLog << "STORE" << DoLog;
        revng_assert(OperandNo == StoreInst::getPointerOperandIndex());
        auto *S = cast<StoreInst>(TheUser);
        if (TheUse->get() == S->getPointerOperand()) {
          if (TaintLog.isEnabled()) {
            TaintLog << "TAINT: " << TheUser << DoLog;
            TaintLog << dumpToString(TheUser) << DoLog;
          }
          Results.TaintedStores.insert(TheUser);
        }
      } break;
      case Instruction::Trunc:
      case Instruction::ZExt:
      case Instruction::SExt:
      case Instruction::BitCast:
      case Instruction::IntToPtr:
      case Instruction::PtrToInt:
      case Instruction::GetElementPtr:
      case Instruction::PHI:
      case Instruction::Add: {
        TaintLog << "OP" << DoLog;
        auto OperandId = GetElementPtrInst::getPointerOperandIndex();
        revng_assert(OpCode != Instruction::GetElementPtr
                     or TheUse->getOperandNo() == OperandId);

        // Taint TheUser, and if this is the first time we taint it we also push
        // on the ToTaintWorkList its first use that is not tainted
        bool JustTainted = Results.TaintedValues.insert(TheUser).second;
        if (TaintLog.isEnabled()) {
          TaintLog << "TAINT: " << TheUser << DoLog;
          TaintLog << dumpToString(TheUser) << DoLog;
        }
        if (JustTainted) {
          TaintLog << "Just Tainted" << DoLog;
          for (const Use &U : TheUser->uses()) {
            TaintLog << "User: " << U.getUser() << DoLog;
            if (Results.TaintedValues.count(U.getUser()) == 0) {
              TaintLog << "PUSH" << DoLog;
              ToTaintWorkList.push(&U);
              TaintLog.indent();
              break;
            }
          }
        }
      } break;
      case Instruction::Call: {

        // 3(a) If the next `Use` to explore is a `CallInst` the taint is
        // propagated interprocedurally to the callee, through the arguments
        // (propagation from caller to callee)
        TaintLog << "CALL" << DoLog;
        auto *TheCall = cast<CallInst>(TheUser);
        Function *Callee = getCallee(TheCall);

        // Indirect calls, calls to functions without body, and calls to
        // Intrinsic::memcpy with non-constant size are considered illegal,
        // because we cannot know how they will affect the CPU State
        if (Callee == nullptr) {
          TaintLog << "Illegal -- indirect call" << DoLog;
          Results.IllegalCalls.insert(TheCall);
          break;
        }
        if (Callee->getIntrinsicID() == Intrinsic::memcpy) {
          unsigned OpNo = TheUse->getOperandNo();
          revng_assert(OpNo == 0 or OpNo == 1);
          if (isa<ConstantInt>(TheCall->getArgOperand(2))) {
            if (OpNo == 0) {
              if (TaintLog.isEnabled()) {
                TaintLog << "TAINT: " << TheUser << DoLog;
                TaintLog << dumpToString(TheUser) << DoLog;
              }
              Results.TaintedStores.insert(TheUser);
            }
            if (OpNo == 1) {
              if (TaintLog.isEnabled()) {
                TaintLog << "TAINT: " << TheUser << DoLog;
                TaintLog << dumpToString(TheUser) << DoLog;
              }
              Results.TaintedLoads.insert(TheUser);
            }
          } else {
            TaintLog << "Illegal -- unknwon size memcpy" << DoLog;
            Results.IllegalCalls.insert(TheCall);
          }
          break;
        } else if (Callee->empty()) {
          TaintLog << "Illegal -- no body" << DoLog;
          Results.IllegalCalls.insert(TheCall);
          break;
        }
        revng_assert(ReachableFunctions.count(Callee) != 0);

        // Select the correct formal argument associated with this use
        const Argument *FormalArgument = nullptr;
        unsigned ArgNo = 0;
        for (const Argument &Arg : Callee->args()) {
          ArgNo = Arg.getArgNo();
          if (TheUse->getOperandNo() == ArgNo) {
            FormalArgument = &Arg;
            break;
          }
        }
        TaintLog << "Found Argument" << DoLog;
        revng_assert(FormalArgument != nullptr);

        // Taint the Argument, and if this is the first time we taint it we
        // also push on the ToTaintWorkList its first use that is not tainted.
        if (TaintLog.isEnabled()) {
          TaintLog << "Argument: " << FormalArgument << DoLog;
          TaintLog << dumpToString(FormalArgument) << DoLog;
        }
        bool JustTainted = Results.TaintedValues.insert(FormalArgument).second;
        if (JustTainted) {
          TaintLog << "Just Tainted" << DoLog;
          for (const Use &U : FormalArgument->uses()) {
            if (TaintLog.isEnabled()) {
              TaintLog << "User: " << U.getUser() << DoLog;
              TaintLog << dumpToString(U.getUser()) << DoLog;
            }
            if (Results.TaintedValues.count(U.getUser()) == 0) {
              TaintLog << "PUSH" << DoLog;
              ToTaintWorkList.push(&U);
              TaintLog.indent();

              // We also push the call to the CallSites stack, because if we put
              // the uses of the argument on the ToTaintWorkList we are actually
              // starting to perform the analysis inside the callee.
              CallSiteInfos.push(CallSiteInfo(TheCall, FormalArgument, ArgNo));
              break;
            }
          }
        } else if (FunctionArgTaintsReturn.count({ Callee, FormalArgument })) {
          // It means that a previous exploration of the graph has reached this
          // argument, and propagated a taint until a return value. This means
          // that this we must taint the call and start explorint its unexplored
          // users if any.
          bool JustTainted = Results.TaintedValues.insert(TheUser).second;
          if (TaintLog.isEnabled()) {
            TaintLog << "TAINT: " << TheUser << DoLog;
            TaintLog << dumpToString(TheUser) << DoLog;
          }
          if (JustTainted) {
            TaintLog << "Just Tainted" << DoLog;
            for (const Use &U : TheUser->uses()) {
              TaintLog << "User: " << U.getUser() << DoLog;
              if (Results.TaintedValues.count(U.getUser()) == 0) {
                TaintLog << "PUSH" << DoLog;
                ToTaintWorkList.push(&U);
                TaintLog.indent();
                break;
              }
            }
          }
        }
      } break;
      case Instruction::Ret: {

        // 3(b) If the next `Use` to explore is a `RetInst` the taint is
        // propagated interprocedurally to the Function.
        // (propagation from the callee to all call sites)
        TaintLog << "RET" << DoLog;
        revng_assert(not CallSiteInfos.empty());

        // Taint the return instruction, then, if this is the first time that we
        // taint also the call site, so that it's marked for propagation of the
        // taint analysis to its uses.
        if (TaintLog.isEnabled()) {
          TaintLog << "TAINT: " << TheUser << DoLog;
          TaintLog << dumpToString(TheUser) << DoLog;
        }
        bool JustTainted = Results.TaintedValues.insert(TheUser).second;
        if (JustTainted) {
          const CallSiteInfo &CSInfo = CallSiteInfos.top();

          Results.TaintedValues.insert(CSInfo.CallSite);

          const Function *Callee = getCallee(CSInfo.CallSite);
          FunctionArgTaintsReturn.insert({ Callee, CSInfo.Arg });

          if (TaintLog.isEnabled()) {
            TaintLog << "TAINT: " << CSInfo.CallSite << DoLog;
            TaintLog << dumpToString(CallSiteInfos.top().CallSite) << DoLog;
            std::string Name = getCallee(CSInfo.CallSite)->getName();
            TaintLog << "pair: < " << Name << ", " << CSInfo.ArgNo << " > "
                     << DoLog;
          }
        }
      } break;
      case Instruction::Switch:
      case Instruction::ICmp:
      case Instruction::And:
      case Instruction::Or:
        break;
      default:
        revng_abort();
      }

      // If we pushed something new on the ToTaintWorkList we want to keep
      // exploring its uses until we reach a leaf.
      if (Size < ToTaintWorkList.size())
        continue;
      TaintLog << "not grown" << DoLog;

      // 4. If we didn't push anything on the WorkList we can start exploring
      // the other `Use`s of the item that is currently on top of the WorkList
      if (Size == ToTaintWorkList.size()) {
        Use *NextUse = TheUse->getNext();
        if (NextUse != nullptr) {
          TaintLog << "advance" << DoLog;
          ToTaintWorkList.top() = NextUse;
          continue;
        }
      }

      TaintLog << "Done" << DoLog;

      // 5. If we didn't push anything on the WorkList we can start popping
      //    `Use`s from the WorkList, until we reach a `Value` that still has
      //    unexplored `Use`s
      //
      // If we reach this point we have finished to explore all the uses of the
      // item that is currently on top of the ToTaintWorkList stack.
      // We want to pop it and to handle Arguments in a special way.
      const Use *UnexploredUse = nullptr;
      while (not ToTaintWorkList.empty() and UnexploredUse == nullptr) {
        const Use *PoppedTopUse = ToTaintWorkList.top();
        if (TaintLog.isEnabled()) {
          TaintLog << "POP : " << PoppedTopUse->get() << DoLog;
          TaintLog << dumpToString(PoppedTopUse->get()) << DoLog;
        }
        if (TaintLog.isEnabled()) {
          TaintLog << "PoppedUser : " << PoppedTopUse->getUser() << DoLog;
          TaintLog << dumpToString(PoppedTopUse->getUser()) << DoLog;
        }
        ToTaintWorkList.pop();
        TaintLog.unindent();
        Argument *Arg = dyn_cast<Argument>(PoppedTopUse->get());
        bool PoppedHasAllExploredSources = PoppedTopUse->getNext() == nullptr;
        if (Arg and PoppedHasAllExploredSources) {
          // 5(a) If we're popping an argument of a function this means that we
          // finished analyzing the uses of that argument. We have to make sure
          // that, if the taint reached the return instructions in the function,
          // the taint is propagated to the call sites.
          TaintLog << "Finish Argument" << DoLog;

          const CallSiteInfo &CSInfo = CallSiteInfos.top();
          revng_assert(CSInfo.Arg == Arg);

          unsigned ArgNo = Arg->getArgNo();
          revng_assert(CSInfo.ArgNo == ArgNo);

          const User *ArgUser = ToTaintWorkList.top()->getUser();
          const auto *CallSite = cast<const CallInst>(ArgUser);
          const Function *Callee = getCallee(CallSite);

          if (TaintLog.isEnabled()) {
            TaintLog << "CallSite: " << CallSite << DoLog;
            TaintLog << dumpToString(CallSite) << DoLog;
          }

          // If the CallSite was tainted it means that the taint analysis
          // reached at least one of the return values of the callee.
          // Hence, the taint can propagate to the uses of the call.
          // The same holds if we already know that the pair (Callee, ArgNo)
          // taints the return.
          if (Results.TaintedValues.count(CallSite) != 0
              or FunctionArgTaintsReturn.count({ Callee, Arg }) != 0) {
            if (CallSite->getNumUses() != 0) {
              UnexploredUse = &*CallSite->use_begin();
            }
          }

          // If PoppedTopUse is an Use of an Argument the next available Use on
          // the stack must be an argument of a CallInst. This CallInst was the
          // call from where we started analyzing the uses of the function
          // Argument for which we just popped the PoppedTopUse.
          CallSiteInfos.pop();
        } else {
          TaintLog << "NOT Finished or NOT Argument" << DoLog;
          UnexploredUse = PoppedTopUse->getNext();
        }
      }

      // 6. After popping the top of the WorkList in 5., if the new top of the
      //    WorkList still has unexplored uses start to explore them.

      // If we have a new UnexploredUse we push it and we continue
      // because that's the new use that must be analyzed.
      if (UnexploredUse != nullptr) {
        TaintLog << "PUSH: " << UnexploredUse->get() << DoLog;
        TaintLog << "User: " << UnexploredUse->getUser() << DoLog;
        ToTaintWorkList.push(UnexploredUse);
        TaintLog.indent();
      }
    }
  }

  QuickMetadata QMD(M->getContext());
  for (CallInst *Call : Results.IllegalCalls) {
    CallInst *Abort = CallInst::Create(M->getFunction("abort"), {}, Call);
    auto IllegalCallsMDKind = M->getContext().getMDKindID("IllegalCalls");
    Abort->setMetadata(IllegalCallsMDKind, QMD.tuple((uint32_t) 0));
  }
  return Results;
}

class WorkItem {

public:
  using size_type = SmallVector<const Use *, 3>::size_type;

private:
  // The value whose sources we're analyzing
  Value *CurrentValue;

  // Sources are kind of the opposite of `Use`s. Every pointer in this vector
  // points to a `Use` whose `User` is the `Value` pointed by the `CurrentValue`
  // member of this `WorkItem`.
  SmallVector<const Use *, 3> Sources;

  // The index of the Source that is currently considered for the analysis
  size_type SourceIndex;

public:
  WorkItem() : CurrentValue(nullptr), Sources(), SourceIndex(0) {}

  explicit WorkItem(Instruction *I) :
    CurrentValue(I),
    Sources(),
    SourceIndex(0) {
    if (not isa<StoreInst>(I)) {
      for (const Use &OpUse : I->operands()) {
        Sources.push_back(&OpUse);
      }
    } else {
      const auto PtrOpNum = StoreInst::getPointerOperandIndex();
      Sources.push_back(&I->getOperandUse(PtrOpNum));
    }
    revng_assert(not Sources.empty());
  }

  explicit WorkItem(Argument *A,
                    const ConstFunctionPtrSet &ReachableFunctions) :
    CurrentValue(A),
    Sources(),
    SourceIndex(0) {
    const Function *F = A->getParent();
    revng_assert(not F->empty());
    CSVAccessLog << "Function: " << F << DoLog;
    const unsigned ArgNo = A->getArgNo();
    CSVAccessLog << "ArgNo: " << ArgNo << DoLog;
    for (const Use &FUse : F->uses()) {
      const User *FUser = FUse.getUser();
      CSVAccessLog << "FUser: " << FUser << DoLog;
      if (const auto *FCall = dyn_cast<const CallInst>(FUser)) {
        CSVAccessLog << "Is a Call" << DoLog;
        const Function *Caller = FCall->getParent()->getParent();
        CSVAccessLog << "Caller: " << Caller << DoLog;
        if (ReachableFunctions.find(Caller) != ReachableFunctions.end()) {
          CSVAccessLog << "Is Reachable" << DoLog;
          CSVAccessLog << "CallInst:" << FCall << DoLog;
          const Use &actualArgUse = FCall->getArgOperandUse(ArgNo);
          CSVAccessLog << "ActualUse:" << actualArgUse.getUser() << DoLog;
          Sources.push_back(&actualArgUse);
        } else {
          CSVAccessLog << "NOT Reachable" << DoLog;
        }
      } else if (const auto *CExpr = dyn_cast<const ConstantExpr>(FUser)) {
        CSVAccessLog << "BitCast" << DoLog;
        const auto OpCode = CExpr->getOpcode();
        revng_assert(OpCode == Instruction::BitCast);
        for (const User *RealCall : CExpr->users()) {
          CSVAccessLog << "RealCall:" << RealCall << DoLog;
          const auto *FCall = dyn_cast<const CallInst>(RealCall);
          CSVAccessLog << "CallInst:" << FCall << DoLog;
          if (FCall) {
            const Function *Caller = FCall->getParent()->getParent();
            CSVAccessLog << "Caller: " << Caller << DoLog;
            if (ReachableFunctions.find(Caller) != ReachableFunctions.end()) {
              const Use &actualArgUse = FCall->getArgOperandUse(ArgNo);
              CSVAccessLog << "ActualUse:" << actualArgUse.getUser() << DoLog;
              Sources.push_back(&actualArgUse);
            }
          }
        }
      }
    }
    // This might be too strict, because the arguments of the root function
    // don't have any sources. However, we assume that we never reach them.
    revng_assert(not Sources.empty());
  }

  explicit WorkItem(CallInst *C, bool IsLoad) :
    CurrentValue(C),
    Sources(),
    SourceIndex(0) {
    const Function *F = getCallee(C);
    revng_assert(F != nullptr); // Assume no indirect calls
    if (F->getIntrinsicID() == Intrinsic::memcpy) {
      const Use &AddrOp = C->getOperandUse(IsLoad ? 1 : 0);
      Sources.push_back(&AddrOp);
      const Use &SizeOp = C->getOperandUse(2);
      Sources.push_back(&SizeOp);
    } else {
      for (const BasicBlock &BB : *F) {
        const TerminatorInst *I = BB.getTerminator();
        if (I and isa<ReturnInst>(I) and I->getNumOperands() != 0) {
          revng_assert(I->getNumOperands() == 1);
          const Use &RetValUse = I->getOperandUse(0);
          Sources.push_back(&RetValUse);
        }
      }
    }
    revng_assert(not Sources.empty());
  }

public:
  friend inline void writeToLog(Logger<true> &L, const WorkItem &I, int) {
    L << "Value: " << I.Val() << DoLog;
    L << "Sources = {" << DoLog;
    L.indent();
    for (const Use *U : I.sources())
      L << U->get() << DoLog;
    L << "}" << DoLog;
    L << "Curr Src Id: " << I.SourceIndex;
    L.unindent();
  }

public:
  Value *Val() const { return CurrentValue; };

  const Use *currentSourceUse() const {
    if (SourceIndex < Sources.size())
      return Sources[SourceIndex];
    return nullptr;
  }

  Value *currentSourceValue() const {
    const Use *CurrUse = currentSourceUse();
    return CurrUse ? CurrUse->get() : nullptr;
  }

  const Use *nextSourceUse() const {
    const auto size = Sources.size();
    if (SourceIndex < size and (SourceIndex + 1) < size)
      return Sources[SourceIndex + 1];
    return nullptr;
  }

  const Value *nextSourceValue() const {
    const Use *CurrUse = nextSourceUse();
    return CurrUse ? CurrUse->get() : nullptr;
  }

  void advanceToNextSource() {
    const auto size = Sources.size();
    if (SourceIndex < size)
      SourceIndex++;
  }

  llvm::iterator_range<SmallVector<const Use *, 3>::const_iterator>
  sources() const {
    return { Sources.begin(), Sources.end() };
  }

  SmallVector<const Use *, 3>::size_type getNumSources() const {
    return Sources.size();
  }
};

/// \brief Gets a valid pointer to `CallInst` if the current source of `Item` is
///         a call from `Root`
///
/// This function returns `nullptr` if the current source of `Item` is not a
/// call from `Root`
static CallInst *
getCurSourceRootCall(const WorkItem &Item, const Function *Root) {
  CallInst *RootCall = nullptr;
  if (isa<Argument>(Item.Val())) {
    User *ActualArgUser = Item.currentSourceUse()->getUser();
    auto *Call = cast<CallInst>(ActualArgUser);
    if (Call->getParent()->getParent() == Root)
      RootCall = Call;
  }
  return RootCall;
}

/// \brief Gets a valid pointer to `CallInst` if the next source of `Item` is
///         a call from `Root`
///
/// This function return `nullptr` if the next source of `Item` is not a call
/// from `Root`
static CallInst *
getNextSourceRootCall(const WorkItem &Item, const Function *Root) {
  CallInst *RootCall = nullptr;
  if (isa<Argument>(Item.Val())) {
    const Use *NextSrcUse = Item.nextSourceUse();
    if (NextSrcUse != nullptr) {
      User *ActualArgUser = NextSrcUse->getUser();
      auto *Call = cast<CallInst>(ActualArgUser);
      if (Call->getParent()->getParent() == Root)
        RootCall = Call;
    }
  }
  return RootCall;
}

/// \brief Gets the `Shift`-th bit of `Input`
static int GetBit(uint64_t Input, int Shift) {
  return (Input >> Shift) & 1;
};

using CallSiteOffsetMap = std::map<CallInst *, CSVOffsets>;
using ValueCallSiteOffsetMap = std::map<Value *, CallSiteOffsetMap>;
using OptCSVOffsets = llvm::Optional<CSVOffsets>;

/// \brief This class is used to fold constant offsets on different instructions
template<class T>
class CRTPOffsetFolder {

protected:
  using offset_iterator = std::set<int64_t>::const_iterator;
  using offset_iterator_range = llvm::iterator_range<offset_iterator>;
  using OffsetPair = std::pair<const CSVOffsets *, const CSVOffsets *>;

protected:
  // These should be constant but ConstantInt::get() does not have
  // const-qualifier on the first argument:
  // static ConstantInt *get(IntegerType *Ty, uint64_t V, bool isSigned=false)
  // However, it does not really change Ty, because it only call const
  // methods on it, so it should be safe.
  IntegerType *Int64Ty;
  IntegerType *Int32Ty;
  const DataLayout &DL;

public:
  CRTPOffsetFolder(const Module &M) :
    Int64Ty(IntegerType::get(M.getContext(), 64)),
    Int32Ty(IntegerType::get(M.getContext(), 32)),
    DL(M.getDataLayout()) {}
  static void insertOrCombine(Value *V,
                              CallInst *C,
                              CSVOffsets &&O,
                              ValueCallSiteOffsetMap &OffsetMap) {
    CSVAccessLog << "MAP: " << V << DoLog;
    bool Inserted;
    CallSiteOffsetMap::iterator It;
    std::tie(It, Inserted) = OffsetMap[V].insert(std::make_pair(C, O));
    if (not Inserted)
      It->second.combine(O);
  }

public:
  /// \brief This method folds the offsets on the sources ot Item
  ///
  /// \param Item is the `WorkItem` whose sources must be folded
  /// \param [in, out] is the Map used to retrieve the values of the offsets of
  ///                  the sources of `Item`, and also to store the result of
  ///                  the folded offsets for `Item`.
  void fold(const WorkItem &Item, ValueCallSiteOffsetMap &OffsetMap) {
    WorkItem::size_type NumSrcs = Item.getNumSources();
    revng_assert(NumSrcs);
    SmallVector<Constant *, 4> Operands(NumSrcs, nullptr);
    // Collect Call Sites across all the sources of this User
    CallPtrSet CallSites;
    SmallVector<const CallSiteOffsetMap *, 4> SrcCallSiteOffsetsPtrs;
    SmallVector<const CSVOffsets *, 4> NonRootOffsetsPtrs;
    SrcCallSiteOffsetsPtrs.reserve(NumSrcs);
    NonRootOffsetsPtrs.reserve(NumSrcs);
    for (const Use *U : Item.sources()) {
      const CallSiteOffsetMap &CallSiteOffsets = OffsetMap.at(U->get());
      revng_assert(not CallSiteOffsets.empty());
      const CSVOffsets *NonRootOffsets = nullptr;
      for (const auto &CSO : CallSiteOffsets) {
        CallInst *TheCall = CSO.first;
        CallSites.insert(TheCall);
        if (TheCall == nullptr) {
          NonRootOffsets = &CSO.second;
        }
      }

      // The order of iteration on SrcCallSiteOffsetsPtrs is the same
      // as the order of iteration on Item.sources()
      SrcCallSiteOffsetsPtrs.push_back(&CallSiteOffsets);

      // The order of iteration on NonRootOffsets is the same as the order
      // of iteration on Item.sources()
      NonRootOffsetsPtrs.push_back(NonRootOffsets);
    }
    revng_assert(NumSrcs == SrcCallSiteOffsetsPtrs.size());
    revng_assert(NumSrcs == NonRootOffsetsPtrs.size());

    if (CSVAccessLog.isEnabled())
      for (const CallInst *C : CallSites)
        CSVAccessLog << "C: " << C << DoLog;

    // Check that each source has all the callsites or nullptr
    for (const auto &CSOffsets : SrcCallSiteOffsetsPtrs) {
      bool found_nullptr = false;
      if (CSOffsets->find(nullptr) != CSOffsets->end()) {
        found_nullptr = true;
      }

      bool found_all_calls = true;
      for (CallInst *C : CallSites) {
        if (C != nullptr and CSOffsets->find(C) == CSOffsets->end()) {
          found_all_calls = false;
          break;
        }
      }
      revng_assert(found_nullptr or found_all_calls);
    }

    for (CallInst *C : CallSites) {
      SmallVector<OffsetPair, 4> SrcOffsets(NumSrcs,
                                            OffsetPair(nullptr, nullptr));
      bool empty_pair = false;
      for (WorkItem::size_type SI = 0; SI < NumSrcs; ++SI) {
        const CSVOffsets *NonRootOffset = NonRootOffsetsPtrs[SI];
        if (C != nullptr)
          SrcOffsets[SI].second = NonRootOffset;
        const auto CallSiteOffsets = SrcCallSiteOffsetsPtrs[SI];
        auto OffsetsEnd = CallSiteOffsets->end();
        auto OffsetsIt = CallSiteOffsets->find(C);
        if (OffsetsIt != OffsetsEnd)
          SrcOffsets[SI].first = &OffsetsIt->second;
        if (SrcOffsets[SI].first == nullptr
            and SrcOffsets[SI].second == nullptr) {
          // This means that one the pairs is empty and we can drop entirely
          // this call. This happens when C == nullptr and one of the sources
          // has no nullptr Callsite
          empty_pair = true;
        }
      }
      if (empty_pair)
        continue;
      CSVAccessLog << "start" << DoLog;
      revng_assert(NumSrcs < (8ULL * sizeof(uint64_t)));
      uint64_t combinations = 1ULL << NumSrcs;
      Value *V = Item.Val();
      Instruction *I = cast<Instruction>(V);
      for (uint64_t i = 0; i < combinations; ++i) {
        CSVAccessLog << "i:" << i << DoLog;
        SmallVector<const CSVOffsets *, 4> OffsetTuple;
        OffsetTuple.reserve(NumSrcs);

        // Build the tuple of offset sets that we want to use to compute the
        // transfer function. If the analyzed call is nullptr we can skip
        // some stuff and keep the computation smaller.
        CSVAccessLog << "callsite: " << C << DoLog;
        if (C != nullptr) {
          for (WorkItem::size_type SI = 0; SI < NumSrcs; ++SI) {
            int bit = GetBit(i, SI);
            const CSVOffsets *O0 = SrcOffsets[SI].first;
            const CSVOffsets *O1 = SrcOffsets[SI].second;
            const CSVOffsets *O = bit ? O1 : O0;
            if (O == nullptr)
              break;
            CSVAccessLog << "nonnull" << DoLog;
            OffsetTuple.push_back(O);
          }
        } else {
          for (WorkItem::size_type SI = 0; SI < NumSrcs; ++SI) {
            const CSVOffsets *O = NonRootOffsetsPtrs[SI];
            if (O == nullptr)
              break;
            CSVAccessLog << "nonnull" << DoLog;
            OffsetTuple.push_back(O);
          }
        }
        CSVAccessLog << "NumSrcs:" << NumSrcs << DoLog;
        CSVAccessLog << "Tuple Size:" << OffsetTuple.size() << DoLog;
        if (OffsetTuple.size() != NumSrcs)
          continue;

        bool Valid;
        CSVOffsets::Kind ResKind;
        SmallVector<Optional<CSVOffsets>, 4> UpdatedOffsetTuple(NumSrcs);
        revng_assert(not UpdatedOffsetTuple[0].hasValue());
        std::tie(Valid,
                 ResKind) = T::checkOffsetTupleIsValid(OffsetTuple,
                                                       I,
                                                       UpdatedOffsetTuple);
        revng_assert(UpdatedOffsetTuple.size() == OffsetTuple.size());
        if (not Valid) {
          insertOrCombine(V, C, CSVOffsets(ResKind), OffsetMap);
          continue;
        }
        CSVAccessLog << "valid tuple" << DoLog;

        SmallVector<offset_iterator_range, 4> OffsetsRanges;
        SmallVector<offset_iterator, 4> OffsetsIt;
        OffsetsRanges.reserve(NumSrcs);
        OffsetsIt.reserve(NumSrcs);

        WorkItem::size_type CartesianSize = 1;
        for (WorkItem::size_type SI = 0; SI < NumSrcs; ++SI) {
          OptCSVOffsets &UpdatedOffsets = UpdatedOffsetTuple[SI];
          bool WasUpdated = UpdatedOffsets.hasValue();
          const CSVOffsets *Tuple = WasUpdated ? UpdatedOffsets.getPointer() :
                                                 OffsetTuple[SI];
          OffsetsRanges.push_back(make_range(Tuple->begin(), Tuple->end()));
          OffsetsIt.push_back(Tuple->begin());
          const WorkItem::size_type OffsetSize = Tuple->size();
          revng_assert(OffsetSize);
          revng_assert(CartesianSize <= CartesianSize * OffsetSize);
          CartesianSize *= OffsetSize;
        }

        do {
          CSVOffsets ResOffset = foldOffsets(ResKind, NumSrcs, I, OffsetsIt);
          insertOrCombine(V, C, std::move(ResOffset), OffsetMap);
          // Advance the iterators
          {
            WorkItem::size_type SI = 0;
            bool wrapped = false;
            do {
              CSVAccessLog << "SI :" << SI << DoLog;
              if (std::next(OffsetsIt[SI]) == OffsetsRanges[SI].end()) {
                OffsetsIt[SI] = OffsetsRanges[SI].begin();
                wrapped = true;
                CSVAccessLog << "WRAP" << DoLog;
              } else {
                CSVAccessLog << "NO-WRAP" << DoLog;
                std::advance(OffsetsIt[SI], 1);
                wrapped = false;
              }
            } while (wrapped and ++SI < NumSrcs);
            CSVAccessLog << "incremented" << DoLog;
          }
        } while (--CartesianSize);
      }
    }
  }

private:
  CSVOffsets foldOffsets(CSVOffsets::Kind ResKind,
                         WorkItem::size_type NumSrcs,
                         Instruction *I,
                         const SmallVector<offset_iterator, 4> &OffsetsIt) {
    return static_cast<T *>(this)->foldOffsets(ResKind, NumSrcs, I, OffsetsIt);
  }
};

/// \brief Specialization of CRTPOffsetFolder for sums and subtractions
class AddSubOffsetFolder : public CRTPOffsetFolder<AddSubOffsetFolder> {

public:
  AddSubOffsetFolder(const Module &M) :
    CRTPOffsetFolder<AddSubOffsetFolder>(M) {}

public:
  friend class CRTPOffsetFolder<AddSubOffsetFolder>;

private:
  static std::pair<bool, CSVOffsets::Kind>
  checkOffsetTupleIsValid(const SmallVector<const CSVOffsets *, 4> &OffsetTuple,
                          const Instruction *I,
                          SmallVector<OptCSVOffsets, 4> &) {
    revng_assert(OffsetTuple.size() == 2);
    auto OpCode = I->getOpcode();
    revng_assert(OpCode == Instruction::Add or OpCode == Instruction::Sub);
    const auto O0 = OffsetTuple[0], O1 = OffsetTuple[1];
    if (OpCode == Instruction::Add) {
      // Cannot add pointers
      revng_assert(not(O0->isPtr() and O1->isPtr()));
    } else {
      // Cannot subtract a pointer from something else
      revng_assert(not O1->isPtr());
    }

    if (O0->isUnknown() or O1->isUnknown()) {
      if (O0->isOnlyInPtr() or O1->isOnlyInPtr())
        return { false, CSVOffsets::Kind::UnknownInPtr };
      if (O0->isInOutPtr() or O1->isInOutPtr())
        return { false, CSVOffsets::Kind::OutAndUnknownInPtr };
      return { false, CSVOffsets::Kind::Unknown };
    }

    {
      bool Num0 = O0->isNumeric();
      if (Num0 or O1->isNumeric()) {
        CSVOffsets::Kind ResKind = Num0 ? O1->getKind() : O0->getKind();
        if (O0->isUnknownInPtr() or O1->isUnknownInPtr())
          return { false, ResKind };
        else
          return { true, ResKind };
      }
    }
    revng_abort();
  }

  CSVOffsets foldOffsets(CSVOffsets::Kind ResultKind,
                         WorkItem::size_type NumSrcs,
                         Instruction *I,
                         const SmallVector<offset_iterator, 4> &OffsetsIt) {
    auto OpCode = I->getOpcode();
    revng_assert(OpCode == Instruction::Add or OpCode == Instruction::Sub);
    SmallVector<Constant *, 4> Operands(NumSrcs, nullptr);
    // Setup operands
    for (WorkItem::size_type SI = 0; SI < NumSrcs; ++SI) {
      const int64_t O = *OffsetsIt[SI];
      Operands[SI] = ConstantInt::get(Int64Ty, APInt(64, O, true));
    }
    // Constant fold the operation with the selected operands
    ArrayRef<Constant *> TmpOp(Operands);
    Constant *Res = ConstantFoldInstOperands(I, TmpOp, DL);
    ConstantInt *R = cast<ConstantInt>(Res);
    const int64_t ResO = R->getSExtValue();
    return CSVOffsets(ResultKind, ResO);
  }
};

/// \brief Specialization of CRTPOffsetFolder for GEPs
class GEPOffsetFolder : public CRTPOffsetFolder<GEPOffsetFolder> {

public:
  GEPOffsetFolder(const Module &M) : CRTPOffsetFolder<GEPOffsetFolder>(M) {}

public:
  friend class CRTPOffsetFolder<GEPOffsetFolder>;

private:
  static std::pair<bool, CSVOffsets::Kind>
  checkOffsetTupleIsValid(const SmallVector<const CSVOffsets *, 4> &OffsetTuple,
                          const Instruction *I,
                          SmallVector<OptCSVOffsets, 4> &UpdatedOffsetTuple) {
    auto OpCode = I->getOpcode();
    revng_assert(OpCode == Instruction::GetElementPtr);
    size_t NOperands = OffsetTuple.size();
    revng_assert(NOperands > 1);
    CSVOffsets::Kind GEPOp0Kind = OffsetTuple[0]->getKind();
    if (CSVOffsets::isUnknownInPtr(GEPOp0Kind)
        or CSVOffsets::isUnknown(GEPOp0Kind))
      return { false, GEPOp0Kind };

    auto *GEP = cast<GetElementPtrInst>(I);
    auto *PtrTy = cast<PointerType>(GEP->getPointerOperandType());
    Type *PointeeTy = PtrTy->getElementType();

    if (GEP->hasIndices() and PointeeTy->isArrayTy()) {

      auto FirstIdxConst = dyn_cast<ConstantInt>(*GEP->idx_begin());
      bool FirstIdxPropagatedConst = OffsetTuple[1]->size() == 1;
      revng_assert(not(FirstIdxConst != nullptr) or FirstIdxPropagatedConst);

      if (FirstIdxPropagatedConst) {

        bool FirstIdxZero = FirstIdxConst->isZero();
        bool FirstIdxPropagatedZero = *OffsetTuple[1]->begin() == 0;
        revng_assert(not FirstIdxZero or FirstIdxPropagatedZero);

        if (FirstIdxPropagatedZero) {
          SmallVector<uint64_t, 4> ConstIdxList = { 0 };

          auto IdxIt = GEP->idx_begin();
          auto IdxEnd = GEP->idx_end();
          int IdxOpNum = 1;
          std::set<int64_t> LastTypeOffsets = { 0 };

          for (; IdxIt != IdxEnd; ++IdxIt, ++IdxOpNum) {
            const CSVOffsets *IdxCSVOffset = OffsetTuple[IdxOpNum];
            revng_assert(not IdxCSVOffset->isPtr());

            Type *ElementTy = GEP->getIndexedType(PointeeTy, ConstIdxList);

            if (ElementTy->isAggregateType()) {
              if (ElementTy->isArrayTy()) {
                if (IdxCSVOffset->isUnknown()) {
                  CSVOffsets U(CSVOffsets::Kind::Numeric, LastTypeOffsets);
                  UpdatedOffsetTuple[IdxOpNum] = std::move(U);
                } else {
                  // If it's not Unknown we can leave it like it is.
                }

                auto *ArrayTy = cast<ArrayType>(ElementTy);
                uint64_t ArrayNumElem = ArrayTy->getNumElements();
                revng_assert(ArrayNumElem);
                LastTypeOffsets.clear();
                for (uint64_t O = 0; O < ArrayNumElem; ++O)
                  LastTypeOffsets.insert(O);

                ConstIdxList.push_back(0);
              } else if (ElementTy->isStructTy()) {
                if (IdxCSVOffset->isUnknown()) {
                  if (IdxIt + 1 != IdxEnd) {
                    // I cannot fold structs with unknown index.
                    // Early exit.
                    return { false, CSVOffsets::makeUnknown(GEPOp0Kind) };
                  } else {
                    revng_assert(not LastTypeOffsets.empty());
                    CSVOffsets U(CSVOffsets::Kind::Numeric, LastTypeOffsets);
                    UpdatedOffsetTuple[IdxOpNum] = std::move(U);
                    break;
                  }
                } else {
                  // If it's not Unknown we can leave it like it is.
                  revng_assert(not IdxCSVOffset->empty());
                  ConstIdxList.push_back(*IdxCSVOffset->begin());

                  revng_assert(IdxCSVOffset->size() != 0);
                  LastTypeOffsets.clear();
                  LastTypeOffsets.insert(IdxCSVOffset->begin(),
                                         IdxCSVOffset->end());
                }
              } else {
                revng_abort();
              }
            } else {
              // I'm done.
              revng_assert(IdxIt + 1 == IdxEnd);
              if (IdxCSVOffset->isUnknown()) {
                CSVOffsets U(CSVOffsets::Kind::Numeric, LastTypeOffsets);
                UpdatedOffsetTuple[IdxOpNum] = std::move(U);
              } else {
                // If it's not Unknown we can leave it like it is.
              }
            }
          }
          return { true, GEPOp0Kind };
        } else {
          // For now we don't handle cases when the first index of the GEP is
          // not zero, so in those case we fall back outside the if and we fold
          // them as ususal.
        }
      } else {
        // For now we don't handle cases when the first index of the GEP is
        // not constant, so in those case we fall back outside the if and we
        // fold them as ususal.
      }
    } else {
      // For now we don't handle cases when the GEP does not index an array, so
      // in those case we fall back outside the if and we fold them as ususal.
    }

    for (size_t O = 1; O < NOperands; O++) {
      revng_assert(not OffsetTuple[O]->isPtr());
      if (OffsetTuple[O]->isUnknown())
        return { false, CSVOffsets::makeUnknown(GEPOp0Kind) };
    }
    return { true, GEPOp0Kind };
  }

  CSVOffsets foldOffsets(CSVOffsets::Kind ResultKind,
                         WorkItem::size_type NumSrcs,
                         Instruction *I,
                         const SmallVector<offset_iterator, 4> &OffsetsIt) {
    const auto *GEP = cast<const GetElementPtrInst>(I);
    const auto PtrOpTy = GEP->getPointerOperand()->getType();
    SmallVector<Constant *, 4> Operands(NumSrcs, nullptr);
    // Setup operands
    int64_t PtrOp = *OffsetsIt[0];
    Operands[0] = Constant::getIntegerValue(PtrOpTy, APInt(64, PtrOp, true));
    for (WorkItem::size_type SI = 1; SI < NumSrcs; ++SI) {
      const int64_t O = *OffsetsIt[SI];
      Operands[SI] = ConstantInt::get(Int32Ty, APInt(32, O, true));
    }
    // Constant fold the operation with the selected operands
    ArrayRef<Constant *> TmpOp(Operands);
    Constant *Res = ConstantFoldInstOperands(I, TmpOp, DL);
    ConstantInt *R = getConstValue(Res, DL);
    const int64_t ResO = getSExtValue(R, DL);
    return CSVOffsets(ResultKind, ResO);
  }
};

/// \brief Specialization of CRTPOffsetFolder for non-address binary operations
class NumericOffsetFolder : public CRTPOffsetFolder<NumericOffsetFolder> {

public:
  NumericOffsetFolder(const Module &M) :
    CRTPOffsetFolder<NumericOffsetFolder>(M) {}

public:
  friend class CRTPOffsetFolder<NumericOffsetFolder>;

private:
  static std::pair<bool, CSVOffsets::Kind>
  checkOffsetTupleIsValid(const SmallVector<const CSVOffsets *, 4> &OffsetTuple,
                          const Instruction *I,
                          SmallVector<OptCSVOffsets, 4> &) {
    revng_assert(OffsetTuple.size() == 2);

    auto OpCode = I->getOpcode();
    revng_assert(OpCode == Instruction::Shl or OpCode == Instruction::AShr
                 or OpCode == Instruction::LShr or OpCode == Instruction::Mul
                 or OpCode == Instruction::URem or OpCode == Instruction::SRem
                 or OpCode == Instruction::SDiv or OpCode == Instruction::UDiv);

    const auto O0 = OffsetTuple[0], O1 = OffsetTuple[1];
    revng_assert(not O0->isPtr() and not O1->isPtr());
    if (O0->isUnknown() or O1->isUnknown()) {
      return std::make_pair(false, CSVOffsets::Kind::Unknown);
    } else {
      return std::make_pair(true, CSVOffsets::Kind::Numeric);
    }
  }

  CSVOffsets foldOffsets(CSVOffsets::Kind ResultKind,
                         WorkItem::size_type NumSrcs,
                         Instruction *I,
                         const SmallVector<offset_iterator, 4> &OffsetsIt) {

    auto OpCode = I->getOpcode();
    revng_assert(OpCode == Instruction::Shl or OpCode == Instruction::AShr
                 or OpCode == Instruction::LShr or OpCode == Instruction::Mul
                 or OpCode == Instruction::URem or OpCode == Instruction::SRem
                 or OpCode == Instruction::SDiv or OpCode == Instruction::UDiv);

    SmallVector<Constant *, 4> Operands(NumSrcs, nullptr);
    // Setup operands
    for (WorkItem::size_type SI = 0; SI < NumSrcs; ++SI) {
      const int64_t O = *OffsetsIt[SI];
      Operands[SI] = ConstantInt::get(Int64Ty, APInt(64, O, true));
    }
    // Constant fold the operation with the selected operands
    ArrayRef<Constant *> TmpOp(Operands);
    Constant *Res = ConstantFoldInstOperands(I, TmpOp, DL);
    ConstantInt *R = cast<ConstantInt>(Res);
    const int64_t ResO = R->getSExtValue();
    return CSVOffsets(ResultKind, ResO);
  }
};

using AccessOffsetMap = std::map<llvm::Instruction *, CSVOffsets>;

class CPUStateAccessOffsetAnalysis {

private:
  const Module &M;
  const Value *CPUStatePtr;
  const Function *RootFunction;
  const ConstFunctionPtrSet &ReachableFunctions;
  const TaintResults &TaintedAccesses;
  VariableManager *Variables;

  AccessOffsetMap &LoadOffsets; // result, maps load or load-memcpy to offsets
  AccessOffsetMap &StoreOffsets; // result, maps store or store-memcpy to
                                 // offsets
  CallSiteOffsetMap &CallSiteLoadOffsets; // result, maps call in root to load
                                          // offsets
  CallSiteOffsetMap &CallSiteStoreOffsets; // result, maps call in root to store
                                           // offsets

  // ValueCallSiteOffsets is used to keep track of the offsets associated with
  // each value. The primary key is the `Value` for which we're tracking the
  // offsets. The secondary key is a `CallInst` representing a call site in
  // root. This call site represents the call from which the mapped offsets are
  // possible. The mapped value is a `CSVOffsets`.
  ValueCallSiteOffsetMap ValueCallSiteOffsets;

  // The two following maps have the same structure as ValueCallSiteOffsets, but
  // they are use to hold results on the specific loads and stores that access
  // CSV. They are used after the computation to generate the metadata to attach
  // to the root call sites and the final results of the analysis.
  ValueCallSiteOffsetMap LoadCallSiteOffsets;
  ValueCallSiteOffsetMap StoreCallSiteOffsets;

  CallPtrSet CrossedCallSites;
  std::vector<WorkItem> WorkList;
  ConstValuePtrSet InExploration;
  AddSubOffsetFolder AddSubFolder;
  NumericOffsetFolder NumericFolder;
  GEPOffsetFolder GEPFolder;

public:
  CPUStateAccessOffsetAnalysis(const Module &Mod,
                               const Value *EnvPtr,
                               const Function *Root,
                               const ConstFunctionPtrSet &Reachable,
                               const TaintResults &Tainted,
                               VariableManager *Vars,
                               AccessOffsetMap &LoadOff,
                               AccessOffsetMap &StoreOff,
                               CallSiteOffsetMap &CallSiteLoadOff,
                               CallSiteOffsetMap &CallSiteStoreOff) :
    M(Mod),
    CPUStatePtr(EnvPtr),
    RootFunction(Root),
    ReachableFunctions(Reachable),
    TaintedAccesses(Tainted),
    Variables(Vars),
    LoadOffsets(LoadOff),
    StoreOffsets(StoreOff),
    CallSiteLoadOffsets(CallSiteLoadOff),
    CallSiteStoreOffsets(CallSiteLoadOff),
    ValueCallSiteOffsets(),
    LoadCallSiteOffsets(),
    StoreCallSiteOffsets(),
    CrossedCallSites(),
    WorkList(),
    InExploration(),
    AddSubFolder(M),
    NumericFolder(M),
    GEPFolder(M) {}

public:
  bool run();

private:
  void cleanup() {
    ValueCallSiteOffsets = {};
    LoadCallSiteOffsets = {};
    StoreCallSiteOffsets = {};
    CrossedCallSites = {};
    WorkList = {};
    InExploration = {};
  }

  /// \brief Analyzes the access to env performed by \p I, saving results
  ///        according to \p IsLoad
  /// \param I is the `Instruction` whose accesses are analyzed
  /// \param IsLoad must be true if called when analyzing loads, false if
  /// called when analyzing stores. This is important because it is used to
  /// update the correct ValueCallSiteOffsetMap (either LoadCallSiteOffsets or
  /// StoreCallSiteOffsets) if during the exploration the analysis ends because
  /// all the immediate sources are already resolved.
  void analyzeAccess(Instruction *I, bool IsLoad);

  /// \brief Explores the sources of `V` and pushes a `WorkItem` on `WorkList`
  ///        if something new is found
  /// \param V is the `Value` whose sources are analyzed
  /// \param IsLoad must be true if called when propagating loads, false if
  /// called when propagating stores. This is important because it is used to
  /// update the correct ValueCallSiteOffsetMap (either LoadCallSiteOffsets or
  /// StoreCallSiteOffsets) if during the exploration the analysis ends because
  /// all the immediate sources are already resolved.
  void exploreImmediateSources(Value *V, bool IsLoad);

  /// \brief Returns an emtpy Optional and fill W if there are unexplored
  ///        sources, otherwise return the offsets
  ///
  /// \param V the `Value` whose sources must be explored.
  /// \param [out] W a `WorkItem` that will be initialized with the unexplored
  ///                sources of `V` if any.
  /// \param IsLoad true if we're exploring from a load
  OptCSVOffsets
  getOffsetsOrExploreSrc(Value *V, WorkItem &W, bool IsLoad) const;

  void insertCallSiteOffset(Value *V, CSVOffsets &&Offset);

  /// \brief Removes the root call site associated with `Item` (if any) from
  ///        `CrossedCallSites`
  /// \return `true` if it was removed, `false` otherwise
  bool removeCurCrossedCallSite(const WorkItem &Item) {
    CallInst *RootCallSite = getCurSourceRootCall(Item, RootFunction);
    if (RootCallSite) {
      auto NumErased = CrossedCallSites.erase(RootCallSite);
      revng_assert(NumErased);
      return true;
    }
    return false;
  }

  /// \brief Returns true if `V` is visited for the first time with the callsite
  ///        `NewCallSite`
  /// \param V is the `Value` that is being visited
  /// \param NewCallSite is the new call site from which we're exploring V and
  ///        we want to check if it's the first time we visit V with that
  ///        particular call site
  ///
  /// This function returns `true` if this is the first visit, `false` otherwise
  bool isNewVisitWithCallSite(Value *V, CallInst *NewCallSite) const {

    // Handle constants in a special way. Constants are kind of global values
    // that can be used across different functions without properly propagating
    // on the call graph across call sites.
    // This has two consequences.
    // 1) On the one hand we don't want to track their exact call site, because
    // they can propagate independently of the call sites. Hence all constants
    // are collected with an 'artificial' nullptr callsite (see the else).
    // 2) On the other hand, when we cross root call sites and we find that one
    // of the arguments is a constant, we want to mark it every time as a new
    // visit. If we don't mark it as a new visit, the root call site will not be
    // crossed, hence it will not be inserted in the `CrossedCallSites`, which
    // is not what we want because it would lead to errors in computing the root
    // call sites that are active for a given exploration. For this reason, if V
    // is a constant and (NewCallSite != nullptr) we always say it's a new
    // visit, because it means we're crossing a root call site towards a
    // constant argument.
    if (isa<ConstantInt>(V)) {
      if (NewCallSite != nullptr)
        return true;
      else
        NewCallSite = nullptr;
    }

    const auto CallSiteOffsetIt = ValueCallSiteOffsets.find(V);
    // If the ValueCallSiteOffsets does not contain V it's a new visit
    if (CallSiteOffsetIt == ValueCallSiteOffsets.end())
      return true;

    // If the ValueCallSiteOffsets contains V we have already analyzed visited
    // this value, but we don't know which call sites were contained in
    // CrossedCallSites during the last visit.

    // If NewCallSite is not nullptr, we are crossing a new callsite in the root
    // function, so we start looking in the ValueCallSiteOffsets for an entry
    // associated to NewCallSite
    if (NewCallSite) {
      // If we find that ValueCallSiteOffsets still does not contain an entry
      // associated to NewCallSite this visit is considered new
      const auto OffsetIt = CallSiteOffsetIt->second.find(NewCallSite);
      if (OffsetIt == CallSiteOffsetIt->second.end())
        return true;
    }

    // If CrossedCallSites is not empty, we have crossed at least one call site
    // in the root function, so we need to look in CrossedCallSites if there is
    // a new call site to analyze
    if (CrossedCallSites.size()) {
      for (CallInst *Call : CrossedCallSites) {
        // If we find a call site in CrossedCallSites for which the
        // ValueCallSiteOffsets
        // still does not contain a result for this value this visit is
        // considered new
        const auto OffsetIt = CallSiteOffsetIt->second.find(Call);
        if (OffsetIt == CallSiteOffsetIt->second.end())
          return true;
      }
    } else {
      // If CrossedCallSites is empty, we haven't crossed any call site in the
      // root function, so we look for nullptr.
      const auto OffsetIt = CallSiteOffsetIt->second.find(nullptr);
      if (OffsetIt == CallSiteOffsetIt->second.end())
        return true;
    }

    // If we reach this point the ValueCallSiteOffsets already contains an entry
    // for V, and all the current CrossedCallSites have already been computed
    // for that entry, so the visit is not new
    return false;
  }

  /// \brief If it's a new visit, insert the call `RootCall` (associated with
  ///        `U`) in `CrossedCallSites`
  ///
  /// \param RootCall must be nullptr or a valid call instruction in root
  /// \param U if RootCall is not `nullptr` this is a `Use` whose `User` must be
  ///        RootCall
  /// \return `true` if it was a new visit, `false` otherwise.
  ///
  /// If this is the first visit, `RootCall` is inserted in `CrossedCallSites`
  /// and the function returns `true`.
  /// If this is not the first visit or `RootCall` it returns `false`
  bool checkNewVisitAndInsertCrossedCallSite(CallInst *RootCall, const Use *U) {
    if (isNewVisitWithCallSite(U->get(), RootCall)) {
      if (RootCall) {
        bool New = CrossedCallSites.insert(RootCall).second;
        revng_assert(New);
      }
      return true;
    }
    return false;
  }

  bool checkNewVisitAndInsertCurCrossedCallSite(const WorkItem &Item) {
    CallInst *RootCallSite = getCurSourceRootCall(Item, RootFunction);
    return checkNewVisitAndInsertCrossedCallSite(RootCallSite,
                                                 Item.currentSourceUse());
  }

  bool checkNewVisitAndInsertNextCrossedCallSite(const WorkItem &Item) {
    CallInst *RootCallSite = getNextSourceRootCall(Item, RootFunction);
    return checkNewVisitAndInsertCrossedCallSite(RootCallSite,
                                                 Item.nextSourceUse());
  }

  /// \brief Selects the next source of `Item`, if possible, returning true on
  ///        success.
  bool selectNextSource(WorkItem &Item) {
    const Use *NextSrcUse = Item.nextSourceUse();
    if (NextSrcUse != nullptr) {
      removeCurCrossedCallSite(Item);
      checkNewVisitAndInsertNextCrossedCallSite(Item);
      Item.advanceToNextSource();
      return true;
    }
    return false;
  }

  void push(WorkItem &&Item) {
    InExploration.insert(Item.Val());
    WorkList.push_back(Item);
    CSVAccessLog.indent(2);
  }

  void pop() {
    InExploration.erase(WorkList.back().Val());
    WorkList.pop_back();
    CSVAccessLog.unindent(2);
  }

  bool isInExploration(const Value *V) {
    return InExploration.find(V) != InExploration.end();
  }

  void computeOffsetsFromSources(const WorkItem &Item, bool IsLoad);

  template<bool IsLoad>
  void computeAggregatedOffsets();
};

using CPUSAOA = CPUStateAccessOffsetAnalysis;

void CPUSAOA::computeOffsetsFromSources(const WorkItem &Item, bool IsLoad) {
  Value *ItemVal = Item.Val();
  if (isa<PHINode>(ItemVal) or isa<Argument>(ItemVal)
      or isa<CallInst>(ItemVal)) {

    // These three cases represent points of convergence of information coming
    // from different origins.
    //
    // For `PHINode` the information comes from the different branches of the
    // phi. As an example the 'then' branch of an 'if' could compute an offsets,
    // whereas the 'else' branch could compute a different offset.
    //
    // For `Argument` the different sources of information are all the actual
    // arguments of all the calls to the function that are associated to the
    // formal `Argument` that we're analyzing. An example is that if the
    // function `int a(int b, int c)` if called in two places, such as `f(1,2)`
    // and `f(3,4)`, we have that 1 and 3 are the sources of the argument `b`.
    //
    // For `CallInst` we are considering the propagation of values from the
    // `return` instructions inside the callee to the call site.
    // For example if we have:
    // ```
    // int a(int b) {
    //   if (b) {
    //     int ret1 = b >> 2;
    //     return ret1;
    //   } else {
    //     int ret2 = b << 2;
    //     return ret2;
    //   }
    // }
    // ```
    // and `a` is called such as:
    // ```
    // y = f(x);
    // ...
    // ```
    // the sources of the `CallInst` are `ret1` and `ret2`.

    CSVAccessLog << "POP JOIN" << DoLog;

    SmallVector<const CallSiteOffsetMap *, 10> SrcCallSiteOffsets;
    SrcCallSiteOffsets.reserve(Item.getNumSources());
    std::map<CallInst *, std::set<WorkItem::size_type>> CallSiteSrcIds;

    // This loop fills `SrcCallSiteOffsets` so that its n-th element will point
    // to the `CallSiteOffsetMap` associated with the n-th source of the Value
    // that we're considering.
    WorkItem::size_type SI = 0;
    for (const Use *Src : Item.sources()) {
      Value *SrcVal = Src->get();
      CSVAccessLog << "SrcVal: " << SrcVal << DoLog;
      const CallSiteOffsetMap &CallSiteOffset = ValueCallSiteOffsets.at(SrcVal);

      // The `CallSiteOffsetMap` associated with `SrcVal` is pushed back into
      // `SrcCallSiteOffsets`, into position `SI`.
      SrcCallSiteOffsets.push_back(&CallSiteOffset);

      // Then, this loop inserts the source index `SI` into the set of source
      // indices associated with the call site `C.first`
      for (const auto &C : CallSiteOffset)
        CallSiteSrcIds[C.first].insert(SI);
      ++SI;
    }

    // Here `SrcCallSiteOffsets[i]` contains a pointer the the
    // `CallSiteOffsetMap` associated with the source number `i` of the analyzed
    // `Value`.
    // Here, for a given call site `C` in `root` the value
    // `CallSiteSrcIds.at(C)` is the set of source indices for which the value
    // currently analyzed is reached from the call site `C`.

    auto Call = dyn_cast<CallInst>(ItemVal);
    const Function *Callee = Call ? getCallee(Call) : nullptr;
    if (Callee != nullptr and Callee->getIntrinsicID() == Intrinsic::memcpy) {

      // Separately handle Intrinsic::memcpy;
      // Calls to Intrinsic::memcpy are a special case.
      // Given that they don't generate any Value, they cannot be pushed on the
      // worklist as a consequence of the backward exploration towards sources.
      // For this reason, if we reach a point where we're trying to compute the
      // offsets from the sources of a call to Intrinsic::memcpy it must be the
      // memcpy where we started from and we don't really want to propagate from
      // the return to the call site, but we want to compute the result and
      // store it in the proper map.

      CSVAccessLog << "MAP Instrinsic::memcpy: " << ItemVal << DoLog;

      revng_assert(isa<ConstantInt>(Call->getArgOperand(2)));
      ValueCallSiteOffsetMap &VCSOffsets = IsLoad ? LoadCallSiteOffsets :
                                                    StoreCallSiteOffsets;
      Value *PtrOp = IsLoad ? Call->getArgOperand(1) : Call->getArgOperand(0);
      auto CSOff = std::make_pair(ItemVal, ValueCallSiteOffsets.at(PtrOp));
      bool New = VCSOffsets.insert(CSOff).second;
      revng_assert(New);

    } else {

      // This loop iterates over each call instruction in `CallSiteSrcIds`,
      // and, for all the source indices associated to that call site, it
      // combines all of the `CSVOffsets` to compute the new `CSVOffsets` of the
      // node that we're popping.
      for (const auto &CallSrc : CallSiteSrcIds) {
        Optional<CSVOffsets> New;
        CallInst *TheCall = CallSrc.first;
        for (const auto i : CallSrc.second) {
          CSVAccessLog << "AT: " << TheCall << DoLog;
          const CSVOffsets &SrcOffset = SrcCallSiteOffsets[i]->at(TheCall);
          if (New)
            New.getValue().combine(SrcOffset);
          else
            New = SrcOffset;
        }
        CSVAccessLog << "MAP JOIN: " << ItemVal << DoLog;
        CSVAccessLog << "    " << New.getValue() << DoLog;

        // Insert the `New` in the `ValueCallSiteOffsets`
        ValueCallSiteOffsets[ItemVal][TheCall] = std::move(New.getValue());
        CSVAccessLog << "CallSite: " << TheCall << DoLog;
        CSVAccessLog << "    " << ValueCallSiteOffsets.at(ItemVal).at(TheCall)
                     << DoLog;
      }
    }

    removeCurCrossedCallSite(Item);

  } else if (auto *Instr = dyn_cast<Instruction>(ItemVal)) {

    CSVAccessLog << "POP INST" << DoLog;

    const auto OpCode = Instr->getOpcode();
    switch (OpCode) {
    case Instruction::ZExt:
    case Instruction::SExt:
    case Instruction::Trunc:
    case Instruction::PtrToInt:
    case Instruction::IntToPtr:
    case Instruction::BitCast: {
      CSVAccessLog << "MAP CAST: " << ItemVal << DoLog;
      Value *Op = Instr->getOperand(0);
      ValueCallSiteOffsets[ItemVal] = ValueCallSiteOffsets.at(Op);
    } break;
    case Instruction::Sub:
    case Instruction::Add: {
      CSVAccessLog << "Add/Sub" << DoLog;
      AddSubFolder.fold(Item, ValueCallSiteOffsets);
    } break;
    case Instruction::Shl:
    case Instruction::AShr:
    case Instruction::LShr:
    case Instruction::Mul:
    case Instruction::URem:
    case Instruction::SRem:
    case Instruction::SDiv:
    case Instruction::UDiv: {
      CSVAccessLog << "NumericFold" << DoLog;
      NumericFolder.fold(Item, ValueCallSiteOffsets);
    } break;
    case Instruction::GetElementPtr: {
      CSVAccessLog << "GEP" << DoLog;
      GEPFolder.fold(Item, ValueCallSiteOffsets);
    } break;
    case Instruction::Load: {
      Value *AddressValue = cast<LoadInst>(Instr)->getPointerOperand();
      auto LoadCSOff = std::make_pair(ItemVal,
                                      ValueCallSiteOffsets.at(AddressValue));
      bool New = LoadCallSiteOffsets.insert(LoadCSOff).second;
      if (CSVAccessLog.isEnabled()) {
        CSVAccessLog << "Load " << dumpToString(Instr) << DoLog;

        for (const auto &CS2O : LoadCSOff.second) {
          CSVAccessLog << "CallSite: ";
          if (CS2O.first)
            CSVAccessLog << dumpToString(Instr);
          else
            CSVAccessLog << "nullptr";
          CSVAccessLog << DoLog;
          CSVAccessLog << CS2O.second << DoLog;
        }
      }
      revng_assert(New);
    } break;
    case Instruction::Store: {
      Value *AddressValue = cast<StoreInst>(Instr)->getPointerOperand();
      auto StoreCSOff = std::make_pair(ItemVal,
                                       ValueCallSiteOffsets.at(AddressValue));
      bool New = StoreCallSiteOffsets.insert(StoreCSOff).second;
      if (CSVAccessLog.isEnabled()) {
        CSVAccessLog << "Store " << dumpToString(Instr) << DoLog;
        for (const auto &CS2O : StoreCSOff.second) {
          CSVAccessLog << "CallSite: ";
          if (CS2O.first)
            CSVAccessLog << dumpToString(Instr);
          else
            CSVAccessLog << "nullptr";
          CSVAccessLog << DoLog;
          CSVAccessLog << CS2O.second << DoLog;
        }
      }
      revng_assert(New);
    } break;
    default:
      revng_abort(dumpToString(Instr).data());
    }
  } else {
    revng_abort();
  }
}

void CPUSAOA::insertCallSiteOffset(Value *V, CSVOffsets &&Offset) {
  CSVAccessLog << "MAP INSERT: " << V << DoLog;
  // If CrossedCallSites is empty we haven't reached the root function during
  // the backward exploration, so the only active call site is nullptr.
  // The same holds if we're inserting the offset for a ConstantInt. The problem
  // with constants is that they are not propagated through the call graph, but
  // they are in a global context and they infect all the places where they're
  // used independently of the call sites crossed during the exploration. For
  // this reason, if we accumulate all the call sites for the constants we may
  // end up in situation where on a given constant we have lots of call sites in
  // the map, but all with the same CSVOffsets. This is bad for two reasons:
  // 1) it increases the size of the map for no reason;
  // 2) it potentially propagates wrong call sites where the constants are used.
  // For this reason we used this workaround, to only insert null call sites for
  // constants.
  if (CrossedCallSites.empty() or isa<ConstantInt>(V)) {
    ValueCallSiteOffsets[V][nullptr] = Offset;
    CSVAccessLog << "CallSite: nullptr" << DoLog;
    CSVAccessLog << "    " << Offset << DoLog;
  } else {
    // In all the other cases use the active set of crossed call sites
    for (const auto &Call : CrossedCallSites) {
      ValueCallSiteOffsets[V][Call] = Offset;
      CSVAccessLog << "CallSite: " << Call << DoLog;
      CSVAccessLog << "    " << Offset << DoLog;
    }
  }
}

OptCSVOffsets
CPUSAOA::getOffsetsOrExploreSrc(Value *V, WorkItem &Item, bool IsLoad) const {
  if (auto *Call = dyn_cast<CallInst>(V)) {
    if (CSVAccessLog.isEnabled()) {
      CSVAccessLog << "CALL" << DoLog;
      CSVAccessLog << dumpToString(Call) << DoLog;
    }
    Item = WorkItem(Call, IsLoad);
  } else if (auto *Arg = dyn_cast<Argument>(V)) {
    CSVAccessLog << "ARG" << DoLog;
    Item = WorkItem(Arg, ReachableFunctions);
  } else if (auto *Instr = dyn_cast<Instruction>(V)) {
    CSVAccessLog << "INST" << DoLog;
    const auto OpCode = Instr->getOpcode();
    switch (OpCode) {
    case Instruction::Load: {
      CSVAccessLog << "LOAD" << DoLog;
      const auto *Load = cast<const LoadInst>(Instr);
      const Value *Ptr = Load->getPointerOperand();
      if (const auto *CSV = dyn_cast<const GlobalVariable>(Ptr)) {
        CSVAccessLog << "GLOBAL" << DoLog;
        if (CSV == CPUStatePtr) {
          CSVAccessLog << "ENV" << DoLog;
          return CSVOffsets(CSVOffsets::Kind::KnownInPtr, 0);
        } else {
          CSVAccessLog << "NOT-ENV" << DoLog;
          return CSVOffsets(CSVOffsets::Kind::Unknown);
        }
      } else {
        CSVAccessLog << "NOT-GLOBAL" << DoLog;
        return CSVOffsets(CSVOffsets::Kind::Unknown);
      }
    }
    case Instruction::Alloca:
      CSVAccessLog << "ALLOCA" << DoLog;
      return CSVOffsets(CSVOffsets::Kind::Unknown);
    case Instruction::Or:
    case Instruction::And:
    case Instruction::ICmp:
      CSVAccessLog << "CMP" << DoLog;
      return CSVOffsets(CSVOffsets::Kind::Unknown);
    case Instruction::Store:
      revng_abort();
    default:
      break;
    }
    // If we reach this point the CSVOffsets of this instruction are not known
    Item = WorkItem(Instr);
  } else if (const auto *IntConst = dyn_cast<const ConstantInt>(V)) {
    int64_t Offset = IntConst->getSExtValue();
    CSVAccessLog << "CONST: " << Offset << DoLog;
    return CSVOffsets(CSVOffsets::Kind::Numeric, Offset);
  } else {
    revng_abort();
  }
  return OptCSVOffsets();
}

void CPUSAOA::exploreImmediateSources(Value *V, bool IsLoad) {
  // Try to get new unexplored sources for V.
  WorkItem NewItem;
  {
    auto ConstKnownOffsets = getOffsetsOrExploreSrc(V, NewItem, IsLoad);
    if (ConstKnownOffsets.hasValue()) {
      CSVAccessLog << "ConstantOffset" << DoLog;

      // If we reach this point, V only has a constant know CSVOffsets and does
      // not really have sources that must be explored. In this case we can just
      // insert the ConstKnownOffsets in the map and we're done.
      insertCallSiteOffset(V, std::move(ConstKnownOffsets.getValue()));
      return;
    }
  }
  CSVAccessLog << "New!: " << NewItem << DoLog;
  // If we reach this point NewItem is valid and contains a vector of sources
  // for V. Iterate an all the sources, looking for the first unexplored one.
  for (const Use *U : NewItem.sources()) {
    CSVAccessLog << "Src: " << NewItem << DoLog;
    // Cut recursion
    if (isInExploration(NewItem.Val())) {
      CSVAccessLog << "RECURSION: " << U->get() << DoLog;
      const ConstValuePtrSet &Tainted = TaintedAccesses.TaintedValues;
      CSVOffsets::Kind NewKind = Tainted.find(NewItem.Val()) != Tainted.end() ?
                                   CSVOffsets::Kind::UnknownInPtr :
                                   CSVOffsets::Kind::Unknown;
      insertCallSiteOffset(U->get(), CSVOffsets(NewKind));
      continue;
    }
    // If we're visiting the sources of an argument we are crossing a
    // new call site, which might lead us into the root function.
    // If it does, we want to register it in the CrossedCallSites
    if (checkNewVisitAndInsertCurCrossedCallSite(NewItem)) {
      CSVAccessLog << "Found" << DoLog;
      push(std::move(NewItem));
      return;
    }
    // Adjust the SourceIndex, to set the correct Source
    if (NewItem.nextSourceUse() != nullptr) {
      NewItem.advanceToNextSource();
      CSVAccessLog << "NextSrc" << DoLog;
    }
  }
  // If we reach this point the current Value V only has source that were
  // already explored.
  if (NewItem.getNumSources()) {
    CSVAccessLog << "DONE" << DoLog;
    computeOffsetsFromSources(NewItem, IsLoad);
  }
}

static bool callsBuiltinMemcpy(const Instruction *TheCall) {
  const Function *Callee = getCallee(TheCall);
  return Callee != nullptr and Callee->getIntrinsicID() == Intrinsic::memcpy;
}

void CPUSAOA::analyzeAccess(Instruction *LoadOrStore, bool IsLoad) {

  // This analysis starts from the Instruction LoadOrStore and works in two
  // alternate steps:
  // 1) it iterates backward, looking for all the values that generate their
  // pointer operands;
  // 2) as soon as the backward exploration reaches the leaves (i.e. the initial
  // values that are used as building blocks for the computation) it starts
  // working forward. At each step of the forward propagation it tries to
  // constant-fold the sources of the current value. If all the offsets of all
  // the sources are known they are constant folded and the analysis keeps
  // working forward. Otherwise, if there is a source that it still unexplored
  // (i.e. it has no known offsets), the analysis starts working backward again,
  // until also the unexplored source is resolved and can be constant folded.

  // Initialization
  if (not callsBuiltinMemcpy(LoadOrStore))
    push(WorkItem(LoadOrStore));
  else
    push(WorkItem(cast<CallInst>(LoadOrStore), IsLoad));

  while (not WorkList.empty()) {
    const auto size = WorkList.size();
    Value *CurSrcVal = WorkList.back().currentSourceValue();
    if (CSVAccessLog.isEnabled()) {
      const auto *CurVal = WorkList.back().Val();
      CSVAccessLog << "Val   : " << CurVal << DoLog;
      CSVAccessLog << "Src   : " << CurSrcVal << DoLog;
    }

    // Explore CurSrcVal's immediate sources (going backward)
    exploreImmediateSources(CurSrcVal, IsLoad);

    // If we pushed something on the WorkList we want to keep exploring back
    if (size < WorkList.size())
      continue;
    CSVAccessLog << "not grown" << DoLog;

    // If we didn't push anything, we are done exploring backward the current
    // source and we want to explore backward the other sources of this value
    if (size == WorkList.size())
      if (selectNextSource(WorkList.back()))
        continue;

    CSVAccessLog << "Done" << DoLog;

    // If we reach this point we have finished exploring all the sources of
    // the item that is currently on top of the WorkList.
    // The backward propagation is complete for now.
    // We want to fold the results of the sources on the Item, store the
    // result in the OffsetMap, and then pop the Item
    do {
      const WorkItem &Item = WorkList.back();
      if (CSVAccessLog.isEnabled()) {
        const auto *Val = Item.Val();
        CSVAccessLog << "TopItemVal: " << Val << DoLog;
        const auto *SrcVal = Item.currentSourceValue();
        CSVAccessLog << "CurSrc    : " << SrcVal << DoLog;
      }

      // Constant fold the finished value and pop it.
      computeOffsetsFromSources(Item, IsLoad);
      pop();
    } while (not WorkList.empty()
             and WorkList.back().nextSourceValue() == nullptr);

    if (not WorkList.empty()) {
      selectNextSource(WorkList.back());
    }
  }
}

template<bool IsLoad>
void CPUSAOA::computeAggregatedOffsets() {
  const InstrPtrSet &Tainted = IsLoad ? TaintedAccesses.TaintedLoads :
                                        TaintedAccesses.TaintedStores;
  ValueCallSiteOffsetMap &AccessCSOffsets = IsLoad ? LoadCallSiteOffsets :
                                                     StoreCallSiteOffsets;
  CallSiteOffsetMap &CallSiteOffsets = IsLoad ? CallSiteLoadOffsets :
                                                CallSiteStoreOffsets;
  AccessOffsetMap &AccessOffsets = IsLoad ? LoadOffsets : StoreOffsets;

  for (std::pair<Value *const, CallSiteOffsetMap> &ACSO : AccessCSOffsets) {

    Value *I = ACSO.first;
    auto DL = M.getDataLayout();

    bool isInstr = isa<Instruction>(I);
    bool isCorrectAccessType = IsLoad ? isa<LoadInst>(I) : isa<StoreInst>(I);
    bool isCallToBuiltinMemcpy = callsBuiltinMemcpy(dyn_cast<Instruction>(I));
    revng_assert(isInstr and (isCorrectAccessType or isCallToBuiltinMemcpy));
    auto *Instr = dyn_cast<Instruction>(I);
    revng_assert(Tainted.count(Instr) != 0);

    int64_t AccessSize;
    if (isCallToBuiltinMemcpy) {
      auto Call = cast<CallInst>(I);
      auto SizeParam = cast<ConstantInt>(Call->getArgOperand(2));
      AccessSize = SizeParam->getSExtValue();
    } else if (IsLoad) {
      auto *Load = cast<LoadInst>(Instr);
      auto *PtrTy = cast<PointerType>(Load->getPointerOperand()->getType());
      Type *LoadedType = PtrTy->getElementType();
      AccessSize = DL.getTypeAllocSize(LoadedType);
    } else {
      auto Store = cast<StoreInst>(Instr);
      auto *PtrTy = cast<PointerType>(Store->getPointerOperand()->getType());
      Type *StoredType = PtrTy->getElementType();
      AccessSize = DL.getTypeAllocSize(StoredType);
    }
    revng_assert(AccessSize != 0);

    CallSiteOffsetMap &CallSiteMap = ACSO.second;
    for (std::pair<CallInst *const, CSVOffsets> &CSO : CallSiteMap) {
      CallInst *const Call = CSO.first;
      CSVOffsets &O = CSO.second;
      revng_assert(O.isPtr());

      bool Inserted;

      // Compute AccessOffsets, i.e. the set of offsets accessed by each access
      {
        AccessOffsetMap::iterator AccessOffsetIt;
        auto Offset = std::make_pair(Instr, O);
        std::tie(AccessOffsetIt, Inserted) = AccessOffsets.insert(Offset);
        if (not Inserted)
          AccessOffsetIt->second.combine(O);
      }

      // Compute CallSiteOffsets, i.e. the set of offsets that might be accessed
      // from a given call in root.
      // This is a little more tricky than computing AccessOffsets, since here,
      // when we collapse on the call site, we lose the information on the
      // specific instruction that caused a given offset to be computed, hence
      // also losing the size of the access. For this reason here we have to
      // take into account the sizes of all the accesses.
      {
        OptCSVOffsets New;
        if (not O.hasOffsetSet()) {
          New = O;
        } else {
          revng_assert(O.size());
          std::set<int64_t> FineGrainedOffsets;
          // Now compute the fine-grained offsets
          for (const int64_t Coarse : O) {
            int64_t Refined = Coarse;
            int64_t End = Coarse + AccessSize;
            while (Refined < End) {
              unsigned InternalOffset = 0;
              GlobalVariable *AccessedVar;
              std::tie(AccessedVar,
                       InternalOffset) = Variables->getByEnvOffset(Refined);
              int64_t SizeAtOffset = 0;
              if (AccessedVar != nullptr) {
                Type *AccessedTy = AccessedVar->getType();
                SizeAtOffset = DL.getTypeAllocSize(AccessedTy) - InternalOffset;
                revng_assert(SizeAtOffset > 0);
                FineGrainedOffsets.insert(Refined - InternalOffset);
                CSVAccessLog << "Value: " << I << DoLog;
                CSVAccessLog << "Insert Refined: " << Refined << DoLog;
              } else {
                // Skip padding one byte at a time, without adding offsets
                SizeAtOffset = 1;
              }
              revng_assert(SizeAtOffset != 0);
              Refined += SizeAtOffset;
            }
          }
          New = CSVOffsets(O.getKind(), FineGrainedOffsets);
        }
        // Finally insert them or combine them
        CallSiteOffsetMap::iterator CallOffsetIt;
        std::tie(CallOffsetIt,
                 Inserted) = CallSiteOffsets.insert({ Call, New.getValue() });
        if (not Inserted)
          CallOffsetIt->second.combine(New.getValue());
      }
    }
  }
}

bool CPUSAOA::run() {

  // Analyze load and store
  for (Instruction *I : TaintedAccesses.TaintedLoads)
    analyzeAccess(I, true);
  for (Instruction *I : TaintedAccesses.TaintedStores)
    analyzeAccess(I, false);

  if (CSVAccessLog.isEnabled()) {
    TaintLog << "== ACCESS ANALYSIS RESULTS ==\n";
    TaintLog << "== Loads ==\n";
    for (Instruction *LoadOrStore : TaintedAccesses.TaintedLoads) {
      TaintLog << "INSTRUCTION: " << LoadOrStore << DoLog;
      TaintLog << dumpToString(LoadOrStore) << DoLog;
      TaintLog.indent(4);
      for (const auto &CSO : LoadCallSiteOffsets.at(LoadOrStore)) {
        TaintLog << "CallSite: " << CSO.first << '\n';
        if (CSO.first != nullptr)
          TaintLog << dumpToString(CSO.first);
        TaintLog << DoLog;
        TaintLog << CSO.second << '\n';
      }
      TaintLog.unindent(4);
      TaintLog << DoLog;
    }
    TaintLog << "== Stores ==\n";
    for (Instruction *LoadOrStore : TaintedAccesses.TaintedStores) {
      TaintLog << "INSTRUCTION: " << LoadOrStore << DoLog;
      TaintLog << dumpToString(LoadOrStore) << DoLog;
      TaintLog.indent(4);
      for (const auto &CSO : StoreCallSiteOffsets.at(LoadOrStore)) {
        TaintLog << "CallSite: " << CSO.first << '\n';
        if (CSO.first != nullptr)
          TaintLog << dumpToString(CSO.first);
        TaintLog << DoLog;
        TaintLog << CSO.second << '\n';
      }
      TaintLog.unindent(4);
      TaintLog << DoLog;
    }
    TaintLog << DoLog;
  }

  // ValueCallSiteOffset is not needed anymore here, because it's only used
  // across different calls to analyzeAccess to optimize the runtime avoiding
  // multiple iterations on the same Values.
  ValueCallSiteOffsets = {};

  // Aggregate the results:
  // - from LoadCallSiteOffsets to CallSiteLoadOffsets and LoadOffsets
  // - from StoreCallSiteOffsets to CallSiteStoreOffsets and StoreOffsets
  computeAggregatedOffsets<true>();
  computeAggregatedOffsets<false>();

  cleanup();
  return not TaintedAccesses.empty();
}

class CPUStateAccessAnalysis {

private:
  // A reference to the analyzed Module
  const Module &M;

  VariableManager *Variables;

  // References to the maps that will be filled by this analysis.
  // Every map maps an Instruction to the CSVOffset representing all the
  // possible offsets that are accessed by that Instruction, being it either a
  // load, a store or a call to Intrinsic::memcpy.
  // We hold two separate maps, one for loads and one for stores, so that calls
  // to memcpy can be in both maps, with different associated offsets.
  AccessOffsetMap CSVLoadOffsetMap;
  AccessOffsetMap CSVStoreOffsetMap;

public:
  CPUStateAccessAnalysis(const Module &Mod, VariableManager *V) :
    M(Mod),
    Variables(V),
    EnvStructType(V->getCPUStateType()),
    DL(Mod.getDataLayout()),
    EnvStructSize(DL.getTypeAllocSize(EnvStructType)),
    Builder(Mod.getContext()),
    Int64Ty(Builder.getInt64Ty()),
    SizeOfEnv(ConstantInt::get(Int64Ty, APInt(64, EnvStructSize, true))),
    Zero(ConstantInt::get(Int64Ty, APInt(64, 0, true))),
    CPUStatePtr(Mod.getGlobalVariable("env")) {}

public:
  bool run();

private:
  template<bool IsLoad>
  std::tuple<Instruction *, Type *, Value *>
  setupOutEnvAccess(Instruction *AccessToFix);

  template<bool IsLoad>
  void correctCPUStateAccesses();

  void setupLoadInEnv(Instruction *LoadToFix,
                      int64_t EnvOffset,
                      SwitchInst *Switch,
                      BasicBlock *NextBB,
                      PHINode *Phi);

  void setupStoreInEnv(Instruction *LoadToFix,
                       int64_t EnvOffset,
                       SwitchInst *Switch,
                       BasicBlock *NextBB);

private:
  Type *EnvStructType;
  const DataLayout &DL;
  int64_t EnvStructSize;
  IRBuilder<> Builder;
  Type *Int64Ty;
  Constant *SizeOfEnv;
  Constant *Zero;
  GlobalVariable *CPUStatePtr;
};

static void addAccessMetadata(const CallSiteOffsetMap &OffsetMap,
                              VariableManager *Variables,
                              QuickMetadata &QMD,
                              unsigned MDKind) {
  for (auto &AccessOffsets : OffsetMap) {
    const CSVOffsets &Offsets = AccessOffsets.second;
    CallInst *const CallSite = AccessOffsets.first;
    if (CallSite == nullptr)
      continue;
    revng_assert(Offsets.isPtr());

    ConstantAsMetadata *UnknownAccess = nullptr;
    MDTuple *AccessedVariablesTuple = nullptr;
    SmallVector<Metadata *, 10> OffsetMetadata;
    OffsetMetadata.reserve(Offsets.size());
    SmallSet<std::string, 10> AccessedVarNames;
    if (Offsets.isUnknownInPtr()) {
      CSVAccessLog << "Unknown access to CSV" << DoLog;
      UnknownAccess = QMD.get((uint32_t) 1);
      AccessedVariablesTuple = QMD.tuple(OffsetMetadata);
    } else {
      UnknownAccess = QMD.get((uint32_t) 0);
      for (const int64_t O : Offsets) {
        CSVAccessLog << "CallSite: " << CallSite << DoLog;
        CSVAccessLog << "Refined: " << O << DoLog;
        GlobalVariable *AccessedVar = Variables->getByEnvOffset(O).first;
        std::string VarName = AccessedVar->getName();
        MDString *VarNameMD = QMD.get(VarName);
        bool NewlyInserted = AccessedVarNames.insert(VarName).second;
        if (NewlyInserted) {
          OffsetMetadata.push_back(VarNameMD);
        }
        CSVAccessLog << "Accessed Var: " << AccessedVar << " Name: " << VarName
                     << " Offset: " << Variables->getByEnvOffset(O).second
                     << DoLog;
      }
      AccessedVariablesTuple = QMD.tuple(OffsetMetadata);
    }
    SmallVector<Metadata *, 2> AccessMetadata = { UnknownAccess,
                                                  AccessedVariablesTuple };
    CallSite->setMetadata(MDKind, QMD.tuple(AccessMetadata));
  }
}

static Value *getLoadAddressValue(Instruction *I) {
  auto OpCode = I->getOpcode();
  Value *Address = nullptr;
  switch (OpCode) {
  case Instruction::Load: {
    Address = cast<LoadInst>(I)->getPointerOperand();
  } break;
  case Instruction::Call: {
    auto *Call = cast<CallInst>(I);
    Function *Callee = getCallee(Call);
    // We only support memcpys where the last parameter is constant
    revng_assert(Callee != nullptr
                 and (Callee->getIntrinsicID() == Intrinsic::memcpy
                      and isa<ConstantInt>(Call->getArgOperand(2))));
    Address = Call->getArgOperand(1);
  } break;
  default:
    revng_abort();
  }
  return Address;
}

static Type *getLoadedType(Instruction *I) {
  if (auto Load = dyn_cast<LoadInst>(I)) {
    auto *PtrTy = cast<PointerType>(Load->getPointerOperand()->getType());
    return PtrTy->getElementType();
  }
  return nullptr;
}

static Value *getStoreAddressValue(Instruction *I) {
  auto OpCode = I->getOpcode();
  Value *Address = nullptr;
  switch (OpCode) {
  case Instruction::Store: {
    Address = cast<StoreInst>(I)->getPointerOperand();
  } break;
  case Instruction::Call: {
    auto *Call = cast<CallInst>(I);
    Function *Callee = getCallee(Call);
    // We only support memcpys where the last parameter is constant
    revng_assert(Callee != nullptr
                 and (Callee->getIntrinsicID() == Intrinsic::memcpy
                      and isa<ConstantInt>(Call->getArgOperand(2))));
    Address = Call->getArgOperand(0);
  } break;
  default:
    revng_abort();
  }
  return Address;
}

static Type *getStoredType(Instruction *I) {
  if (auto Store = dyn_cast<StoreInst>(I)) {
    auto *PtrTy = cast<PointerType>(Store->getPointerOperand()->getType());
    return PtrTy->getElementType();
  }
  return nullptr;
}

static void fixEnv2EnvMemCopies(const Module &M,
                                AccessOffsetMap &CSVLoadOffsetMap,
                                AccessOffsetMap &CSVStoreOffsetMap) {
  QuickMetadata QMD(M.getContext());
  IRBuilder<> Builder(M.getContext());
  Type *CharTy = Builder.getInt8Ty();

  std::set<Instruction *> AccessToRemove;
  AccessOffsetMap NewLoadCSOffsets;
  AccessOffsetMap NewStoreCSOffsets;

  for (auto &InstCSOffset : CSVLoadOffsetMap) {
    Instruction *Instr = InstCSOffset.first;
    auto It = CSVStoreOffsetMap.find(Instr);
    if (It != CSVStoreOffsetMap.end()) {
      auto *Call = cast<CallInst>(Instr);
      Function *Memcpy = getCallee(Call);
      revng_assert(Memcpy != nullptr
                   and (Memcpy->getIntrinsicID() == Intrinsic::memcpy
                        and isa<ConstantInt>(Call->getArgOperand(2))));

      Value *MemcpySize = Call->getArgOperand(2);
      Value *MemcpySrc = Call->getArgOperand(1);
      Value *MemcpyDst = Call->getArgOperand(0);

      Function *F = Instr->getParent()->getParent();
      Builder.SetInsertPoint(&*F->getEntryBlock().begin());
      AllocaInst *TmpBuffer = Builder.CreateAlloca(CharTy, MemcpySize);

      Builder.SetInsertPoint(Instr);
      CallInst *MemcpyLoad = Builder.CreateMemCpy(TmpBuffer,
                                                  TmpBuffer->getAlignment(),
                                                  MemcpySrc,
                                                  1,
                                                  MemcpySize);
      NewLoadCSOffsets.insert({ MemcpyLoad, InstCSOffset.second });

      CallInst *MemcpyStore = Builder.CreateMemCpy(MemcpyDst,
                                                   1,
                                                   TmpBuffer,
                                                   TmpBuffer->getAlignment(),
                                                   MemcpySize);
      NewStoreCSOffsets.insert({ MemcpyStore, It->second });

      AccessToRemove.insert(Instr);
    }
  }

  for (Instruction *Instr : AccessToRemove) {
    CSVLoadOffsetMap.erase(Instr);
    CSVStoreOffsetMap.erase(Instr);
    Instr->eraseFromParent();
  }
  CSVLoadOffsetMap.insert(NewLoadCSOffsets.begin(), NewLoadCSOffsets.end());
  CSVStoreOffsetMap.insert(NewStoreCSOffsets.begin(), NewStoreCSOffsets.end());
}

static ConstantInt *getConstantOffset(Type *Int64Ty, int64_t O) {
  auto *EnvOffsetConst = ConstantInt::get(Int64Ty, APInt(64, O, true));
  return cast<ConstantInt>(EnvOffsetConst);
}

static Value *buildEnvOffsetValue(IRBuilder<> Builder,
                                  GlobalVariable *CPUStatePtr,
                                  Value *Address,
                                  Type *OffsetTy) {
  LoadInst *LoadEnv = Builder.CreateLoad(CPUStatePtr);
  Value *EnvAsInt64 = Builder.CreateZExtOrBitCast(LoadEnv, OffsetTy);
  Value *AddressAsInt64 = Builder.CreatePtrToInt(Address, OffsetTy);
  return Builder.CreateSub(AddressAsInt64, EnvAsInt64);
}

template<bool IsLoad>
std::tuple<Instruction *, Type *, Value *>
CPUStateAccessAnalysis::setupOutEnvAccess(Instruction *AccessToFix) {
  LLVMContext &Context = M.getContext();
  BasicBlock *AccessToFixBB = AccessToFix->getParent();
  Function *F = AccessToFixBB->getParent();
  auto InstrIt = AccessToFix->getIterator();
  revng_assert(InstrIt != AccessToFixBB->end());
  revng_assert(std::next(InstrIt) != AccessToFixBB->end());
  // Create a new block NextBB and move there all the instructions after
  // the access
  BasicBlock *NextBB = BasicBlock::Create(Context, "AfterAccess", F);
  AccessToFixBB->replaceSuccessorsPhiUsesWith(NextBB);
  NextBB->getInstList().splice(NextBB->end(),
                               AccessToFixBB->getInstList(),
                               std::next(InstrIt),
                               AccessToFixBB->getInstList().end());
  revng_assert(not NextBB->empty());

  // Create a new block OutAccessBB only for accesses outside env, and
  // clone the accessing instruction in there. This clone of the
  // accessing instruction in the OutAccessBB will be leaved untouched
  // by the substitution performed later.
  BasicBlock *OutAccessBB = BasicBlock::Create(Context, "OutAccess", F);
  Builder.SetInsertPoint(OutAccessBB);
  BranchInst *OutToNextBranchInst = Builder.CreateBr(NextBB);
  Instruction *OutAccess = AccessToFix->clone();
  OutAccess->insertBefore(OutToNextBranchInst);

  // Create a new block InAccessBB only for accesses inside env (if any).
  // The accessing instruction is cloned into the new InAccessBB
  // and we insert a branch instruction to the NextBB.
  BasicBlock *InAccessBB = BasicBlock::Create(Context, "InAccess", F);
  Builder.SetInsertPoint(InAccessBB);
  BranchInst *InToNextBranchInst = Builder.CreateBr(NextBB);
  Instruction *InAccessToFix = AccessToFix->clone();
  InAccessToFix->insertBefore(InToNextBranchInst);

  // Create a conditional branch to jump to InAccessBB if the access is
  // going to be in env, or to OutAccessBB if the access is going to be
  // out of env
  Value *Address = IsLoad ? getLoadAddressValue(AccessToFix) :
                            getStoreAddressValue(AccessToFix);

  Builder.SetInsertPoint(AccessToFixBB);
  Value *OffsetValue = buildEnvOffsetValue(Builder,
                                           CPUStatePtr,
                                           Address,
                                           Int64Ty);
  Value *GEZero = Builder.CreateICmpSGE(OffsetValue, Zero);
  Value *LTSizeOf = Builder.CreateICmpSLT(OffsetValue, SizeOfEnv);
  Value *IsInCSV = Builder.CreateOr(GEZero, LTSizeOf);
  Builder.CreateCondBr(IsInCSV, InAccessBB, OutAccessBB);

  if (IsLoad) {
    Type *LoadedType = getLoadedType(AccessToFix);
    if (LoadedType != nullptr) {
      Builder.SetInsertPoint(&NextBB->front());
      PHINode *PN = Builder.CreatePHI(LoadedType, 2);
      AccessToFix->replaceAllUsesWith(PN);
      PN->addIncoming(OutAccess, OutAccessBB);
      PN->addIncoming(InAccessToFix, InAccessBB);
    }
    return std::make_tuple(InAccessToFix, LoadedType, OffsetValue);
  }
  return std::make_tuple(InAccessToFix, nullptr, OffsetValue);
}

void CPUStateAccessAnalysis::setupLoadInEnv(Instruction *LoadToFix,
                                            int64_t EnvOffset,
                                            SwitchInst *Switch,
                                            BasicBlock *NextBB,
                                            PHINode *Phi) {
  LLVMContext &Context = M.getContext();
  Function *F = LoadToFix->getParent()->getParent();
  auto *OffsetConstInt = getConstantOffset(Int64Ty, EnvOffset);
  BasicBlock *CaseBlock = BasicBlock::Create(Context, "CaseInLoad");
  Builder.SetInsertPoint(CaseBlock);
  BranchInst *Break = Builder.CreateBr(NextBB);

  Instruction *Clone = LoadToFix->clone();
  Clone->insertBefore(Break);

  bool Ok = false;
  Builder.SetInsertPoint(Clone);
  switch (LoadToFix->getOpcode()) {

  case Instruction::Load: {

    Type *OriginalLoadedType = Clone->getType();
    unsigned Size = DL.getTypeAllocSize(OriginalLoadedType);
    revng_assert(Size != 0);
    auto *Loaded = Variables->loadFromEnvOffset(Builder, Size, EnvOffset);
    Ok = Loaded != nullptr;
    if (Ok) {
      Type *LoadedType = Loaded->getType();
      if (LoadedType != OriginalLoadedType) {
        unsigned LoadedSize = DL.getTypeAllocSize(LoadedType);
        revng_assert(LoadedSize == Size);
        Loaded = Builder.CreateIntToPtr(Loaded, OriginalLoadedType);
      }
      CaseBlock->insertInto(F);
      Switch->addCase(OffsetConstInt, CaseBlock);
      // Add an incoming edge for the PHI after the switch if necessary.
      if (Phi != nullptr)
        Phi->addIncoming(Loaded, CaseBlock);
      Clone->replaceAllUsesWith(Loaded);
      Clone->eraseFromParent();
    } else {
      delete CaseBlock;
      CaseBlock = nullptr; // Prevent this from being used
    }

  } break;

  case Instruction::Call: {
    CallInst *Call = cast<CallInst>(Clone);
    Ok = Variables->memcpyAtEnvOffset(Builder, Call, EnvOffset, true);
    Clone->eraseFromParent();
    if (Ok) {
      CaseBlock->insertInto(F);
      Switch->addCase(OffsetConstInt, CaseBlock);
    } else {
      delete CaseBlock;
      CaseBlock = nullptr; // Prevent this from being used
    }

  } break;

  default:
    revng_abort();
  }
}

void CPUStateAccessAnalysis::setupStoreInEnv(Instruction *StoreToFix,
                                             int64_t EnvOffset,
                                             SwitchInst *Switch,
                                             BasicBlock *NextBB) {
  LLVMContext &Context = M.getContext();
  Function *F = StoreToFix->getParent()->getParent();
  auto *OffsetConstInt = getConstantOffset(Int64Ty, EnvOffset);
  BasicBlock *CaseBlock = BasicBlock::Create(Context, "CaseInStore");
  Builder.SetInsertPoint(CaseBlock);
  BranchInst *Break = Builder.CreateBr(NextBB);

  Instruction *Clone = StoreToFix->clone();
  Clone->insertBefore(Break);

  bool Ok = false;
  Builder.SetInsertPoint(Clone);
  switch (StoreToFix->getOpcode()) {

  case Instruction::Store: {
    auto *Store = cast<StoreInst>(Clone);
    auto *ToStore = Store->getValueOperand();
    unsigned Size = DL.getTypeAllocSize(ToStore->getType());
    revng_assert(Size != 0);
    Ok = Variables->storeToEnvOffset(Builder, Size, EnvOffset, ToStore);
  } break;

  case Instruction::Call: {
    CallInst *Call = cast<CallInst>(Clone);
    Ok = Variables->memcpyAtEnvOffset(Builder, Call, EnvOffset, false);
  } break;

  default:
    revng_abort();
  }

  Clone->eraseFromParent();
  if (Ok) {
    CaseBlock->insertInto(F);
    Switch->addCase(OffsetConstInt, CaseBlock);
  } else {
    delete CaseBlock;
  }
}

template<bool IsLoad>
void CPUStateAccessAnalysis::correctCPUStateAccesses() {
  AccessOffsetMap &CSVAccessOffsetMap = IsLoad ? CSVLoadOffsetMap :
                                                 CSVStoreOffsetMap;

  if (IsLoad)
    FixAccessLog << "######## Fixing Loads ########" << DoLog;
  else
    FixAccessLog << "######## Fixing Stores ########" << DoLog;

  for (std::pair<Instruction *const, CSVOffsets> &IOff : CSVAccessOffsetMap) {
    Instruction *const I = IOff.first;
    CSVOffsets &Offsets = IOff.second;
    CSVOffsets::Kind OKind = Offsets.getKind();
    FixAccessLog << "Fixing access: " << I
                 << "\nCSVOffsets Kind: " << CSVOffsets::toString(OKind)
                 << DoLog;
    revng_assert(CSVOffsets::isPtr(OKind));
    Function *F = I->getParent()->getParent();
    Instruction *AccessToFix = I;

    switch (OKind) {
    case CSVOffsets::Kind::Unknown:
    case CSVOffsets::Kind::Numeric:
      revng_abort();
    case CSVOffsets::Kind::OutAndKnownInPtr:
    case CSVOffsets::Kind::UnknownInPtr:
    case CSVOffsets::Kind::OutAndUnknownInPtr:
    case CSVOffsets::Kind::KnownInPtr: {

      revng_log(FixAccessLog, "Before: " << dumpToString(F));

      // This is necessary to get the correct debug info.
      // Setting the insert point to an Instruction also updates the Builder
      // to use its debug info until the insert point is set to a new
      // instruction.
      // Given that in the rest of the code we mostly use
      // SetInsertPoint(BasicBlock *), which does not reset the debug info, we
      // want to do it now, otherwise we might end up using the wrong debug
      // info from an instruction of a previous iteration of this loop.
      Builder.SetInsertPoint(I);

      Value *Address = nullptr;
      Type *LoadedType = nullptr; // This is not used if IsLoad is false
      Type *StoredType = nullptr; // This is not used if IsLoad is true
      if (IsLoad) {
        Address = getLoadAddressValue(I);
        LoadedType = getLoadedType(I);
      } else {
        Address = getStoreAddressValue(I);
        StoredType = getStoredType(I);
      }

      Value *OffsetValue = nullptr;
      if (CSVOffsets::isInOutPtr(OKind))
        std::tie(AccessToFix,
                 LoadedType,
                 OffsetValue) = setupOutEnvAccess<IsLoad>(I);

      revng_assert(AccessToFix != nullptr);
      LLVMContext &Context = M.getContext();
      QuickMetadata QMD(Context);

      if (IsLoad) {
        if (LoadedType != nullptr and LoadedType->isPointerTy()) {
          FixAccessLog << "REPLACE!" << DoLog;
          Constant *NullPtr = Constant::getNullValue(LoadedType);
          AccessToFix->replaceAllUsesWith(NullPtr);
          break; // out from the big switch to the verify and cleanup code
        }
        // TODO: Handle memcpy, not necessary for now
      } else {
        if (StoredType != nullptr and StoredType->isPointerTy()) {
          FixAccessLog << "REPLACE!" << DoLog;
          break; // out from the big switch to the verify and cleanup code
        }
        // TODO: Handle memcpy, not necessary for now
      }

      // filter out cases where a switch is not necessary
      if (Offsets.size() == 1) {
        int64_t Offset = *Offsets.begin();

        Instruction *Clone = AccessToFix->clone();
        Clone->insertAfter(AccessToFix);
        AccessToFix->replaceAllUsesWith(Clone);

        Builder.SetInsertPoint(Clone);
        bool Ok = false;
        switch (AccessToFix->getOpcode()) {

        case Instruction::Load: {
          if (IsLoad) {
            Type *OriginalLoadedType = Clone->getType();
            unsigned Size = DL.getTypeAllocSize(OriginalLoadedType);
            revng_assert(Size != 0);
            auto *Loaded = Variables->loadFromEnvOffset(Builder, Size, Offset);
            Ok = Loaded != nullptr;
            if (Ok) {
              Type *LoadedType = Loaded->getType();
              if (LoadedType != OriginalLoadedType) {
                unsigned LoadedSize = DL.getTypeAllocSize(LoadedType);
                revng_assert(LoadedSize == Size);
                Loaded = Builder.CreateIntToPtr(Loaded, OriginalLoadedType);
              }
              Clone->replaceAllUsesWith(Loaded);
            }
          } else {
            revng_abort();
          }
        } break;

        case Instruction::Store: {
          if (not IsLoad) {
            auto *Store = cast<StoreInst>(Clone);
            auto *ToStore = Store->getValueOperand();
            unsigned Size = DL.getTypeAllocSize(ToStore->getType());
            revng_assert(Size != 0);
            Ok = Variables->storeToEnvOffset(Builder, Size, Offset, ToStore);
          } else {
            revng_abort();
          }
        } break;

        case Instruction::Call: {
          CallInst *Call = cast<CallInst>(Clone);
          Ok = Variables->memcpyAtEnvOffset(Builder, Call, Offset, IsLoad);
        } break;

        default:
          revng_abort();
        }

        if (not Ok) {
          Builder.SetInsertPoint(Clone);
          CallInst *CallAbort = Builder.CreateCall(M.getFunction("abort"));
          auto InvalidMDKind = Context.getMDKindID("InvalidUniqueInAccess");
          CallAbort->setMetadata(InvalidMDKind, QMD.tuple((uint32_t) 0));
        } else {
          Clone->eraseFromParent();
        }

        break; // out of the big switch to the verify and cleanup code
      }

      // Create a new block NextBB, after the Switch, and move there all the
      // instructions after the access
      BasicBlock *AccessToFixBB = AccessToFix->getParent();
      auto InstrIt = AccessToFix->getIterator();
      revng_assert(InstrIt != AccessToFixBB->end());
      revng_assert(std::next(InstrIt) != AccessToFixBB->end());
      BasicBlock *NextBB = BasicBlock::Create(Context, "AfterInAccess", F);
      AccessToFixBB->replaceSuccessorsPhiUsesWith(NextBB);
      NextBB->getInstList().splice(NextBB->end(),
                                   AccessToFixBB->getInstList(),
                                   std::next(InstrIt),
                                   AccessToFixBB->getInstList().end());
      revng_assert(not NextBB->empty());
      revng_assert(std::next(InstrIt) == AccessToFixBB->end());
      // If we're processing loads, add a PHI in NextBB if necessary
      PHINode *Phi = nullptr;
      if (IsLoad) {
        if (LoadedType != nullptr) {
          Builder.SetInsertPoint(&NextBB->front());
          Phi = Builder.CreatePHI(LoadedType, Offsets.size());
          AccessToFix->replaceAllUsesWith(Phi);
        }
      }

      // Create the default BB for the switch, calling revng_abort()
      BasicBlock *Default = BasicBlock::Create(Context, "DefaultInAccess", F);
      Builder.SetInsertPoint(Default);
      CallInst *CallAbort = Builder.CreateCall(M.getFunction("abort"));
      auto UnexpectedInMDKind = Context.getMDKindID("UnexpectedInAccess");
      CallAbort->setMetadata(UnexpectedInMDKind, QMD.tuple((uint32_t) 0));
      Builder.CreateUnreachable();

      // Create the offset value to use as a variable for the switch if
      // necessary
      Builder.SetInsertPoint(AccessToFixBB);
      if (OffsetValue == nullptr)
        OffsetValue = buildEnvOffsetValue(Builder,
                                          CPUStatePtr,
                                          Address,
                                          Int64Ty);
      revng_assert(OffsetValue != nullptr);

      if (CSVOffsets::isUnknownInPtr(OKind)) {

        if (FixAccessLog.isEnabled()) {
          ++NumUnknown;
          FunToNumUnknown[F->getName()]++;
          FunToUnknowns[F->getName()].insert(dumpToString(AccessToFix));
        }

        SwitchInst *SwitchOffset = Builder.CreateSwitch(OffsetValue,
                                                        Default,
                                                        EnvStructSize);
        for (int64_t CurrEnvOff = 0; CurrEnvOff < EnvStructSize; ++CurrEnvOff) {
          if (IsLoad)
            setupLoadInEnv(AccessToFix, CurrEnvOff, SwitchOffset, NextBB, Phi);
          else
            setupStoreInEnv(AccessToFix, CurrEnvOff, SwitchOffset, NextBB);
        }
        revng_assert(SwitchOffset->getNumCases() > 0);
        break; // out from the switch, to the verify and cleanup code
      }

      revng_assert(Offsets.size() > 0);
      SwitchInst *SwitchOffset = Builder.CreateSwitch(OffsetValue,
                                                      Default,
                                                      Offsets.size());
      for (const int64_t CurrEnvOff : Offsets) {
        if (IsLoad)
          setupLoadInEnv(AccessToFix, CurrEnvOff, SwitchOffset, NextBB, Phi);
        else
          setupStoreInEnv(AccessToFix, CurrEnvOff, SwitchOffset, NextBB);
      }

      if (IsLoad) {
        if (Phi != nullptr and Phi->getNumIncomingValues() == 0) {
          Builder.SetInsertPoint(Phi);

          CallInst *CallAbort = Builder.CreateCall(M.getFunction("abort"));
          auto NeverValidInMDKind = Context.getMDKindID("NeverValidInLoad");
          CallAbort->setMetadata(NeverValidInMDKind, QMD.tuple((uint32_t) 0));

          Instruction *DisabledInLoad = AccessToFix->clone();
          DisabledInLoad->insertBefore(Phi);
          Phi->replaceAllUsesWith(DisabledInLoad);
          Phi->eraseFromParent();
        }
      }
    } break;
    }

    // Verify the transformation and cleanup the access to fix
    revng_log(FixAccessLog, "After: " << dumpToString(F));
    if (I != AccessToFix) {
      if (FixAccessLog.isEnabled()) {
        FixAccessLog << "Erasing AccessToFix: " << AccessToFix << DoLog;
        FixAccessLog << dumpToString(AccessToFix) << DoLog;
      }
      AccessToFix->eraseFromParent();
    }
    if (FixAccessLog.isEnabled()) {
      FixAccessLog << "Erasing I: " << I << DoLog;
      FixAccessLog << dumpToString(I) << DoLog;
    }
    I->eraseFromParent();
  }
}

bool CPUStateAccessAnalysis::run() {

  if (CPUStatePtr == nullptr)
    return false;

  // Get the root Function
  const Function *RootFunction = M.getFunction("root");
  revng_assert(RootFunction);

  // Preprocessing: detect all the functions that are directly reachable from
  // the RootFunction
  auto ReachedFunctions = computeDirectlyReachableFunctions(RootFunction);

  // Start with a forward taint analysis, to detect all the tainted Values,
  // and all the tainted loads and stores.
  CSVAccessLog << "Before Taint Analysis" << DoLog;
  const auto TaintResults = forwardTaintAnalysis(CPUStatePtr, ReachedFunctions);
  CSVAccessLog << "After Taint Analysis" << DoLog;

  // If there are no tainted loads and stores we don't need to run the CPUSAOA.
  if (TaintResults.TaintedLoads.empty() and TaintResults.TaintedStores.empty())
    return TaintResults.IllegalCalls.size();

  if (TaintLog.isEnabled()) {
    TaintLog << "==== Tainted Loads =====\n";
    for (const Instruction *I : TaintResults.TaintedLoads) {
      TaintLog << I << DoLog;
      TaintLog << dumpToString(I) << DoLog;
      std::string Name = I->getParent()->getParent()->getName();
      TaintLog << "In Function: " << Name << DoLog;
    }
    TaintLog << "==== Tainted Stores ====\n";
    for (const Instruction *I : TaintResults.TaintedStores) {
      TaintLog << I << DoLog;
      TaintLog << dumpToString(I) << DoLog;
      std::string Name = I->getParent()->getParent()->getName();
      TaintLog << "In Function: " << Name << DoLog;
    }
    TaintLog << "==== Illegal Calls =====\n";
    for (const Instruction *I : TaintResults.IllegalCalls) {
      TaintLog << I << DoLog;
      TaintLog << dumpToString(I) << DoLog;
      std::string Name = I->getParent()->getParent()->getName();
      TaintLog << "In Function: " << Name << DoLog;
    }
    TaintLog << "========================" << DoLog;
  }

  CallSiteOffsetMap CallSiteLoadOffset;
  CallSiteOffsetMap CallSiteStoreOffset;
  auto AccessOffsetAnalysis = CPUSAOA(M,
                                      CPUStatePtr,
                                      RootFunction,
                                      ReachedFunctions,
                                      TaintResults,
                                      Variables,
                                      CSVLoadOffsetMap,
                                      CSVStoreOffsetMap,
                                      CallSiteLoadOffset,
                                      CallSiteStoreOffset);
  bool Found = AccessOffsetAnalysis.run();
  LLVMContext &Context = M.getContext();
  QuickMetadata QMD(Context);
  if (Found) {
    auto LoadMDKind = Context.getMDKindID("revng.csvaccess.offsets."
                                          "load");
    addAccessMetadata(CallSiteLoadOffset, Variables, QMD, LoadMDKind);
    auto StoreMDKind = Context.getMDKindID("revng.csvaccess.offsets."
                                           "store");
    addAccessMetadata(CallSiteStoreOffset, Variables, QMD, StoreMDKind);
  }

  // CallSiteLoadOffset and CallSiteStoreOffset are not used anymore
  CallSiteLoadOffset = {};
  CallSiteStoreOffset = {};

  // Decompose memcpy from env to env into two separate memcpy, the fisrt to do
  // the load, the second to do the store
  fixEnv2EnvMemCopies(M, CSVLoadOffsetMap, CSVStoreOffsetMap);
  // Fix loads
  correctCPUStateAccesses<true>();
  // Fix stores
  correctCPUStateAccesses<false>();
  revng_assert(not verifyModule(M, &dbgs()));

  if (FixAccessLog.isEnabled()) {
    FixAccessLog << "Num Unknowns: " << NumUnknown << DoLog;

    for (const auto &Fun2Num : FunToNumUnknown)
      FixAccessLog << Fun2Num.first << ": " << Fun2Num.second << DoLog;

    for (const auto &Fun2Unknowns : FunToUnknowns)
      for (const auto &U : Fun2Unknowns.second)
        FixAccessLog << Fun2Unknowns.first << ": " << U << DoLog;
  }

  return Found;
}

bool CPUStateAccessAnalysisPass::runOnModule(Module &Mod) {
  CPUStateAccessAnalysis CSVAccessAnalysis(Mod, Variables);
  return CSVAccessAnalysis.run();
}

char CPUStateAccessAnalysisPass::ID = 0;

using RegisterCPUSAAP = RegisterPass<CPUStateAccessAnalysisPass>;
static RegisterCPUSAAP
  X("cpustate-access-analysis", "CPUState Access Analysis Pass", false, false);
