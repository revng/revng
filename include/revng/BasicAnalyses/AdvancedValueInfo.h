#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <algorithm>
#include <iterator>
#include <memory>
#include <set>
#include <vector>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/Analysis/LazyValueInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/ConstantRange.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/Support/raw_os_ostream.h"

#include "revng/ADT/ConstantRangeSet.h"
#include "revng/BasicAnalyses/MaterializedValue.h"
#include "revng/Support/Debug.h"
#include "revng/Support/GraphAlgorithms.h"
#include "revng/Support/IRHelpers.h"
#include "revng/Support/MonotoneFramework.h"

inline Logger<> AVILogger("avi");

using range_size_t = uint64_t;
const range_size_t MaxMaterializedValues = (1 << 16);

inline unsigned getTypeSize(const llvm::DataLayout &DL, llvm::Type *T) {
  using namespace llvm;
  if (auto *IntegerTy = dyn_cast<IntegerType>(T))
    return IntegerTy->getBitWidth();
  else if (auto *PtrTy = dyn_cast<PointerType>(T))
    return DL.getPointerSize() * 8;
  else
    revng_abort();
}

/// Return the only Unknown value in the SCEV (if no AddRec/CouldNotCompute)
inline llvm::Value *
getUniqueUnknown(llvm::ScalarEvolution &SE, const llvm::SCEV *SC) {
  using namespace llvm;

  class FindSingleUnknown {
  private:
    bool Stop = false;

  public:
    Value *UniqueUnknown = nullptr;

  public:
    bool follow(const SCEV *S) {
      if (Stop)
        return false;

      switch (S->getSCEVType()) {
      case scConstant:
      case scTruncate:
      case scZeroExtend:
      case scSignExtend:
      case scMulExpr:
      case scSMaxExpr:
      case scUMaxExpr:
      case scSMinExpr:
      case scUMinExpr:
      case scUDivExpr:
      case scAddExpr:
      case scPtrToInt:
      case scSequentialUMinExpr:
        break;

      case scUnknown:
        if (UniqueUnknown == nullptr) {
          UniqueUnknown = cast<SCEVUnknown>(S)->getValue();
        } else {
          UniqueUnknown = nullptr;
          Stop = true;
        }

        break;

      case scAddRecExpr:
      case scCouldNotCompute:
        UniqueUnknown = nullptr;
        Stop = true;
        break;
      }

      return not Stop;
    }

    bool isDone() const { return Stop; }
  };

  FindSingleUnknown FSU;
  visitAll(SC, FSU);

  if (FSU.UniqueUnknown != nullptr)
    revng_assert(not SE.containsAddRecurrence(SC));

  return FSU.UniqueUnknown;
}

namespace detail {
using GraphValue = llvm::SmallVector<llvm::BasicBlock *, 2>;
using Graph = llvm::DenseMap<llvm::BasicBlock *, std::unique_ptr<GraphValue>>;
} // namespace detail

inline llvm::ConstantInt *replaceAllUnknownsWith(llvm::ScalarEvolution &SE,
                                                 const llvm::SCEV *SC,
                                                 llvm::ConstantInt *C) {
  using namespace llvm;

  revng_assert(not SE.containsAddRecurrence(SC));

  class Rewriter : public SCEVRewriteVisitor<Rewriter> {
  private:
    ConstantInt *NewConstant;

  public:
    Rewriter(ScalarEvolution &SE, ConstantInt *NewConstant) :
      SCEVRewriteVisitor(SE), NewConstant(NewConstant) {}

    const SCEV *visitUnknown(const SCEVUnknown *) {
      return SE.getConstant(NewConstant);
    }
  };

  Rewriter RW(SE, C);
  return cast<SCEVConstant>(RW.visit(SC))->getValue();
}

struct CFGEdge {
  llvm::BasicBlock *Start;
  llvm::BasicBlock *End;

  bool operator<(const CFGEdge &Other) const {
    return std::tie(Start, End) < std::tie(Other.Start, Other.End);
  }

  void dump() const debug_function { dump(dbg); }

  template<typename T>
  void dump(T &Output) const {
    Output << getName(Start) << "->" << getName(End);
  }
};

/// Monotone framework to collect ConstantRangeSets from LazyValueInfo
namespace DisjointRanges {

class Element {
private:
  using Container = std::map<llvm::Instruction *, ConstantRangeSet>;
  Container Ranges;

public:
  Element() {}

  static Element bottom() { return Element(); }
  Element copy() const { return *this; }

public:
  void combine(const Element &Other) {

    // We can't use zipmap_ranges since Other is const
    for (auto &P : Other.Ranges) {
      auto It = Ranges.find(P.first);
      if (It == Ranges.end())
        Ranges[P.first] = P.second;
      else
        It->second = It->second.unionWith(P.second);
    }
  }

  bool lowerThanOrEqual(const Element &Other) const {
    for (auto &P : Ranges) {
      auto It = Other.Ranges.find(P.first);
      if (It == Other.Ranges.end() or not It->second.contains(P.second))
        return false;
    }

    return true;
  }

  ConstantRangeSet &operator[](llvm::Instruction *I) { return Ranges[I]; }
  const ConstantRangeSet &operator[](llvm::Instruction *I) const {
    return Ranges.at(I);
  }

  bool hasKey(llvm::Instruction *I) const { return Ranges.count(I) != 0; }
};

class Analysis
  : public MonotoneFramework<
      Analysis,
      llvm::BasicBlock *,
      Element,
      ReversePostOrder,
      llvm::iterator_range<
        llvm::SmallVector<llvm::BasicBlock *, 2>::const_iterator>> {
private:
  using Range = llvm::iterator_range<
    llvm::SmallVector<llvm::BasicBlock *, 2>::const_iterator>;
  using Base = MonotoneFramework<Analysis,
                                 llvm::BasicBlock *,
                                 Element,
                                 ReversePostOrder,
                                 Range>;

public:
private:
  // Using unique_ptr<SmallVector> is ensuring that the key type of the dense
  // map has a size equal to a pointer, which a significant speedup.
  detail::Graph SuccessorsCache;

  const detail::Graph *GraphSuccessors;

private:
  llvm::BasicBlock *Entry;
  llvm::LazyValueInfo &LVI;
  const llvm::DominatorTree &DT;
  std::map<llvm::Instruction *, ConstantRangeSet> InstructionRanges;
  std::set<CFGEdge> TargetEdges;
  llvm::SmallVector<llvm::BasicBlock *, 4> WhiteList;

public:
  Analysis(const llvm::SmallVectorImpl<llvm::BasicBlock *> &RPOT,
           llvm::LazyValueInfo &LVI,
           const llvm::DominatorTree &DT,
           const std::vector<llvm::Instruction *> &TargetInstructions,
           const std::vector<CFGEdge> &TargetEdges,
           const detail::Graph &FunctionCFG) :
    Base(RPOT),
    GraphSuccessors(&FunctionCFG),
    Entry(RPOT[0]),
    LVI(LVI),
    DT(DT) {
    using namespace llvm;

    registerExtremal(Entry);

    for (Instruction *I : TargetInstructions) {
      if (auto *Ty = dyn_cast<IntegerType>(I->getType())) {
        InstructionRanges[I] = { ConstantRange(Ty->getIntegerBitWidth(),
                                               true) };
      }
    }

    for (const CFGEdge &E : TargetEdges) {
      this->TargetEdges.insert(E);
    }

    for (BasicBlock *BB : RPOT) {
      WhiteList.push_back(BB);
    }
    llvm::sort(WhiteList);

    for (BasicBlock *BB : RPOT) {
      cacheSuccessors(BB);
    }
  }

  Element extremalValue(llvm::BasicBlock *) const { return Element(); }

  void assertLowerThanOrEqual(const Element &A, const Element &B) const {
    revng_assert(A.lowerThanOrEqual(B));
  }

  DefaultInterrupt<Element> transfer(llvm::BasicBlock *BB) {
    if (TargetEdges.count({ BB, nullptr }) != 0) {
      Element Result = *compute(State[BB], BB, nullptr, true);
      return DefaultInterrupt<Element>::createInterrupt(Result);
    }

    return DefaultInterrupt<Element>::createInterrupt(State[BB]);
  }

  std::optional<Element> handleEdge(const Element &Original,
                                    llvm::BasicBlock *Source,
                                    llvm::BasicBlock *Destination) {
    bool IsTargetEdge = TargetEdges.count({ Source, Destination }) != 0;
    return compute(Original, Source, Destination, IsTargetEdge);
  }

  void cacheSuccessors(llvm::BasicBlock *L) {
    using namespace llvm;
    const auto &Res = SuccessorsCache
                        .try_emplace(L,
                                     std::make_unique<::detail::GraphValue>());
    auto &Result = Res.first->getSecond();
    auto &AllSuccessors = *GraphSuccessors->find(L)->getSecond();

    std::set_intersection(AllSuccessors.begin(),
                          AllSuccessors.end(),
                          WhiteList.begin(),
                          WhiteList.end(),
                          std::back_inserter(*Result));
  }

  const llvm::SmallVector<llvm::BasicBlock *, 2> &
  getSuccessors(llvm::BasicBlock *L) const {
    return *SuccessorsCache.find(L)->second;
  }

  Range successors(llvm::BasicBlock *L, DefaultInterrupt<Element> &I) const {
    const auto &Successors = getSuccessors(L);
    return llvm::make_range(Successors.begin(), Successors.end());
  }

  size_t
  successor_size(llvm::BasicBlock *L, DefaultInterrupt<Element> &I) const {
    return getSuccessors(L).size();
  }

  const ConstantRangeSet &get(llvm::Instruction *I) const {
    return InstructionRanges.at(I);
  }

  void dump() const debug_function { dump(dbg); }

  template<typename T>
  void dump(T &Output) const {
    for (auto &P : InstructionRanges) {
      Output << getName(P.first) << ": ";
      P.second.dump(Output);
      Output << "\n";
    }
  }

  void dumpFinalState() const { revng_abort(); }

private:
  std::optional<Element> compute(const Element &Original,
                                 llvm::BasicBlock *Source,
                                 llvm::BasicBlock *Destination,
                                 bool IsTargetEdge) {
    Element Result = Original;
    for (auto &P : InstructionRanges) {
      llvm::Instruction *I = P.first;
      ConstantRangeSet &InstructionRangeSet = P.second;

      if (not DT.dominates(I->getParent(), Source))
        continue;

      unsigned BitWidth = I->getType()->getIntegerBitWidth();
      InstructionRangeSet.setWidth(BitWidth);

      auto NewRange = llvm::ConstantRange::getFull(BitWidth);
      if (Destination == nullptr)
        NewRange = LVI.getConstantRange(I, Source->getTerminator());
      else
        NewRange = LVI.getConstantRangeOnEdge(I, Source, Destination);

      bool IsNew = not Result.hasKey(I);
      ConstantRangeSet &RangeSet = Result[I];
      if (IsNew) {
        RangeSet = NewRange;
      } else {
        RangeSet = RangeSet.intersectWith({ NewRange });
      }

      // If it's on target edge and the range set is smaller, register it
      if (IsTargetEdge)
        InstructionRangeSet = RangeSet;
    }

    return { std::move(Result) };
  }
};

} // namespace DisjointRanges

inline bool isPhiLike(llvm::Value *V) {
  return (llvm::isa<llvm::PHINode>(V) or llvm::isa<llvm::SelectInst>(V));
}

inline bool isMemory(llvm::Value *V) {
  using namespace llvm;
  V = skipCasts(V);
  return not(isa<GlobalVariable>(V) or isa<AllocaInst>(V));
}

/// An operation producing a result and having a single free operand
struct Operation {
  static const unsigned UseSCEV = std::numeric_limits<unsigned>::max();

  llvm::User *V;
  unsigned FreeOperandIndex;
  ConstantRangeSet Range;
  range_size_t RangeSize;

  Operation() : V(nullptr), FreeOperandIndex(0), Range(), RangeSize(0) {}
  Operation(llvm::User *V,
            unsigned FreeOperandIndex,
            llvm::ConstantRange Range,
            range_size_t RangeSize) :
    V(V),
    FreeOperandIndex(FreeOperandIndex),
    Range(Range),
    RangeSize(RangeSize) {}

  unsigned getBitSize() const {
    using namespace llvm;
    return cast<IntegerType>(V->getType())->getBitWidth();
  }

  bool usesSCEV() const { return FreeOperandIndex == UseSCEV; }

  void dump(unsigned Indent = 0,
            llvm::ScalarEvolution *SCEV = nullptr) const debug_function {
    dump(dbg, Indent, SCEV);
  }

  template<typename T>
  void dump(T &Output,
            unsigned Indent = 0,
            llvm::ScalarEvolution *SCEV = nullptr) const {
    std::string Prefix(Indent, ' ');
    Output << Prefix << V << "\n";

    Output << Prefix;
    if (usesSCEV()) {
      if (SCEV != nullptr) {
        Output << dumpToString(SCEV->getSCEV(V)) << "\n";
      } else {
        Output << "UsesSCEV\n";
      }
    } else {
      Output << "FreeOperandIndex: " << FreeOperandIndex << "\n";
    }

    Output << Prefix;
    Range.dump(Output);
    Output << "\n";

    Output << Prefix << "RangeSize: " << RangeSize << "\n";
  }
};

/// Class representing an expression on the IR
class Expression {
private:
  const llvm::DataLayout &DL;
  llvm::ScalarEvolution &SE;
  std::vector<Operation> OperationsStack;
  unsigned SmallestRangeIndex;
  bool PhiIsSmallest;
  MaterializedValues Values;
  bool Materialized;

public:
  using PhiEdges = std::vector<CFGEdge>;

public:
  Expression(const llvm::DataLayout &DL, llvm::ScalarEvolution &SE) :
    DL(DL), SE(SE) {
    reset();
  }

  void reset() {
    SmallestRangeIndex = 0;
    PhiIsSmallest = false;
    Materialized = false;
    OperationsStack.clear();
    Values.clear();
  }

  void dump(unsigned Indent = 0) const debug_function { dump(dbg, Indent); }

  template<typename T>
  void dump(T &Output, unsigned Indent = 0) const {
    std::string Prefix(Indent, ' ');

    Output << Prefix << "OperationStack: \n";
    unsigned I = 0;
    for (const Operation &Op : OperationsStack) {
      Output << Prefix << "  " << I;
      if (I == SmallestRangeIndex)
        Output << " [smallest]";
      Output << ":\n";
      Op.dump(Output, Indent + 4);
      ++I;
    }
    Output << "\n";

    Output << Prefix << "PhiIsSmallest: " << PhiIsSmallest << "\n";
    Output << Prefix << "Values: {";
    for (const MaterializedValue &Value : Values) {
      Output << " ";
      Value.dump(Output);
    }
    Output << " }\n";
    Output << Prefix << "Materialized: " << Materialized << "\n";
  }

  bool lastIsPhi() const { return isPhiLike(OperationsStack.back().V); }

  range_size_t smallestRangeSize() const {
    return OperationsStack.at(SmallestRangeIndex).RangeSize;
  }

  llvm::Value *smallestRangeValue() const {
    return OperationsStack.at(SmallestRangeIndex).V;
  }

  /// Builds a chain of single non-const-operand instructions until you find a
  /// phi or a load from a global variable.
  llvm::Instruction *buildExpression(llvm::User *U) {
    using namespace llvm;
    Instruction *Result = nullptr;
    do {

      revng_log(AVILogger, "  Considering " << U);

      //
      // Identify the free operand
      //
      unsigned Index = 0;
      unsigned NextIndex = 0;
      Value *Next = nullptr;

      unsigned BitWidth = getTypeSize(DL, U->getType());
      auto Range = ConstantRange::getFull(BitWidth);
      range_size_t RangeSize = MaxMaterializedValues;
      auto *I = dyn_cast<Instruction>(U);

      if (auto *Call = dyn_cast<CallInst>(U)) {
        if (Function *Callee = Call->getCalledFunction()) {
          if (Callee->getIntrinsicID() == Intrinsic::bswap) {
            // Handle bswap intrinsic
            Use &FirstArg = Call->getArgOperandUse(0);
            Next = FirstArg.get();
            NextIndex = FirstArg.getOperandNo();
          }
        }
      } else if (I != nullptr) {
        // We found an instruction
        for (Value *Operand : U->operands()) {
          if (not isa<Constant>(Operand)) {
            if (Next != nullptr) {
              Next = nullptr;
              break;
            }

            NextIndex = Index;
            Next = Operand;
          }

          Index++;
        }

        if (Next == nullptr and I->getNumOperands() > 1) {
          // The instruction has more than one free operand, let's give SCEV a
          // shot
          Next = getUniqueUnknown(SE, SE.getSCEV(I));
          if (Next == I)
            Next = nullptr;
          NextIndex = Operation::UseSCEV;
        }

      } else if (auto *C = dyn_cast<ConstantInt>(U)) {
        RangeSize = 1;
        Range = ConstantRange(C->getValue());
      } else if (isa<ConstantPointerNull>(U) or isa<UndefValue>(U)) {
        RangeSize = 0;
        Range = ConstantRange(BitWidth, false);
      } else {
        revng_assert(isa<Constant>(U));
        Next = U->getOperand(0);
      }

      // Push on the stack
      OperationsStack.push_back({ U, NextIndex, Range, RangeSize });

      if (isPhiLike(U)) {
        Result = I;
        Next = nullptr;
      } else if (auto *Load = dyn_cast<LoadInst>(U)) {
        Next = Load->getPointerOperand();
        if (not isMemory(skipCasts(Load->getPointerOperand())))
          Next = nullptr;
      }

      U = cast_or_null<User>(Next);

    } while (U != nullptr);
    return Result;
  }

  llvm::Instruction *isInteresting(const Operation &O) {
    if (O.Range.isFullSet())
      if (auto *I = dyn_cast<llvm::Instruction>(O.V))
        if (isa<llvm::IntegerType>(I->getType()))
          return I;
    return nullptr;
  };

  void computeOperationString(llvm::ArrayRef<llvm::Instruction *> Targets,
                              llvm::LazyValueInfo &LVI,
                              const llvm::DominatorTree &DT,
                              PhiEdges &Edges,
                              llvm::BasicBlock *StopAt,
                              const detail::Graph &Graph) {
    using namespace llvm;

    if (Targets.size() == 0)
      return;

    BasicBlock *StartBB = Targets.back()->getParent();

    const CFGEdge &FirstEdge = Edges.front();
    revng_assert(FirstEdge.End == nullptr);
    BasicBlock *EndBB = FirstEdge.Start;

    BasicBlock *LimitedStartBB = StartBB;
    if (not DT.dominates(StopAt, LimitedStartBB))
      LimitedStartBB = StopAt;

    SmallPtrSet<BasicBlock *, 4> IgnoreList = { StopAt };

    auto Reachable = nodesBetweenReverse(EndBB, LimitedStartBB, &IgnoreList);
    for (Instruction *I : Targets)
      Reachable.insert(I->getParent());
    Reachable.insert(EndBB);

    // Note: ordering in reverse post order is more costly than beneficial
    Reachable.erase(StartBB);
    SmallVector<BasicBlock *, 8> ReachableVector{ StartBB };
    for (BasicBlock *BB : Reachable)
      ReachableVector.push_back(BB);

    DisjointRanges::Analysis DR(ReachableVector,
                                LVI,
                                DT,
                                Targets,
                                Edges,
                                Graph);
    DR.initialize();
    DR.run();

    for (Operation &O : OperationsStack) {
      if (auto *I = isInteresting(O)) {
        O.Range = DR.get(I);
        AVILogger << I << ": ";
        O.Range.dump(AVILogger);
        AVILogger << DoLog;
        O.RangeSize = O.Range.size().getLimitedValue();
      }
    }
  }

  /// Use LVI to build an expression about \p V
  ///
  /// 1. Build a chain of single non-const-operand instructions until you find a
  ///    phi or a load from a global variable.
  /// 2. For each instruction in the chain record the number of possible values
  ///    according to LVI.
  /// 3. Iterate over the chain looking for the instruction associated with the
  ///    smallest range.
  llvm::Instruction *buildAndAnalyzeExpression(llvm::LazyValueInfo &LVI,
                                               const llvm::DominatorTree &DT,
                                               PhiEdges &Edges,
                                               llvm::Value *V,
                                               llvm::BasicBlock *StopAt,
                                               const detail::Graph &Graph) {
    using namespace llvm;

    revng_log(AVILogger, "Building expression for " << V);

    reset();

    User *U = cast<User>(V);
    llvm::Instruction *Result = buildExpression(U);

    std::vector<Instruction *> Targets;
    for (const Operation &O : OperationsStack) {
      if (auto *I = isInteresting(O)) {
        Targets.push_back(I);
      }
    }

    computeOperationString(Targets, LVI, DT, Edges, StopAt, Graph);

    for (const Operation &O : OperationsStack) {
      // Get the LVI and record if it's the smallest
      if (OperationsStack[SmallestRangeIndex].RangeSize > O.RangeSize) {
        SmallestRangeIndex = &O - &*OperationsStack.begin();
      }
    }

    return Result;
  }

  /// Materialize all the values in this expression
  template<typename MemoryOracle>
  MaterializedValues materialize(MemoryOracle &MO) {
    using namespace llvm;

    revng_assert(not Materialized);
    Materialized = true;

    IntegerType *SmallestType = nullptr;
    for (const Operation &Operation : OperationsStack) {
      if (auto *ITy = dyn_cast<IntegerType>(Operation.V->getType())) {
        if (SmallestType == nullptr
            or SmallestType->getBitWidth() > ITy->getBitWidth()) {
          SmallestType = ITy;
        }
      }
    }

    range_size_t WorstCase = MaxMaterializedValues;
    if (SmallestType != nullptr)
      WorstCase = std::min(SmallestType->getBitMask(), WorstCase);

    Type *SmallestOperationType = nullptr;
    {
      const auto &SmallestOperation = OperationsStack.at(SmallestRangeIndex);
      SmallestOperationType = SmallestOperation.V->getType();

      if (not PhiIsSmallest) {
        // Materialize all the values, so we can process them one by one
        revng_assert(Values.size() == 0);
        if (SmallestOperation.RangeSize >= WorstCase)
          return {};
        Values.resize(SmallestOperation.RangeSize);

        auto It = SmallestOperation.Range.begin();
        const auto End = SmallestOperation.Range.end();
        for (MaterializedValue &Entry : Values) {
          revng_assert(It != End);
          Entry = { *It };
          ++It;
        }

      } else {
        // The Values vector has already been initialized
        revng_assert(lastIsPhi());
      }

      OperationsStack.resize(SmallestRangeIndex);
    }

    // Process one value at a time
    for (MaterializedValue &Entry : Values) {
      using CI = ConstantInt;
      using CE = ConstantExpr;

      if (AVILogger.isEnabled()) {
        AVILogger << "Now materializing ";
        Entry.dump(AVILogger);
        AVILogger << DoLog;
      }

      std::optional<std::string> SymbolName;
      Constant *Current = nullptr;
      {
        auto Value = Entry.value();
        if (SmallestOperationType->isIntegerTy()) {
          Current = CI::get(SmallestOperationType, Value);
        } else if (SmallestOperationType->isPointerTy()) {
          auto &C = SmallestOperationType->getContext();
          Current = CI::get(DL.getIntPtrType(C), Value);
          Current = CE::getIntToPtr(Current, SmallestOperationType);
        } else {
          revng_abort();
        }
      }

      // Materialize the value I through the operations stack
      auto It = OperationsStack.rbegin();
      auto End = OperationsStack.rend();
      if (It != End) {
        auto Range = make_range(It, End);

        for (const Operation &Op : Range) {

          if (AVILogger.isEnabled()) {
            AVILogger << "  Processing:";
            Op.dump(AVILogger, 4);
            AVILogger << DoLog;
          }

          // After we get a symbol name we only track casts, additions and
          // subtractions
          auto *I = dyn_cast<Instruction>(Op.V);

          Module *M = nullptr;
          LLVMContext *Context = nullptr;
          const DataLayout *DL = nullptr;
          if (I != nullptr) {
            M = I->getParent()->getParent()->getParent();
            Context = &M->getContext();
            DL = &M->getDataLayout();
          }

          if (SymbolName and I != nullptr) {
            auto *Call = dyn_cast<CallInst>(I);
            // TODO: AND should be allowed only if second operand is a mask
            //       compatible with being a PC mask
            if (not(I->isCast() or I->getOpcode() == Instruction::Add
                    or I->getOpcode() == Instruction::Sub
                    or I->getOpcode() == Instruction::And
                    or (Call != nullptr
                        and Call->getIntrinsicID() == Intrinsic::bswap))) {
              return {};
            }
          }

          if (auto *C = dyn_cast<Constant>(Op.V)) {
            Current = C;
          } else if (auto *C = dyn_cast<Constant>(Op.V)) {
            revng_assert(Op.V->getNumOperands() == 1);
            Current = cast<Constant>(C->getOperand(0));
          } else if (auto *Load = dyn_cast<LoadInst>(Op.V)) {
            revng_assert(isMemory(skipCasts(Load->getPointerOperand())));

            const MaterializedValue &Loaded = MO.load(Load->getType(), Current);

            if (AVILogger.isEnabled()) {
              AVILogger << "  MemoryOracle says its ";
              Loaded.dump(AVILogger);
              AVILogger << DoLog;
            }

            if (not Loaded.isValid()) {
              // Couldn't read memory, bail out
              return {};
            }

            if (Loaded.hasSymbol())
              SymbolName = Loaded.symbolName();

            Type *LoadedType = Load->getType();
            if (LoadedType->isPointerTy()) {
              Current = CI::get(DL->getIntPtrType(*Context), Loaded.value());
              Current = CE::getIntToPtr(Current, LoadedType);
            } else {
              Current = CI::get(cast<IntegerType>(LoadedType), Loaded.value());
            }

          } else if (auto *Call = dyn_cast<CallInst>(Op.V)) {
            Function *Callee = Call->getCalledFunction();
            revng_assert(Callee != nullptr
                         && Callee->getIntrinsicID() == Intrinsic::bswap);

            using CI = ConstantInt;
            Current = CI::get(*Context,
                              cast<CI>(Current)->getValue().byteSwap());

          } else if (I != nullptr) {

            if (Op.usesSCEV()) {
              Current = replaceAllUnknownsWith(SE,
                                               SE.getSCEV(I),
                                               cast<ConstantInt>(Current));
            } else {
              // Build operands list patching the free operand
              SmallVector<Constant *, 4> Operands;
              unsigned Index = 0;
              for (Value *Operand : Op.V->operands()) {
                if (auto *ConstantOperand = dyn_cast<Constant>(Operand)) {
                  Operands.push_back(ConstantOperand);
                } else {
                  revng_assert(Index == Op.FreeOperandIndex);
                  Operands.push_back(Current);
                }
                Index++;
              }

              Current = ConstantFoldInstOperands(I,
                                                 Operands,
                                                 MO.getDataLayout());
            }

            revng_assert(Current != nullptr);

          } else {
            revng_abort();
          }
        }
      }

      // Ignore undef
      if (isa<UndefValue>(Current))
        return {};

      APInt Value(getTypeSize(DL, Current->getType()), 0);
      if (not Current->isNullValue()) {
        if (auto *CI = dyn_cast<ConstantInt>(skipCasts(Current))) {
          Value = CI->getValue();
        } else {
          skipCasts(Current)->dump();
          revng_abort("Unexpected Constant");
        }
      }

      if (SymbolName) {
        revng_assert(not llvm::StringRef(*SymbolName).contains('\0'));
        Entry = { *SymbolName, Value };
      } else {
        Entry = { Value };
      }
    }

    return std::move(Values);
  }

  void setPhiValues(MaterializedValues PhiValues) {
    const Operation &SmallestOperation = OperationsStack.at(SmallestRangeIndex);
    revng_assert(lastIsPhi());
    revng_assert(PhiValues.size() < SmallestOperation.RangeSize);
    Values = std::move(PhiValues);
    PhiIsSmallest = true;
    SmallestRangeIndex = OperationsStack.size() - 1;
  }
};

/// Context for processing a phi node
class PhiProcess {
public:
  /// The considered Phi
  llvm::Instruction *Phi;

  /// Index of the next incoming value of Phi to handle
  unsigned NextIncomingIndex;

  /// Set of possible values for this phi
  MaterializedValues Values;

  /// Expression representing the value of the current incoming value of the phi
  Expression Expr;

  /// Processing of the current index (NextIncomingIndex - 1) is in progress
  bool Unfinished;

  /// Size of the smallest range at the previous level
  range_size_t UpperBound;

  /// Did we exceed MaxMaterializedValues?
  bool TooLarge;

public:
  PhiProcess(const llvm::DataLayout &DL,
             llvm::ScalarEvolution &SE,
             llvm::Instruction *Phi,
             range_size_t UpperBound) :
    Phi(Phi),
    NextIncomingIndex(0),
    Expr(DL, SE),
    Unfinished(false),
    UpperBound(UpperBound),
    TooLarge(false) {

    revng_assert(isPhiLike(Phi));
  }

  void dump() const debug_function { dump(dbg); }

  template<typename T>
  void dump(T &Output) const {
    Output << "Phi: " << Phi << "\n";
    Output << "NextIncomingIndex: " << NextIncomingIndex << "\n";

    Output << "Values: {";
    for (const MaterializedValue &Value : Values) {
      Output << " ";
      Value.dump(Output);
    }
    Output << " }\n";

    Output << "Expr:\n";
    Expr.dump(Output, 2);

    Output << "Unfinished: " << Unfinished << "\n";
    Output << "UpperBound: " << UpperBound << "\n";
    Output << "TooLarge: " << TooLarge << "\n";
  }
};

/// Analysis to associate to each value a ConstantRangeSet using LazyValueInfo
///
/// \tparam MemoryOracle the type of the class used to produce obtain the result
///         of memory accesses from constant addresses.
template<typename MemoryOracle>
class AdvancedValueInfo {
private:
  llvm::LazyValueInfo &LVI;
  llvm::ScalarEvolution &SE;
  const llvm::DominatorTree &DT;
  MemoryOracle &MO;
  llvm::BasicBlock *StopAt;

  // Using unique_ptr<SmallVector> is ensuring that the key type of the dense
  // map has a size equal to a pointer, which a significant speedup
  detail::Graph SuccessorsCache;

public:
  AdvancedValueInfo(llvm::LazyValueInfo &LVI,
                    llvm::ScalarEvolution &SE,
                    const llvm::DominatorTree &DT,
                    MemoryOracle &MO,
                    llvm::BasicBlock *StopAt) :
    LVI(LVI), SE(SE), DT(DT), MO(MO), StopAt(StopAt) {
    auto *Parent = DT.getRoot()->getParent();

    for (llvm::BasicBlock &BB : *Parent) {
      auto Ptr = std::make_unique<detail::GraphValue>();
      auto Res = SuccessorsCache.try_emplace(&BB, std::move(Ptr));
      auto &Vector = *(*Res.first).getSecond();
      for (auto *Successor : llvm::successors(&BB)) {
        Vector.push_back(Successor);
      }

      llvm::sort(Vector);
    }
  }

  MaterializedValues explore(llvm::BasicBlock *BB, llvm::Value *V);
};

template<class MemoryOracle>
MaterializedValues
AdvancedValueInfo<MemoryOracle>::explore(llvm::BasicBlock *BB, llvm::Value *V) {
  using namespace llvm;
  const llvm::DataLayout &DL = getModule(BB)->getDataLayout();

  revng_log(AVILogger, "Exploring " << V << " in " << BB);

  // Create a fake Phi for the initial entry
  PHINode *FakePhi = PHINode::Create(V->getType(), 1);
  FakePhi->addIncoming(V, BB);

  struct DeleteValueOnReturn {
    Instruction *I;
    ~DeleteValueOnReturn() {
      if (I->getParent() != nullptr)
        eraseFromParent(I);
      else
        I->deleteValue();
    }
  };
  DeleteValueOnReturn X{ FakePhi };

  std::set<Instruction *> VisitedPhis;
  std::vector<PhiProcess> PendingPhis{
    { DL, SE, FakePhi, MaxMaterializedValues }
  };
  Expression::PhiEdges Edges;

  while (true) {
    PhiProcess &Current = PendingPhis.back();

    if (AVILogger.isEnabled()) {
      AVILogger << "Processing ";
      Current.dump(AVILogger);
      AVILogger << DoLog;
    }

    Instruction *NextPhi = nullptr;

    if (not Current.Unfinished) {
      // No processing in progress, proceed
      unsigned NextIndex = Current.NextIncomingIndex;
      Value *NextValue = nullptr;
      CFGEdge NewEdge;

      if (auto *Phi = dyn_cast<PHINode>(Current.Phi)) {
        NextValue = Phi->getIncomingValue(NextIndex);
        NewEdge = { Phi->getIncomingBlock(NextIndex), Phi->getParent() };
      } else if (auto *Select = dyn_cast<SelectInst>(Current.Phi)) {
        NextValue = Select->getOperand(1 + NextIndex);
        NewEdge = { Select->getParent(), nullptr };
      } else {
        revng_abort();
      }

      Edges.push_back(NewEdge);

      NextPhi = Current.Expr.buildAndAnalyzeExpression(LVI,
                                                       DT,
                                                       Edges,
                                                       NextValue,
                                                       StopAt,
                                                       SuccessorsCache);
      Current.NextIncomingIndex++;
    }

    // Don't enter in loops
    if (VisitedPhis.count(NextPhi) != 0)
      NextPhi = nullptr;

    if (NextPhi != nullptr) {
      VisitedPhis.insert(NextPhi);

      Current.Unfinished = true;

      // The last node of the Expression we just build is a phi node,
      // we have to suspend processing and proceed towards it
      PendingPhis.emplace_back(DL,
                               SE,
                               NextPhi,
                               Current.Expr.smallestRangeSize());
    } else {
      // The last node is not a phi, we're done on this incoming value of
      // the phi

      // Drop this edge from the list of edges
      Edges.pop_back();

      MaterializedValues Result;
      size_t UpperBound = Current.Expr.smallestRangeSize();
      bool IsSmallerThanUpperBound = UpperBound < Current.UpperBound;
      bool PhiDone = not IsSmallerThanUpperBound;
      if (IsSmallerThanUpperBound) {
        // Materialize the current expression
        Result = std::move(Current.Expr.materialize<MemoryOracle>(MO));

        // Reset the unfinished flag
        Current.Unfinished = false;

        range_size_t NewSize = Current.Values.size() + Result.size();
        if (Current.TooLarge or NewSize > MaxMaterializedValues) {
          Current.TooLarge = true;
          Current.Values.clear();
          IsSmallerThanUpperBound = false;
        } else {
          // Merge results in Current.Values
          Current.Values.insert(Current.Values.end(),
                                Result.begin(),
                                Result.end());
        }

        unsigned IncomingCount = 0;
        if (auto *Phi = dyn_cast<PHINode>(Current.Phi)) {
          IncomingCount = Phi->getNumIncomingValues();
        } else if (isa<SelectInst>(Current.Phi)) {
          IncomingCount = 2;
        } else {
          revng_abort();
        }

        if (Current.NextIncomingIndex == IncomingCount) {
          // We're done with this phi
          PhiDone = true;

          // Save and deduplicate the result
          Result = std::move(Current.Values);
          llvm::sort(Result);
          auto LastIt = std::unique(Result.begin(), Result.end());
          Result.erase(LastIt, Result.end());

          IsSmallerThanUpperBound = Result.size() < Current.UpperBound;
        }
      }

      if (PhiDone) {
        if (PendingPhis.size() == 1)
          return Result;

        // Pop
        PendingPhis.pop_back();

        // Inform the new top of the stack about the results
        revng_assert(PendingPhis.back().Unfinished);

        if (IsSmallerThanUpperBound)
          PendingPhis.back().Expr.setPhiValues(std::move(Result));
      }
    }
  }

  revng_abort();
}
