#ifndef REVNG_ADVANCEDVALUEINFO_H
#define REVNG_ADVANCEDVALUEINFO_H

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <set>

// LLVM includes
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/Analysis/LazyValueInfo.h"
#include "llvm/IR/ConstantRange.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/Support/raw_os_ostream.h"

// Local libraries includes
#include "revng/ADT/ConstantRangeSet.h"
#include "revng/BasicAnalyses/MaterializedValue.h"
#include "revng/Support/Debug.h"
#include "revng/Support/GraphAlgorithms.h"
#include "revng/Support/IRHelpers.h"
#include "revng/Support/MonotoneFramework.h"

struct Edge {
  llvm::BasicBlock *Start;
  llvm::BasicBlock *End;

  bool operator<(const Edge &Other) const {
    return std::tie(Start, End) < std::tie(Other.Start, Other.End);
  }

  void dump() const debug_function {
    dbg << getName(Start) << "->" << getName(End);
  }
};

/// \brief Monotone framework to collect ConstantRangeSets from LazyValueInfo
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
  : public MonotoneFramework<Analysis,
                             llvm::BasicBlock *,
                             Element,
                             ReversePostOrder,
                             llvm::SmallVector<llvm::BasicBlock *, 2>> {
private:
  using Base = MonotoneFramework<Analysis,
                                 llvm::BasicBlock *,
                                 Element,
                                 ReversePostOrder,
                                 llvm::SmallVector<llvm::BasicBlock *, 2>>;

private:
  llvm::BasicBlock *Entry;
  llvm::LazyValueInfo &LVI;
  const llvm::DominatorTree &DT;
  std::map<llvm::Instruction *, ConstantRangeSet> InstructionRanges;
  std::set<Edge> TargetEdges;
  std::set<llvm::BasicBlock *> WhiteList;

public:
  Analysis(const llvm::SmallVectorImpl<llvm::BasicBlock *> &RPOT,
           llvm::LazyValueInfo &LVI,
           const llvm::DominatorTree &DT,
           const std::vector<llvm::Instruction *> &TargetInstructions,
           const std::vector<Edge> &TargetEdges) :
    Base(RPOT),
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

    for (const Edge &E : TargetEdges) {
      this->TargetEdges.insert(E);
    }

    for (BasicBlock *BB : RPOT) {
      WhiteList.insert(BB);
    }
  }

  Element extremalValue(llvm::BasicBlock *BB) const { return Element(); }

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

  llvm::Optional<Element> handleEdge(const Element &Original,
                                     llvm::BasicBlock *Source,
                                     llvm::BasicBlock *Destination) {
    bool IsTargetEdge = TargetEdges.count({ Source, Destination }) != 0;
    return compute(Original, Source, Destination, IsTargetEdge);
  }

  llvm::SmallVector<llvm::BasicBlock *, 2>
  successors(llvm::BasicBlock *L, DefaultInterrupt<Element> &I) const {
    using namespace llvm;
    SmallVector<BasicBlock *, 2> Result;
    for (llvm::BasicBlock *Successor : make_range(succ_begin(L), succ_end(L)))
      if (WhiteList.count(Successor) != 0)
        Result.push_back(Successor);
    return Result;
  }

  size_t
  successor_size(llvm::BasicBlock *L, DefaultInterrupt<Element> &I) const {
    return successors(L, I).size();
  }

  const ConstantRangeSet &get(llvm::Instruction *I) const {
    return InstructionRanges.at(I);
  }

  void dump() const {
    for (auto &P : InstructionRanges) {
      dbg << getName(P.first) << ": ";
      P.second.dump();
      dbg << "\n";
    }
  }

private:
  llvm::Optional<Element> compute(const Element &Original,
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

      llvm::ConstantRange NewRange(BitWidth);
      if (Destination == nullptr)
        NewRange = LVI.getConstantRange(I, Source);
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

inline bool isInterestingRange(uint64_t Size, llvm::IntegerType *T) {
  return Size <= MaxMaterializedValues and Size < T->getBitMask();
}

inline bool isInterestingRange(uint64_t Size, llvm::Type *T) {
  return isInterestingRange(Size, llvm::cast<llvm::IntegerType>(T));
}

inline bool isInterestingRange(uint64_t Size, llvm::Value *V) {
  return isInterestingRange(Size, V->getType());
}

inline bool isPhiLike(llvm::Value *V) {
  return (llvm::isa<llvm::PHINode>(V) or llvm::isa<llvm::SelectInst>(V));
}

inline bool isMemory(llvm::Value *V) {
  using namespace llvm;
  return not(isa<GlobalVariable>(V) or isa<AllocaInst>(V));
}

/// \brief An operation producing a result and having a single free operand
struct Operation {
  llvm::User *V;
  unsigned FreeOperandIndex;
  ConstantRangeSet Range;
  uint64_t RangeSize;

  Operation() : V(nullptr), FreeOperandIndex(0), Range(), RangeSize(0) {}
  Operation(llvm::User *V,
            unsigned FreeOperandIndex,
            llvm::ConstantRange Range,
            uint64_t RangeSize) :
    V(V),
    FreeOperandIndex(FreeOperandIndex),
    Range(Range),
    RangeSize(RangeSize) {}

  unsigned getBitSize() const {
    using namespace llvm;
    return cast<IntegerType>(V->getType())->getBitWidth();
  }

  void dump(unsigned Indent) const debug_function {
    std::string Prefix(Indent, ' ');
    dbg << Prefix;
    V->dump();
    dbg << Prefix << "FreeOperandIndex: " << FreeOperandIndex << "\n";

    dbg << Prefix;
    Range.dump();
    dbg << "\n";

    dbg << Prefix << "RangeSize: " << RangeSize << "\n";
  }
};

/// \brief Class representing an expression on the IR
class Expression {
private:
  std::vector<Operation> OperationsStack;
  unsigned SmallestRangeIndex;
  bool PhiIsSmallest;
  MaterializedValues Values;
  bool Materialized;

public:
  using PhiEdges = std::vector<Edge>;

public:
  Expression() { reset(); }

  void reset() {
    SmallestRangeIndex = 0;
    PhiIsSmallest = false;
    Materialized = false;
    OperationsStack.clear();
    Values.clear();
  }

  void dump(unsigned Indent) const debug_function {
    std::string Prefix(Indent, ' ');

    dbg << Prefix << "OperationStack: \n";
    unsigned I = 0;
    for (const Operation &Op : OperationsStack) {
      dbg << Prefix << "  " << I;
      if (I == SmallestRangeIndex)
        dbg << " [smallest]";
      dbg << ":\n";
      Op.dump(Indent + 4);
      ++I;
    }
    dbg << "\n";

    dbg << Prefix << "PhiIsSmallest: " << PhiIsSmallest << "\n";
    dbg << Prefix << "Values: {";
    for (const MaterializedValue &Value : Values) {
      dbg << " ";
      Value.dump();
    }
    dbg << " }\n";
    dbg << Prefix << "Materialized: " << Materialized << "\n";
  }

  bool lastIsPhi() const { return isPhiLike(OperationsStack.back().V); }

  uint64_t smallestRangeSize() const {
    return OperationsStack.at(SmallestRangeIndex).RangeSize;
  }

  llvm::Value *smallestRangeValue() const {
    return OperationsStack.at(SmallestRangeIndex).V;
  }

  /// Use LVI to build an expression about \p V
  ///
  /// 1. Build a chain of single non-const-operand instructions until you find a
  ///    phi or a load from a global variable.
  /// 2. For each instruction in the chain record the number of possible values
  ///    according to LVI.
  /// 3. Iterate over the chain looking for the instruction associated with the
  ///    smallest range.
  llvm::Instruction *
  buildExpression(llvm::LazyValueInfo &LVI,
                  const llvm::DominatorTree &DT,
                  PhiEdges &Edges,
                  llvm::Value *V,
                  const std::vector<llvm::BasicBlock *> &RPOT) {
    using namespace llvm;
    Instruction *Result = nullptr;

    reset();

    User *U = cast<User>(V);
    do {
      //
      // Identify the free operand
      //
      unsigned Index = 0;
      unsigned NextIndex = 0;
      Value *Next = nullptr;

      ConstantRange Range(64);
      uint64_t RangeSize = std::numeric_limits<uint64_t>::max();
      auto *I = dyn_cast<Instruction>(U);

      if (auto *Call = dyn_cast<CallInst>(U)) {
        if (Function *Callee = Call->getCalledFunction()) {
          if (Callee->getIntrinsicID() == Intrinsic::bswap) {
            Use &FirstArg = Call->getArgOperandUse(0);
            Next = FirstArg.get();
            NextIndex = FirstArg.getOperandNo();
          }
        }
      } else if (I != nullptr) {

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

      } else if (auto *C = dyn_cast<ConstantInt>(U)) {
        RangeSize = 1;
        Range = ConstantRange(APInt(64, getLimitedValue(C)));
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

    auto IsInteresting = [](const Operation &O) -> llvm::Instruction * {
      if (O.Range.isFullSet())
        if (auto *I = dyn_cast<Instruction>(O.V))
          if (isa<IntegerType>(I->getType()))
            return I;
      return nullptr;
    };

    std::vector<Instruction *> Targets;
    for (const Operation &O : OperationsStack) {
      if (auto *I = IsInteresting(O)) {
        Targets.push_back(I);
      }
    }

    if (Targets.size() != 0) {
      SmallVector<BasicBlock *, 8> FilteredRPOT;
      BasicBlock *StartBB = Targets.back()->getParent();
      const Edge &FirstEdge = Edges.front();
      revng_assert(FirstEdge.End == nullptr);
      BasicBlock *EndBB = FirstEdge.Start;
      auto Reachable = nodesBetweenReverse(EndBB, StartBB);

      for (BasicBlock *BB : RPOT)
        if (Reachable.count(BB) != 0)
          FilteredRPOT.push_back(BB);

      DisjointRanges::Analysis DR(FilteredRPOT, LVI, DT, Targets, Edges);
      DR.initialize();
      DR.run();

      for (Operation &O : OperationsStack) {
        if (auto *I = IsInteresting(O)) {
          O.Range = DR.get(I);
          O.RangeSize = O.Range.size().getLimitedValue();
        }
      }
    }

    for (const Operation &O : OperationsStack) {
      // Get the LVI and record if it's the smallest
      if (OperationsStack[SmallestRangeIndex].RangeSize > O.RangeSize) {
        SmallestRangeIndex = &O - &*OperationsStack.begin();
      }
    }

    return Result;
  }

  /// \brief Materialize all the values in this expression
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

    uint64_t WorstCase = MaxMaterializedValues;
    if (SmallestType != nullptr)
      WorstCase = std::min(SmallestType->getBitMask(), WorstCase);

    const Operation &SmallestOperation = OperationsStack.at(SmallestRangeIndex);

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
        Entry = { It->getLimitedValue() };
        ++It;
      }

    } else {
      // The Values vector has already been initialized
      revng_assert(lastIsPhi());
    }

    revng_assert(OperationsStack.size() != 0);

    // Process one value at a time
    for (MaterializedValue &Entry : Values) {
      using CI = ConstantInt;
      using CE = ConstantExpr;

      llvm::Optional<llvm::StringRef> SymbolName;
      auto *Current = CI::get(SmallestOperation.V->getType(), Entry.value());

      OperationsStack.resize(SmallestRangeIndex);

      // Materialize the value I through the operations stack
      auto It = OperationsStack.rbegin();
      auto End = OperationsStack.rend();
      if (It != End) {
        auto Range = make_range(It, End);
        for (const Operation &Op : Range) {

          // After we get a symbol name we only track casts, additions and
          // subtractions
          auto *I = dyn_cast<Instruction>(Op.V);
          if (SymbolName
              and not(I != nullptr
                      and (I->isCast() or I->getOpcode() == Instruction::Add
                           or I->getOpcode() == Instruction::Sub))) {
            return {};
          }

          if (auto *C = dyn_cast<Constant>(Op.V)) {
            Current = C;
          } else if (auto *C = dyn_cast<Constant>(Op.V)) {
            revng_assert(Op.V->getNumOperands() == 1);
            Current = cast<Constant>(C->getOperand(0));
          } else if (auto *Load = dyn_cast<LoadInst>(Op.V)) {
            revng_assert(isMemory(skipCasts(Load->getPointerOperand())));

            MaterializedValue Loaded = MO.load(Current);
            if (not Loaded.isValid()) {
              // Couldn't read memory, bail out
              return {};
            }

            if (Loaded.hasSymbol())
              SymbolName = Loaded.symbolName();

            Type *LoadedType = Load->getType();
            if (LoadedType->isPointerTy()) {
              auto *M = Load->getParent()->getParent()->getParent();
              const DataLayout &DL = M->getDataLayout();
              LLVMContext &C = M->getContext();
              Current = CI::get(DL.getIntPtrType(C), Loaded.value());
              Current = CE::getIntToPtr(Current, LoadedType);
            } else {
              Current = CI::get(cast<IntegerType>(LoadedType), Loaded.value());
            }

          } else if (auto *Call = dyn_cast<CallInst>(Op.V)) {
            Function *Callee = Call->getCalledFunction();
            revng_assert(Callee != nullptr
                         && Callee->getIntrinsicID() == Intrinsic::bswap);

            uint64_t Value = getLimitedValue(cast<ConstantInt>(Current));

            Type *T = Call->getType();
            if (T->isIntegerTy(16))
              Value = ByteSwap_16(Value);
            else if (T->isIntegerTy(32))
              Value = ByteSwap_32(Value);
            else if (T->isIntegerTy(64))
              Value = ByteSwap_64(Value);
            else
              revng_unreachable("Unexpected type");

            Current = ConstantInt::get(T, Value);
          } else if (I != nullptr) {

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

            Current = ConstantFoldInstOperands(I, Operands, MO.getDataLayout());
            revng_assert(Current != nullptr);

          } else {
            revng_abort();
          }
        }
      }

      uint64_t Value = 0;
      if (not Current->isNullValue())
        Value = getLimitedValue(skipCasts(Current));

      if (SymbolName)
        Entry = { *SymbolName, Value };
      else
        Entry = { Value };
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

/// \brief Context for processing a phi node
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
  uint64_t UpperBound;

  /// Did we exceed MaxMaterializedValues?
  bool TooLarge;

public:
  PhiProcess(llvm::Instruction *Phi, uint64_t UpperBound) :
    Phi(Phi),
    NextIncomingIndex(0),
    Unfinished(false),
    UpperBound(UpperBound),
    TooLarge(false) {

    revng_assert(isPhiLike(Phi));
  }

  void dump() const debug_function {
    dbg << "Phi: ";
    Phi->dump();

    dbg << "NextIncomingIndex: " << NextIncomingIndex << "\n";

    dbg << "Values: {";
    for (const MaterializedValue &Value : Values) {
      dbg << " ";
      Value.dump();
    }
    dbg << " }\n";

    dbg << "Expr:\n";
    Expr.dump(2);

    dbg << "Unfinished: " << Unfinished << "\n";
    dbg << "UpperBound: " << UpperBound << "\n";
    dbg << "TooLarge: " << TooLarge << "\n";
  }
};

/// \brief Analyis to associate to each value a ConstantRangeSet using
///        LazyValueInfo
///
/// \tparam MemoryOracle the type of the class used to produce obtain the result
///         of memory accesses from constant addresses.
template<typename MemoryOracle>
class AdvancedValueInfo {
private:
  llvm::LazyValueInfo &LVI;
  const llvm::DominatorTree &DT;
  MemoryOracle &MO;
  const std::vector<llvm::BasicBlock *> &RPOT;

public:
  AdvancedValueInfo(llvm::LazyValueInfo &LVI,
                    const llvm::DominatorTree &DT,
                    MemoryOracle &MO,
                    const std::vector<llvm::BasicBlock *> &RPOT) :
    LVI(LVI),
    DT(DT),
    MO(MO),
    RPOT(RPOT) {}

  MaterializedValues explore(llvm::BasicBlock *BB, llvm::Value *V);
};

template<class MemoryOracle>
MaterializedValues
AdvancedValueInfo<MemoryOracle>::explore(llvm::BasicBlock *BB, llvm::Value *V) {
  using namespace llvm;

  // Create a fake Phi for the initial entry
  PHINode *FakePhi = PHINode::Create(V->getType(), 1);
  FakePhi->addIncoming(V, BB);

  struct DeleteValueOnReturn {
    Instruction *I;
    ~DeleteValueOnReturn() {
      if (I->getParent() != nullptr)
        I->eraseFromParent();
      else
        I->deleteValue();
    }
  };
  DeleteValueOnReturn X{ FakePhi };

  std::set<Instruction *> VisitedPhis;
  std::vector<PhiProcess> PendingPhis{
    { FakePhi, std::numeric_limits<uint64_t>::max() }
  };
  Expression::PhiEdges Edges;

  while (true) {
    PhiProcess &Current = PendingPhis.back();

    Instruction *NextPhi = nullptr;

    if (not Current.Unfinished) {
      // No processing in progress, proceed
      uint64_t NextIndex = Current.NextIncomingIndex;
      Value *NextValue = nullptr;
      Edge NewEdge;

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

      NextPhi = Current.Expr.buildExpression(LVI, DT, Edges, NextValue, RPOT);
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
      PendingPhis.emplace_back(NextPhi, Current.Expr.smallestRangeSize());
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

        uint64_t NewSize = Current.Values.size() + Result.size();
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
        } else if (auto *Select = dyn_cast<SelectInst>(Current.Phi)) {
          IncomingCount = 2;
        } else {
          revng_abort();
        }

        if (Current.NextIncomingIndex == IncomingCount) {
          // We're done with this phi
          PhiDone = true;

          // Save and deduplicate the result
          Result = std::move(Current.Values);
          std::sort(Result.begin(), Result.end());
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

#endif // REVNG_ADVANCEDVALUEINFO_H
