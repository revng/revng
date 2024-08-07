/// \file DataFlowGraph.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/Support/GraphWriter.h"

#include "revng/ValueMaterializer/DataFlowGraph.h"

using namespace llvm;

using range_size_t = uint64_t;

Logger<> &Log = ValueMaterializerLogger;

const range_size_t MaxMaterializedValues = (1 << 16);

template<typename Range>
using RangeValueType = std::decay_t<decltype(*std::declval<Range>().begin())>;

template<typename Range>
auto allCombinations(SmallVector<Range, 2> Ranges)
  -> cppcoro::generator<SmallVector<RangeValueType<Range>, 2>> {
  using iterator = std::decay_t<decltype(Ranges[0].begin())>;
  using value_type = std::decay_t<decltype(*Ranges[0].begin())>;
  using ResultType = SmallVector<value_type, 2>;

  struct Entry {
    iterator Current;
    iterator Begin;
    iterator End;
  };
  SmallVector<Entry, 2> Entries;

  for (Range &R : Ranges) {
    auto Begin = R.begin();
    auto End = R.end();
    if (Begin == End)
      co_return;
    Entries.push_back({ Begin, Begin, End });
  }

  bool Done = false;
  while (not Done) {
    ResultType Result;

    // Boolean to indicate whether the current iterator has reached the end and
    // we need therefore to increment the next iterator in the list
    bool BumpNextIterator = true;
    for (Entry &Entry : Entries) {
      Result.push_back(*Entry.Current);

      if (BumpNextIterator) {
        // The last iterator has reached the end, bump me
        ++Entry.Current;
        BumpNextIterator = false;
      }

      if (Entry.Current == Entry.End) {
        // The current iterator has reached the end, mark the next iterator to
        // be incremented
        Entry.Current = Entry.Begin;
        BumpNextIterator = true;
      }
    }

    Done = BumpNextIterator;

    co_yield Result;
  }
}

std::string aviFormatter(const APInt &Value) {
  if (Value.getBitWidth() == 128) {
    auto AsAddress = MetaAddress::decomposeIntegerPC(Value);
    if (AsAddress.isValid())
      return AsAddress.toString();
  }

  if (Value.isAllOnesValue()) {
    return "max";
  } else {
    SmallString<32> Result;
    Result.push_back('0');
    Result.push_back('x');
    Value.toStringUnsigned(Result, 16);
    return Result.str().str();
  }
}

void DataFlowGraph::removeCycles() {

  DenseSet<Node *> SCCNodes;

  // Find all the nodes member of a cyclic SCC
  auto End = scc_end(this);
  for (auto It = scc_begin(this); It != End; ++It)
    if (It.hasCycle())
      for (const auto &Node : *It)
        SCCNodes.insert(Node);

  // Purge all successors
  for (Node *SCCNode : SCCNodes)
    while (SCCNode->hasSuccessors())
      SCCNode->removeSuccessor(SCCNode->successors().begin());
}

void DataFlowGraph::purgeUnreachable() {
  using namespace llvm;

  DenseSet<Node *> Reachable;

  for (Node *N : depth_first(this))
    Reachable.insert(N);

  erase_if(this->Nodes,
           [&](auto &N) { return not Reachable.contains(N.get()); });
}

void DataFlowGraph::enforceLimits(Limits TheLimits) {
  using namespace llvm;

  std::map<Node *, Limits> LimitsMap;
  LimitsMap[getEntryNode()] = TheLimits;
  SmallPtrSet<Node *, 8> ToPurge;

  for (Node *N : inverse_post_order(this)) {
    auto NewLimit = LimitsMap.at(N);
    Value *V = N->Value;

    if (isa<LoadInst>(V)) {
      if (not NewLimit.consumeLoad())
        ToPurge.insert(N);
    } else if (isPhiLike(V)) {
      if (not NewLimit.consumePhiLike())
        ToPurge.insert(N);
    }

    LimitsMap[N] = NewLimit;
  }

  for (Node *N : ToPurge)
    N->clearSuccessors();
}

RecursiveCoroutine<DataFlowGraph::Node *>
DataFlowGraph::processValue(Value *V, Limits Limits) {
  using namespace llvm;

  auto It = NodeMap.find(V);
  if (It != NodeMap.end())
    rc_return It->second;

  // Create node
  Node *NewNode = addNode(V);
  NodeMap[V] = NewNode;

  // Note: in the following, limits are enforced opportunistically to avoid
  //       materializing the full data flow graph and then prune most of it.
  //       However, in order to ensure the limits are properly enforced use
  //       enforceLimits, which performs a proper visit and drops all the nodes
  //       in excess.

  // Handle Phi-like instructions first, then regular Users
  if (auto *Phi = dyn_cast<PHINode>(V)) {
    if (Limits.consumePhiLike())
      for (Value *Incoming : Phi->incoming_values())
        NewNode->addSuccessor(rc_recur processValue(Incoming, Limits));
  } else if (auto *Select = dyn_cast<SelectInst>(V)) {
    if (Limits.consumePhiLike()) {
      auto *True = Select->getTrueValue();
      NewNode->addSuccessor(rc_recur processValue(True, Limits));
      auto *False = Select->getFalseValue();
      NewNode->addSuccessor(rc_recur processValue(False, Limits));
    }
  } else if (isa<GlobalVariable>(V) or isa<Argument>(V)) {
    // Ignore operands
  } else if (auto *Call = dyn_cast<CallBase>(V)) {
    // Stop at calls, except for bswap
    if (auto *Callee = getCalledFunction(Call)) {
      if (Callee->getIntrinsicID() == Intrinsic::bswap
          or Callee->getIntrinsicID() == Intrinsic::fshl) {
        auto *Operand = Call->getArgOperand(0);
        NewNode->addSuccessor(rc_recur processValue(Operand, Limits));
      }
    }
  } else if (auto *U = dyn_cast<User>(V)) {

    if (not isa<LoadInst>(U) or Limits.consumeLoad()) {
      for (Value *Operand : U->operands())
        NewNode->addSuccessor(rc_recur processValue(Operand, Limits));
    }

  } else {
    revng_abort("Unexpected value");
  }

  rc_return NewNode;
}

std::string DataFlowNode::valueToString() const {
  if (Value->hasName()) {
    return Value->getName().str();
  } else if (auto *CI = dyn_cast<ConstantInt>(Value)) {
    return aviFormatter(CI->getValue());
  } else if (auto *CE = dyn_cast<ConstantPointerNull>(Value)) {
    return "nullptr";
  } else if (auto *CE = dyn_cast<ConstantExpr>(Value)) {
    return CE->getOpcodeName();
  } else if (auto *I = dyn_cast<Instruction>(Value)) {
    return (Twine(getName(Value)) + " (" + Twine(I->getOpcodeName()) + ")")
      .str();
  } else {
    revng_abort();
  }
}

static MaterializedValues materialize(const ConstantRangeSet &Range) {
  MaterializedValues Result;
  for (const APInt &Value : Range)
    Result.push_back(MaterializedValue::fromConstant(Value));
  return Result;
}

static MaterializedValue
materialize(MemoryOracle &MO,
            Value *Operation,
            const SmallVector<MaterializedValue, 2> &Operands) {
  using namespace llvm;

  auto *UserOperation = cast<User>(Operation);
  if (isa<GlobalVariable>(Operation)) {
    revng_assert(Operands.size() == 0);
  } else if (auto *Call = dyn_cast<CallBase>(Operation)) {
    auto *Callee = getCalledFunction(Call);
    revng_assert(Callee != nullptr);
    revng_assert(Callee->getIntrinsicID() == Intrinsic::bswap
                 or Callee->getIntrinsicID() == Intrinsic::fshl);
  } else {
    revng_assert(UserOperation->getNumOperands() == Operands.size());
  }

  if (auto *C = dyn_cast<ConstantInt>(Operation)) {
    // Integer constant
    return MaterializedValue::fromConstant(C);
  } else if (auto *C = dyn_cast<ConstantExpr>(Operation)) {
    // Handle casts
    revng_assert(C->getNumOperands() == 1 and C->isCast());
    return Operands[0];
  } else if (auto *Load = dyn_cast<LoadInst>(Operation)) {
    // Handle loads

    // Check we're accessing memory
    if (not isMemory(skipCasts(Load->getPointerOperand())))
      return MaterializedValue::invalid();

    unsigned LoadSize = Load->getType()->getIntegerBitWidth() / 8;
    return Operands[0].load(MO, LoadSize);
  } else if (auto *Call = dyn_cast<CallInst>(Operation)) {
    // Handle bswap and fshl.
    Function *Callee = getCalledFunction(Call);
    revng_assert(Callee != nullptr
                 and (Callee->getIntrinsicID() == Intrinsic::bswap
                      or Callee->getIntrinsicID() == Intrinsic::fshl));

    return Operands[0].byteSwap();
  } else if (auto *Instruction = dyn_cast<llvm::Instruction>(Operation)) {
    // Regular instruction, constant fold it
    return MaterializedValue::apply(Instruction, Operands);
  } else if (isa<GlobalVariable>(Operation)) {
    return MaterializedValue::invalid();
  } else {
    revng_abort();
  }
}

RecursiveCoroutine<std::optional<MaterializedValues>>
DataFlowGraph::materializeImpl(DataFlowGraph::Node *N,
                               MemoryOracle &MO,
                               NodeValuesMap &Results) const {
  using namespace llvm;
  using Node = DataFlowGraph::Node;

  auto It = Results.find(N);
  if (It != Results.end())
    rc_return It->second;

  revng_log(Log, "Materializing " << N->valueToString());
  LoggerIndent<> Indent(Log);

  // Prevent attempting to materialize more than MaxMaterializedValues
  if (N->SizeLowerBound > MaxMaterializedValues) {
    revng_log(Log,
              "Too many values to materialize: " << N->SizeLowerBound
                                                 << ". Bailing out.");
    rc_return std::nullopt;
  }

  if (N->UseOracle)
    rc_return{ { ::materialize(*N->OracleRange), {} } };

  MaterializedValues Result;

  if (isPhiLike(N->Value)) {

    revng_log(Log, "It's a phi-like: merge input ranges.");

    // For phi-likes, merge all the results of the successors
    for (Node *Successor : N->successors()) {
      auto MaybeMaterialized = rc_recur materializeImpl(Successor, MO, Results);
      if (not MaybeMaterialized)
        rc_return std::nullopt;

      for (MaterializedValue &Value : *MaybeMaterialized)
        Result.push_back(Value);
    }

  } else {
    // Regular instruction: constant fold with all the possible operands
    // combinations

    SmallVector<MaterializedValues, 2> MaterializedValuesVector;
    SmallVector<iterator_range<MaterializedValues::iterator>, 2> Ranges;

    // Ensure we don't reallocate
    MaterializedValuesVector.reserve(N->successorCount());

    // Build vector of ranges
    for (Node *Successor : N->successors()) {
      auto MaybeMaterialized = rc_recur materializeImpl(Successor, MO, Results);

      if (not MaybeMaterialized)
        rc_return std::nullopt;

      MaterializedValuesVector.push_back(*MaybeMaterialized);
      Ranges.push_back(make_range(MaterializedValuesVector.back().begin(),
                                  MaterializedValuesVector.back().end()));
    }

    OverflowSafeInt<uint64_t> ToMaterialize = 1;
    for (auto &MaterializedValues : MaterializedValuesVector)
      ToMaterialize *= MaterializedValues.size();

    if (not ToMaterialize or *ToMaterialize > MaxMaterializedValues) {
      if (Log.isEnabled()) {
        Log << "Too many combinations to materialize:\n";
        unsigned OperandIndex = 0;
        for (auto &MaterializedValues : MaterializedValuesVector) {
          Log << "  Operand #" << OperandIndex << ": "
              << MaterializedValues.size() << "\n";
          ++OperandIndex;
        }

        Log << "Bailing out." << DoLog;
      }

      rc_return{};
    }

    for (SmallVector<MaterializedValue, 2> &Operands :
         allCombinations(Ranges)) {

      auto Value = ::materialize(MO, N->Value, Operands);

      if (not Value.isValid())
        rc_return std::nullopt;

      Result.push_back(Value);
    }
  }

  //
  // Deduplicate results
  //
  uint64_t PreDeduplicationSize = Result.size();
  sort(Result);
  auto LastIt = std::unique(Result.begin(), Result.end());
  Result.erase(LastIt, Result.end());

  if (Result.size() != PreDeduplicationSize) {
    revng_log(Log,
              (Result.size() - PreDeduplicationSize) << " values were "
                                                        "duplicates");
  }

  if (Result.size() > MaxMaterializedValues) {
    revng_log(Log,
              "Too many values materialized: " << Result.size()
                                               << ". Bailing out.");
    rc_return std::nullopt;
  } else {
    revng_log(Log, Result.size() << " values have been materialized");
  }

  // Refine results using the oracle's results
  if (N->OracleRange.has_value()) {
    size_t PreFilteringSize = Result.size();

    auto NotInOracleRange = [&N](const MaterializedValue &Value) {
      auto Range = ConstantRangeSet(Value.value());
      return (not Value.hasSymbol() and not N->OracleRange->contains(Range));
    };
    erase_if(Result, NotInOracleRange);

    if (Log.isEnabled() and Result.size() != PreFilteringSize) {
      Log << "We removed " << (PreFilteringSize - Result.size()) << " out of "
          << PreFilteringSize
          << " thanks to a constraint provided by the "
             "oracle: ";
      N->OracleRange->dump(Log);
      Log << DoLog;
    }
  }

  rc_return Result;
}

using DFG = DataFlowGraph;

std::string DOTGraphTraits<const DFG *>::getGraphProperties(const DFG *) {
  return "  node [shape=box];\n  rankdir = BT;\n";
}

std::string DOTGraphTraits<const DFG *>::getNodeLabel(const DFG::Node *Node,
                                                      const DFG *Graph) {
  std::string Result;
  {
    raw_string_ostream Stream(Result);
    Node->dump(Stream);
  }
  replaceAll(Result, "\n", "\\l");
  return Result;
}

std::string
DOTGraphTraits<const DFG *>::getNodeAttributes(const DFG::Node *Node,
                                               const DFG *Graph) {
  std::string Result;

  if (isPhiLike(Node->Value))
    Result += "style=dashed";

  if (Node->UseOracle) {
    if (Result.size() != 0)
      Result += ",";
    Result += "color=red";
  }

  return Result;
}

void DataFlowGraph::dump() const {
  WriteGraph(this, "dfg");
}
