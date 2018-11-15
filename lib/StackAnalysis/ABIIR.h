#ifndef ABIIR_H
#define ABIIR_H

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <stack>

// LLVM includes
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

// Local includes
#include "ASSlot.h"
#include "FunctionABI.h"

namespace StackAnalysis {

/// \brief Instruction of the ABI IR
class ABIIRInstruction {
public:
  enum Opcode { Load, Store, DirectCall, IndirectCall };

private:
  /// Instruction opcode
  Opcode O;

  /// Load/store target address
  const ASSlot Target;

  //
  // Call-only fields
  //

  /// Reference to the function call
  FunctionCall Call;

  /// Result of the ABI analysis for the callee
  ///
  /// \note FunctionABI can be quite large, create an instance only if needed.
  std::unique_ptr<FunctionABI> ABI;

  /// Set of caller stack slots written by the callee
  std::set<int32_t> WrittenStackSlots;

private:
  ABIIRInstruction(Opcode O,
                   FunctionCall Call,
                   FunctionABI ABI,
                   std::set<int32_t> WrittenStackSlots) :
    O(O),
    Target(ASSlot::invalid()),
    Call(Call),
    ABI(new FunctionABI(std::move(ABI))),
    WrittenStackSlots(std::move(WrittenStackSlots)) {
    revng_assert(O == DirectCall);
  }

  ABIIRInstruction(Opcode O, ASSlot Target) :
    O(O),
    Target(Target),
    ABI(),
    WrittenStackSlots() {
    revng_assert(O == Load || O == Store);
  }

  ABIIRInstruction(Opcode O, FunctionCall Call) :
    O(O),
    Target(ASSlot::invalid()),
    Call(Call),
    ABI(),
    WrittenStackSlots() {
    revng_assert(O == IndirectCall);
  }

public:
  static ABIIRInstruction createLoad(const ASSlot Target) {
    return ABIIRInstruction(Load, Target);
  }

  static ABIIRInstruction createStore(const ASSlot Target) {
    return ABIIRInstruction(Store, Target);
  }

  static ABIIRInstruction
  createDirectCall(FunctionCall Call,
                   FunctionABI ABI,
                   std::set<int32_t> WrittenStackSlots) {
    return ABIIRInstruction(DirectCall,
                            Call,
                            std::move(ABI),
                            std::move(WrittenStackSlots));
  }

  static ABIIRInstruction createIndirectCall(FunctionCall Call) {
    return ABIIRInstruction(IndirectCall, Call);
  }

public:
  Opcode opcode() const { return O; }

  bool isCall() const { return O == DirectCall or O == IndirectCall; }

  bool isStore() const { return O == Store; }

  const ASSlot target() const {
    revng_assert(O == Load || O == Store);
    return Target;
  }

  const FunctionABI &abi() const {
    revng_assert(O == DirectCall);
    return *ABI;
  }

  const std::set<int32_t> &stackArguments() const {
    revng_assert(O == DirectCall);
    return WrittenStackSlots;
  }

  FunctionCall call() const {
    revng_assert(isCall());
    revng_assert(Call.callInstruction() != nullptr);
    return Call;
  }

  void dump(const llvm::Module *M) const debug_function { dump(dbg, M); }

  template<typename T>
  void dump(T &Output, const llvm::Module *M) const {
    switch (O) {
    case Load:
      Output << "Load from ";
      target().dump(M, Output);
      break;
    case Store:
      Output << "Store to ";
      target().dump(M, Output);
      break;
    case DirectCall:
      Output << "DirectCall to " << getName(call().callee());
      Output << " from " << getName(call().callInstruction());
      break;
    case IndirectCall:
      Output << "IndirectCall from " << getName(call().callInstruction());
      break;
    }
  }
};

/// \brief Basic block of the ABI IR, a container of ABIIRInstructions
class ABIIRBasicBlock {
  // The ABIFunction class is our friend so it can finalize us
  friend class ABIFunction;

public:
  using links_container = llvm::SmallVector<ABIIRBasicBlock *, 2>;
  using links_iterator = typename links_container::iterator;
  using links_const_iterator = typename links_container::const_iterator;
  using links_range = llvm::iterator_range<links_iterator>;
  using links_const_range = llvm::iterator_range<links_const_iterator>;

  using container = std::vector<ABIIRInstruction>;

  using iterator = typename container::iterator;
  using const_iterator = typename container::const_iterator;

  using reverse_iterator = typename container::reverse_iterator;
  using const_reverse_iterator = typename container::const_reverse_iterator;

  using range = llvm::iterator_range<iterator>;
  using const_range = llvm::iterator_range<const_iterator>;

  using reverse_range = llvm::iterator_range<reverse_iterator>;
  using const_reverse_range = llvm::iterator_range<const_reverse_iterator>;

private:
  /// The instructions contained in this basic block
  std::vector<ABIIRInstruction> Instructions;

  /// List of successors
  links_container Successors;

  /// List of predecessors
  ///
  /// \note This field is initialized only after ABIFunction::finalize is called
  links_container Predecessors;

  /// Reference to the corresponding basic block
  llvm::BasicBlock *BB;

  /// Flag to identify return basic blocks
  bool IsReturn;

public:
  ABIIRBasicBlock(llvm::BasicBlock *BB) : BB(BB), IsReturn(false) {}

public:
  /// \brief Purge basic block content
  void clear() {
    revng_assert(Predecessors.empty());
    Instructions.clear();
    Successors.clear();
    IsReturn = false;
  }

  bool isReturn() const { return IsReturn; }
  void setReturn() { IsReturn = true; }

  void append(ABIIRInstruction I) { Instructions.push_back(std::move(I)); }

  void addSuccessor(ABIIRBasicBlock *Successor) {
    Successors.push_back(Successor);
  }

  size_t successor_size() const { return Successors.size(); }
  links_const_range successors() const {
    return llvm::make_range(Successors.begin(), Successors.end());
  }
  links_range successors() {
    return llvm::make_range(Successors.begin(), Successors.end());
  }

  size_t predecessor_size() const { return Predecessors.size(); }
  links_const_range predecessors() const {
    return llvm::make_range(Predecessors.begin(), Predecessors.end());
  }

  template<bool Forward>
  size_t next_size() const {
    return Forward ? successor_size() : predecessor_size();
  }
  template<bool Forward>
  links_const_range next() const {
    return Forward ? successors() : predecessors();
  }

  size_t size() const { return Instructions.size(); }

  iterator begin() { return Instructions.begin(); }
  iterator end() { return Instructions.end(); }
  const_iterator begin() const { return Instructions.begin(); }
  const_iterator end() const { return Instructions.end(); }

  reverse_iterator rbegin() { return Instructions.rbegin(); }
  reverse_iterator rend() { return Instructions.rend(); }
  const_reverse_iterator rbegin() const { return Instructions.rbegin(); }
  const_reverse_iterator rend() const { return Instructions.rend(); }

  llvm::BasicBlock *basicBlock() const { return BB; }
  llvm::StringRef getName() const { return ::getName(BB); }

  void
  dump(const llvm::Module *M, const char *Prefix = "") const debug_function {
    dump(dbg, M, Prefix);
  }

  template<typename T>
  void dump(T &Output, const llvm::Module *M, const char *Prefix = "") const {

    Output << Prefix << "From basic block " << ::getName(BB);
    if (IsReturn)
      Output << " [IsReturn]";
    Output << "\n";

    if (not Predecessors.empty()) {
      Output << Prefix << "Predecessors:\n";
      for (const ABIIRBasicBlock *Predecessor : Predecessors) {
        Output << Prefix << "  " << ::getName(Predecessor->basicBlock())
               << "\n";
      }
      Output << Prefix << "\n";
    }

    Output << Prefix << "Instructions:\n";
    for (const ABIIRInstruction &I : Instructions) {
      Output << Prefix << "  ";
      I.dump(Output, M);
      Output << "\n";
    }
    Output << "\n";

    if (not Successors.empty()) {
      Output << Prefix << "Successors:\n";
      for (const ABIIRBasicBlock *Successor : Successors)
        Output << Prefix << "  " << ::getName(Successor->basicBlock()) << "\n";
      Output << Prefix << "\n";
    }
  }
};

/// \brief The ABI IR, a container of ABIIRBasicBlocks
class ABIFunction {
public:
  template<typename K, typename V>
  using VectorOfPairs = std::vector<std::pair<K, V>>;
  using calls_container = VectorOfPairs<ABIIRBasicBlock *, ABIIRInstruction *>;

  using calls_iterator = calls_container::iterator;
  using calls_range = llvm::iterator_range<calls_iterator>;

  using calls_const_iterator = calls_container::const_iterator;
  using calls_const_range = llvm::iterator_range<calls_const_iterator>;

  using returns_container = std::vector<ABIIRBasicBlock *>;
  using returns_iterator = returns_container::iterator;
  using returns_range = llvm::iterator_range<returns_iterator>;

  using returns_const_iterator = returns_container::const_iterator;
  using returns_const_range = llvm::iterator_range<returns_const_iterator>;

private:
  /// Storage for ABI IR basic blocks, associated to their original counterpart
  ///
  /// \note Don't move after Entry
  std::map<llvm::BasicBlock *, ABIIRBasicBlock> BBMap;

  /// Pointer to the entry basic block of this function
  llvm::BasicBlock *Entry;
  ABIIRBasicBlock *IREntry;

  /// Vector of all the function calls in this function
  calls_container Calls;

  /// Vector of all the return basic blocks
  returns_container FinalBBs;

public:
  ABIFunction(llvm::BasicBlock *Entry) :
    Entry(Entry),
    IREntry(&BBMap.emplace(Entry, ABIIRBasicBlock(Entry)).first->second) {}

  ABIIRBasicBlock *entry() const { return IREntry; }

  size_t size() const { return BBMap.size(); }

  /// \brief Purge all the data in this IR
  void reset() {
    BBMap.clear();
    Calls.clear();
    FinalBBs.clear();
    IREntry = &BBMap.emplace(Entry, ABIIRBasicBlock(Entry)).first->second;
  }

  /// \brief Finalize the IR after initially populating it
  ///
  /// This method basically populates the backward links of the CFG, identifies
  /// all the function calls and ensures the entry basic block has no inbound
  /// edges.
  void finalize();

  /// \brief Identify calls leading to contradition
  std::set<FunctionCall> incoherentCalls();

  std::set<int32_t> writtenRegisters() const;

  ABIIRBasicBlock &get(llvm::BasicBlock *BB) {
    auto It = BBMap.find(BB);
    if (It != BBMap.end())
      return It->second;

    return BBMap.emplace(BB, ABIIRBasicBlock(BB)).first->second;
  }

  const ABIIRBasicBlock &get(llvm::BasicBlock *BB) const {
    auto It = BBMap.find(BB);
    revng_assert(It != BBMap.end());
    return It->second;
  }

  calls_const_range calls() const {
    return llvm::make_range(Calls.begin(), Calls.end());
  }

  size_t calls_size() const { return Calls.size(); }

  returns_const_range finals() const {
    return llvm::make_range(FinalBBs.begin(), FinalBBs.end());
  }

  size_t finals_size() const { return FinalBBs.size(); }

  bool verify() const debug_function;

  /// \brief Dump a GraphViz file on stdout representing this function
  void dumpDot() const debug_function;

  void dump(const llvm::Module *M) const debug_function { dump(dbg, M); }

  template<typename T>
  void dump(T &Output, const llvm::Module *M) const {
    std::set<const ABIIRBasicBlock *> Visited;
    std::set<const ABIIRBasicBlock *> Entries;

    for (auto &P : BBMap)
      if (P.second.predecessor_size() == 0)
        Entries.insert(&P.second);

    for (const ABIIRBasicBlock *BB : Entries) {
      if (Visited.count(BB) != 0)
        continue;

      std::stack<const ABIIRBasicBlock *> WorkList;
      WorkList.push(BB);
      while (!WorkList.empty()) {
        const ABIIRBasicBlock *Current = WorkList.top();
        WorkList.pop();

        Visited.insert(Current);

        Current->dump(Output, M, "  ");

        for (const ABIIRBasicBlock *Successor : Current->successors())
          if (Visited.count(Successor) == 0)
            WorkList.push(Successor);
      }
    }
  }
};

/// \brief Identify calls leading to contradition
///
/// \note This is implemented in incoherentcallsanalysis.cpp
std::set<FunctionCall>
computeIncoherentCalls(ABIIRBasicBlock *Entry,
                       std::vector<ABIIRBasicBlock *> &Extremals);

template<typename T, bool Forward>
inline T instructionRange(ABIIRBasicBlock *BB);

template<>
inline ABIIRBasicBlock::range
instructionRange<ABIIRBasicBlock::range, true>(ABIIRBasicBlock *BB) {
  ABIIRBasicBlock::iterator InstructionIt = BB->begin();
  return llvm::make_range(InstructionIt, BB->end());
}

template<>
inline ABIIRBasicBlock::reverse_range
instructionRange<ABIIRBasicBlock::reverse_range, false>(ABIIRBasicBlock *BB) {
  ABIIRBasicBlock::reverse_iterator InstructionIt = BB->rbegin();
  return llvm::make_range(InstructionIt, BB->rend());
}

} // namespace StackAnalysis

// Provide graph traits for usage with, e.g., llvm::ReversePostOrderTraversal
namespace llvm {

template<>
struct GraphTraits<StackAnalysis::ABIIRBasicBlock *> {
  using NodeRef = StackAnalysis::ABIIRBasicBlock *;
  using ChildIteratorType = StackAnalysis::ABIIRBasicBlock::links_iterator;

  static NodeRef getEntryNode(StackAnalysis::ABIIRBasicBlock *BB) { return BB; }

  static inline ChildIteratorType
  child_begin(StackAnalysis::ABIIRBasicBlock *N) {
    return N->successors().begin();
  }

  static inline ChildIteratorType child_end(StackAnalysis::ABIIRBasicBlock *N) {
    return N->successors().end();
  }
};

} // namespace llvm

#endif // ABIIR_H
