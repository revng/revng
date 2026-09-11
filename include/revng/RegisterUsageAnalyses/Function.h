#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <limits>
#include <string>

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/GlobalVariable.h"

#include "revng/ADT/GenericGraph.h"
#include "revng/Support/Generator.h"

namespace rua {

namespace OperationType {

enum Values : uint8_t {
  Invalid,
  Read,
  Write,
  Clobber
};

inline llvm::StringRef getName(Values V) {
  switch (V) {
  case Invalid:
    return "Invalid";
  case Read:
    return "Read";
  case Write:
    return "Write";
  case Clobber:
    return "Clobber";
  default:
    revng_abort();
    break;
  }
}

} // namespace OperationType

class Operation {
public:
  OperationType::Values Type = OperationType::Invalid;

  /// Index of the CSV this operation targets, assigned by Function::csvIndex.
  /// Default-constructed operations are never analyzed (Type == Invalid).
  uint8_t Target = 0;
};

static_assert(sizeof(Operation) == 2);

struct Block {
public:
  using OperationsVector = llvm::SmallVector<Operation, 8>;
  using iterator = OperationsVector::iterator;

public:
  /// \note Only for debugging purposes
  std::string Label;
  OperationsVector Operations;

public:
  Block() = default;

public:
  std::string label() const {
    if (Label.size() > 0)
      return Label;
    else
      return ("0x" + llvm::Twine::utohexstr(reinterpret_cast<intptr_t>(this)))
        .str();
  }

public:
  auto begin() const { return Operations.begin(); }
  auto end() const { return Operations.end(); }

  auto begin() { return Operations.begin(); }
  auto end() { return Operations.end(); }

  auto rbegin() const { return Operations.rbegin(); }
  auto rend() const { return Operations.rend(); }

  auto rbegin() { return Operations.rbegin(); }
  auto rend() { return Operations.rend(); }
};

using BlockNode = BidirectionalNode<Block>;

class Function : public GenericGraph<BlockNode> {
private:
  llvm::DenseMap<uint8_t, llvm::GlobalVariable *> IndexToCSV;
  llvm::DenseMap<llvm::GlobalVariable *, uint8_t> CSVToIndex;

public:
  Function() = default;

public:
  uint8_t csvIndex(llvm::GlobalVariable *CSV) {
    auto It = CSVToIndex.find(CSV);
    if (It != CSVToIndex.end())
      return It->second;

    auto CSVCount = CSVToIndex.size();

    // The index is stored in Operation::Target (a uint8_t), so a function
    // cannot index more CSVs than that type can hold.
    revng_assert(CSVCount <= std::numeric_limits<uint8_t>::max());

    CSVToIndex[CSV] = CSVCount;
    IndexToCSV[CSVCount] = CSV;

    revng_assert(CSVToIndex.size() == IndexToCSV.size());
    revng_assert(CSVToIndex.size() == 1 + CSVCount);

    return CSVCount;
  }

  llvm::GlobalVariable *csvByIndex(uint8_t Index) const {
    auto It = IndexToCSV.find(Index);
    revng_assert(It != IndexToCSV.end());
    return It->second;
  }

  uint8_t csvCount() const { return IndexToCSV.size(); }

  cppcoro::generator<llvm::GlobalVariable *>
  csvsInSet(const llvm::BitVector &Set) {
    for (unsigned Index : Set.set_bits()) {
      co_yield csvByIndex(Index);
    }
  }

  std::string toString(const Operation &Operation) const {
    auto *CSV = csvByIndex(Operation.Target);
    return (OperationType::getName(Operation.Type).str() + "("
            + CSV->getName().str() + ")");
  }

public:
  void simplify(const llvm::SmallPtrSetImpl<Function::Node *> &ToPreserve) {
    llvm::erase_if(Nodes, [&ToPreserve](std::unique_ptr<Node> &Owning) -> bool {
      auto *N = Owning.get();

      // Check preconditions
      if (N->predecessorCount() != 1)
        return false;

      Node *Predecessor = *N->predecessors().begin();
      if (Predecessor->successorCount() != 1)
        return false;

      revng_assert(*Predecessor->successors().begin() == N);

      if (N == Predecessor)
        return false;

      // Do not simplify nodes in ToPreserve
      if (ToPreserve.contains(N) or ToPreserve.contains(Predecessor))
        return false;

      for (Node *Successor : N->successors())
        if (Successor == N or Successor == Predecessor)
          return false;

      // Drop incoming edge
      N->clearPredecessors();

      // Move over successors
      auto Successors = to_vector(N->successors());
      for (auto &Successor : Successors)
        Predecessor->addSuccessor(Successor);

      // Drop outgoing edges
      N->clearSuccessors();

      // Move operations
      for (Operation &Operation : N->Operations)
        Predecessor->Operations.push_back(Operation);

      // Drop
      return true;
    });
  }

public:
  template<typename S>
  void dump(S &Stream) const {
    for (const Node *N :
         llvm::ReversePostOrderTraversal<const Function *>(this)) {
      Stream << N->label() << ":\n";
      for (const Operation &Operation : N->Operations) {
        Stream << "  " << toString(Operation) << "\n";
      }

      Stream << "  Successors:\n";
      for (const Node *Successor : N->successors()) {
        Stream << "    " << Successor->label() << "\n";
      }

      Stream << "\n";
    }
  }

  void dump() const debug_function { dump(dbg); }
};

} // namespace rua

template<>
struct llvm::DOTGraphTraits<const rua::Function *>
  : public llvm::DefaultDOTGraphTraits {
  using EdgeIterator = llvm::GraphTraits<rua::Function *>::ChildIteratorType;
  DOTGraphTraits(bool IsSimple = false) : DefaultDOTGraphTraits(IsSimple) {}

  static std::string getNodeLabel(const rua::Function::Node *Node,
                                  const rua::Function *Graph) {
    std::string Label;
    Label += Node->label();
    Label += ":\\l";
    for (const rua::Operation &Operation : Node->Operations) {
      Label += "  " + Graph->toString(Operation) + "\\l";
    }

    return Label;
  }
};
