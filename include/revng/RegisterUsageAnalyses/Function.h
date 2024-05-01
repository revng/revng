#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <string>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SmallVector.h"

#include "revng/ADT/GenericGraph.h"
#include "revng/Model/Register.h"

template<>
struct llvm::DenseMapInfo<model::Register::Values> {
  static model::Register::Values getEmptyKey() {
    return model::Register::Count;
  }

  static model::Register::Values getTombstoneKey() {
    return static_cast<model::Register::Values>(model::Register::Count + 1);
  }

  static unsigned getHashValue(const model::Register::Values &S) { return S; }

  static bool isEqual(const model::Register::Values &LHS,
                      const model::Register::Values &RHS) {
    return LHS == RHS;
  }
};

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
  uint8_t Target = model::Register::Invalid;
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
  llvm::DenseMap<uint8_t, model::Register::Values> IndexToRegister;
  llvm::DenseMap<model::Register::Values, uint8_t> RegisterToIndex;

public:
  Function() = default;

public:
  uint8_t registerIndex(model::Register::Values Register) {
    auto It = RegisterToIndex.find(Register);
    if (It != RegisterToIndex.end())
      return It->second;

    auto RegistersCount = RegisterToIndex.size();
    RegisterToIndex[Register] = RegistersCount;
    IndexToRegister[RegistersCount] = Register;

    revng_assert(RegisterToIndex.size() == IndexToRegister.size());
    revng_assert(RegisterToIndex.size() == 1 + RegistersCount);

    return RegistersCount;
  }

  model::Register::Values registerByIndex(uint8_t Index) const {
    auto It = IndexToRegister.find(Index);
    revng_assert(It != IndexToRegister.end());
    return It->second;
  }

  uint8_t registersCount() const { return IndexToRegister.size(); }

  cppcoro::generator<model::Register::Values>
  registersInSet(const llvm::BitVector &Set) {
    for (unsigned Index : Set.set_bits()) {
      co_yield registerByIndex(Index);
    }
  }

  std::string toString(const Operation &Operation) const {
    auto Register = registerByIndex(Operation.Target);
    return (OperationType::getName(Operation.Type).str() + "("
            + model::Register::getName(Register).str() + ")");
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
