#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/SmallString.h"

#include "revng/ADT/KeyedObjectContainer.h"
#include "revng/ADT/MutableSet.h"
#include "revng/ADT/SortedVector.h"
#include "revng/Model/TupleTree.h"
#include "revng/Support/MetaAddress.h"
#include "revng/Support/MetaAddress/KeyTraits.h"
#include "revng/Support/MetaAddress/YAMLTraits.h"
#include "revng/Support/YAMLTraits.h"

// Forward declarations
namespace model {
class Function;
class Binary;
class FunctionEdge;
class BasicBlock;
} // namespace model

template<typename T, char Separator>
struct CompositeScalar {
  static_assert(std::tuple_size_v<T> >= 0);

  template<size_t I = 0>
  static void output(const T &Value, void *Ctx, llvm::raw_ostream &Output) {
    if constexpr (I < std::tuple_size_v<T>) {

      if constexpr (I != 0) {
        Output << Separator;
      }

      using element = std::tuple_element_t<I, T>;
      Output << getNameFromYAMLScalar<element>(get<I>(Value));

      CompositeScalar::output<I + 1>(Value, Ctx, Output);
    }
  }

  template<size_t I = 0>
  static llvm::StringRef input(llvm::StringRef Scalar, void *Ctx, T &Value) {
    if constexpr (I < std::tuple_size_v<T>) {
      auto [Before, After] = Scalar.split(Separator);

      using element = std::tuple_element_t<I, T>;
      get<I>(Value) = getValueFromYAMLScalar<element>(Before);

      return CompositeScalar::input<I + 1>(After, Ctx, Value);
    } else {
      revng_assert(Scalar.size() == 0);
      return Scalar;
    }
  }

  static llvm::yaml::QuotingType mustQuote(llvm::StringRef) {
    return llvm::yaml::QuotingType::Double;
  }
};

template<>
struct KeyedObjectTraits<MetaAddress>
  : public IdentityKeyedObjectTraits<MetaAddress> {};

//
// FunctionEdgeType
//
namespace model::FunctionEdgeType {
enum Values {
  /// Invalid value
  Invalid,
  /// Branch due to function-local CFG (a regular branch)
  DirectBranch,
  /// A call to a fake function
  FakeFunctionCall,
  /// A return from a fake function
  FakeFunctionReturn,
  /// A function call for which the cache was able to produce a summary
  FunctionCall,
  /// A function call for which the target is unknown
  IndirectCall,
  /// A proper function return
  Return,
  /// A branch returning to the return address, but leaving the stack
  /// in an unexpected situation
  BrokenReturn,
  /// A branch representing an indirect tail call
  IndirectTailCall,
  /// A branch representing a longjmp or similar constructs
  LongJmp,
  /// A killer basic block (killer syscall or endless loop)
  Killer,
  /// The basic block ends with an unreachable instruction
  Unreachable
};
} // namespace model::FunctionEdgeType

namespace llvm::yaml {
template<>
struct ScalarEnumerationTraits<model::FunctionEdgeType::Values> {
  template<typename T>
  static void enumeration(T &io, model::FunctionEdgeType::Values &V) {
    using namespace model::FunctionEdgeType;
    io.enumCase(V, "Invalid", Invalid);
    io.enumCase(V, "DirectBranch", DirectBranch);
    io.enumCase(V, "FakeFunctionCall", FakeFunctionCall);
    io.enumCase(V, "FakeFunctionReturn", FakeFunctionReturn);
    io.enumCase(V, "FunctionCall", FunctionCall);
    io.enumCase(V, "IndirectCall", IndirectCall);
    io.enumCase(V, "Return", Return);
    io.enumCase(V, "BrokenReturn", BrokenReturn);
    io.enumCase(V, "IndirectTailCall", IndirectTailCall);
    io.enumCase(V, "LongJmp", LongJmp);
    io.enumCase(V, "Killer", Killer);
    io.enumCase(V, "Unreachable", Unreachable);
  }
};
} // namespace llvm::yaml

namespace model::FunctionEdgeType {

inline llvm::StringRef getName(Values V) {
  return getNameFromYAMLScalar(V);
}

inline Values fromName(llvm::StringRef Name) {
  return getValueFromYAMLScalar<Values>(Name);
}

} // namespace model::FunctionEdgeType

//
// FunctionEdge
//
class model::FunctionEdge {
public:
  MetaAddress Destination;
  FunctionEdgeType::Values Type;

  bool operator==(const FunctionEdge &Other) const = default;

  bool operator<(const FunctionEdge &Other) const {
    auto ThisTie = std::tie(Destination, Type);
    auto OtherTie = std::tie(Other.Destination, Other.Type);
    return ThisTie < OtherTie;
  }
};
INTROSPECTION_NS(model, FunctionEdge, Destination, Type);

template<>
struct llvm::yaml::ScalarTraits<model::FunctionEdge>
  : CompositeScalar<model::FunctionEdge, '-'> {};

template<>
struct KeyedObjectTraits<model::FunctionEdge>
  : public IdentityKeyedObjectTraits<model::FunctionEdge> {};

//
// FunctionType
//
namespace model::FunctionType {
enum Values {
  Invalid, ///< An invalid entry
  Regular, ///< A normal function
  NoReturn, ///< A noreturn function
  Fake ///< A fake function
};
}

namespace llvm::yaml {
template<>
struct ScalarEnumerationTraits<model::FunctionType::Values> {
  static void enumeration(IO &io, model::FunctionType::Values &V) {
    using namespace model::FunctionType;
    io.enumCase(V, "Invalid", Invalid);
    io.enumCase(V, "Regular", Regular);
    io.enumCase(V, "NoReturn", NoReturn);
    io.enumCase(V, "Fake", Fake);
  }
};
} // namespace llvm::yaml

class model::BasicBlock {
public:
  MetaAddress Start;
  MetaAddress End;
  SortedVector<FunctionEdge> Successors;

public:
  BasicBlock(const MetaAddress &Start, const MetaAddress &End) :
    Start(Start), End(End) {}
};
INTROSPECTION_NS(model, BasicBlock, Start, End, Successors);

template<>
struct llvm::yaml::MappingTraits<model::BasicBlock>
  : public TupleLikeMappingTraits<model::BasicBlock> {};

template<>
struct KeyedObjectTraits<model::BasicBlock> {
  static std::pair<MetaAddress, MetaAddress> key(const model::BasicBlock &Obj) {
    return { Obj.Start, Obj.End };
  }

  static model::BasicBlock fromKey(std::pair<MetaAddress, MetaAddress> Obj) {
    return model::BasicBlock(Obj.first, Obj.second);
  }
};

template<>
struct llvm::yaml::ScalarTraits<std::pair<MetaAddress, MetaAddress>>
  : CompositeScalar<std::pair<MetaAddress, MetaAddress>, '-'> {};

//
// Function
//
class model::Function {
public:
  MetaAddress Entry;
  std::string Name;
  FunctionType::Values Type;
  SortedVector<model::BasicBlock> CFG;

public:
  Function(const MetaAddress &Entry) : Entry(Entry) {}

public:
  /// Get a set of range of addresses representing the body of the function
  ///
  /// \note The result of this function is deterministic: the first range
  ///       represents the entry basic block, all the other are return sorted by
  ///       address.
  std::vector<std::pair<MetaAddress, MetaAddress>> basicBlockRanges() const;

public:
  bool verifyCFG() const debug_function;
  void dump() const debug_function;
};
INTROSPECTION_NS(model, Function, Entry, Name, Type, CFG)

template<>
struct llvm::yaml::MappingTraits<model::Function>
  : public TupleLikeMappingTraits<model::Function> {};

template<>
struct KeyedObjectTraits<model::Function> {
  static MetaAddress key(const model::Function &F) { return F.Entry; }
  static model::Function fromKey(const MetaAddress &Key) {
    return model::Function(Key);
  };
};

static_assert(is_KeyedObjectContainer_v<MutableSet<model::Function>>);

//
// Binary
//
class model::Binary {
public:
  MutableSet<model::Function> Functions;
};
INTROSPECTION_NS(model, Binary, Functions)

template<>
struct llvm::yaml::MappingTraits<model::Binary>
  : public TupleLikeMappingTraits<model::Binary> {};
