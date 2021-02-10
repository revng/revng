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
class Register;
class BasicBlock;
} // namespace model

template<>
struct KeyedObjectTraits<MetaAddress>
  : public IdentityKeyedObjectTraits<MetaAddress> {};

namespace model::RegisterState {
enum Values {
  Invalid,
  No,
  NoOrDead,
  Dead,
  Yes,
  YesOrDead,
  Maybe,
  Contradiction
};

inline llvm::StringRef getName(Values V) {
  return getNameFromYAMLScalar(V);
}

inline Values fromName(llvm::StringRef Name) {
  return getValueFromYAMLScalar<Values>(Name);
}

} // namespace model::RegisterState

namespace llvm::yaml {
template<>
struct ScalarEnumerationTraits<model::RegisterState::Values> {
  template<typename T>
  static void enumeration(T &io, model::RegisterState::Values &V) {
    using namespace model::RegisterState;
    io.enumCase(V, "Invalid", Invalid);
    io.enumCase(V, "No", No, QuotingType::Double);
    io.enumCase(V, "NoOrDead", NoOrDead);
    io.enumCase(V, "Dead", Dead);
    io.enumCase(V, "Yes", Yes, QuotingType::Double);
    io.enumCase(V, "YesOrDead", YesOrDead);
    io.enumCase(V, "Maybe", Maybe);
    io.enumCase(V, "Contradiction", Contradiction);
  }
};
} // namespace llvm::yaml

class model::Register {
public:
  std::string Name;
  RegisterState::Values Argument = RegisterState::Invalid;
  RegisterState::Values ReturnValue = RegisterState::Invalid;

public:
  Register(const std::string &Name) : Name(Name) {}

public:
  bool verify() const debug_function {
    return Argument != RegisterState::Invalid
           and ReturnValue != RegisterState::Invalid;
  }
};
INTROSPECTION_NS(model, Register, Name, Argument, ReturnValue);

template<>
struct llvm::yaml::MappingTraits<model::Register>
  : public TupleLikeMappingTraits<model::Register> {};

template<>
struct KeyedObjectTraits<model::Register> {
  static std::string key(const model::Register &Obj) { return Obj.Name; }

  static model::Register fromKey(const std::string Name) {
    return model::Register(Name);
  }
};

//
// FunctionEdgeType
//

/// Type of edge on the CFG
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

inline bool hasDestination(Values V) {
  switch (V) {
  case Invalid:
    revng_abort();
    break;
  case DirectBranch:
  case FakeFunctionCall:
  case FakeFunctionReturn:
  case FunctionCall:
    return true;

  case IndirectCall:
  case Return:
  case BrokenReturn:
  case IndirectTailCall:
  case LongJmp:
  case Killer:
  case Unreachable:
    return false;
  }
}

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

/// An edge on the CFG
class model::FunctionEdge {
public:
  /// Edge target. If invalid, it's an indirect edge
  MetaAddress Destination;
  FunctionEdgeType::Values Type;
  SortedVector<model::Register> Registers;

public:
  FunctionEdge(MetaAddress Destination, FunctionEdgeType::Values Type) :
    Destination(Destination), Type(Type) {}

public:
  bool operator<(const FunctionEdge &Other) const {
    auto ThisTie = std::tie(Destination, Type);
    auto OtherTie = std::tie(Other.Destination, Other.Type);
    return ThisTie < OtherTie;
  }

  bool verify() const debug_function;
};
INTROSPECTION_NS(model, FunctionEdge, Destination, Type, Registers);

template<>
struct llvm::yaml::MappingTraits<model::FunctionEdge>
  : public TupleLikeMappingTraits<model::FunctionEdge> {};

template<>
struct KeyedObjectTraits<model::FunctionEdge> {
  using Key = std::pair<MetaAddress, model::FunctionEdgeType::Values>;
  static Key key(const model::FunctionEdge &Obj) {
    return { Obj.Destination, Obj.Type };
  }

  static model::FunctionEdge fromKey(const Key &Obj) {
    return model::FunctionEdge{ Obj.first, Obj.second };
  }
};

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
  std::string Name;
  SortedVector<FunctionEdge> Successors;

public:
  BasicBlock(const MetaAddress &Start) : Start(Start) {}
};
INTROSPECTION_NS(model, BasicBlock, Start, End, Successors);

template<>
struct llvm::yaml::MappingTraits<model::BasicBlock>
  : public TupleLikeMappingTraits<model::BasicBlock> {};

template<>
struct KeyedObjectTraits<model::BasicBlock> {
  static MetaAddress key(const model::BasicBlock &Obj) { return Obj.Start; }

  static model::BasicBlock fromKey(const MetaAddress &Obj) {
    return model::BasicBlock(Obj);
  }
};

//
// Function
//
class model::Function {
public:
  MetaAddress Entry;
  std::string Name;
  FunctionType::Values Type;
  SortedVector<model::BasicBlock> CFG;
  SortedVector<model::Register> Registers;

public:
  Function(const MetaAddress &Entry) : Entry(Entry) {}

public:
  bool verify() const debug_function;
  void dumpCFG() const debug_function;
};
INTROSPECTION_NS(model, Function, Entry, Name, Type, CFG, Registers)

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

public:
  bool verify() const debug_function;
};
INTROSPECTION_NS(model, Binary, Functions)

template<>
struct llvm::yaml::MappingTraits<model::Binary>
  : public TupleLikeMappingTraits<model::Binary> {};
