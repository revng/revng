#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/SmallString.h"

#include "revng/ADT/KeyedObjectContainer.h"
#include "revng/ADT/MutableSet.h"
#include "revng/ADT/SortedVector.h"
#include "revng/ADT/UpcastablePointer.h"
#include "revng/ADT/UpcastablePointer/YAMLTraits.h"
#include "revng/Model/Register.h"
#include "revng/Model/TupleTree.h"
#include "revng/Support/MetaAddress.h"
#include "revng/Support/MetaAddress/YAMLTraits.h"
#include "revng/Support/YAMLTraits.h"

// Forward declarations
namespace model {
class Function;
class Binary;
class FunctionEdge;
class CallEdge;
class FunctionABIRegister;
class BasicBlock;
} // namespace model

// TODO: Prevent changing the keys. Currently we need them to be public and
//       non-const for serialization purposes.

template<>
struct KeyedObjectTraits<MetaAddress>
  : public IdentityKeyedObjectTraits<MetaAddress> {};

class model::FunctionABIRegister {
public:
  Register::Values Register;
  RegisterState::Values Argument = RegisterState::Invalid;
  RegisterState::Values ReturnValue = RegisterState::Invalid;

public:
  FunctionABIRegister(const Register::Values &Register) : Register(Register) {}
  bool operator==(const FunctionABIRegister &Other) const = default;

public:
  bool verify() const debug_function {
    return (Register != Register::Invalid and Argument != RegisterState::Invalid
            and ReturnValue != RegisterState::Invalid);
  }
};
INTROSPECTION_NS(model, FunctionABIRegister, Register, Argument, ReturnValue);

template<>
struct llvm::yaml::MappingTraits<model::FunctionABIRegister>
  : public TupleLikeMappingTraits<model::FunctionABIRegister> {};

template<>
struct KeyedObjectTraits<model::FunctionABIRegister> {
  static model::Register::Values key(const model::FunctionABIRegister &Obj) {
    return Obj.Register;
  }

  static model::FunctionABIRegister
  fromKey(const model::Register::Values &Register) {
    return model::FunctionABIRegister(Register);
  }
};

//
// FunctionEdgeType
//

// TODO: we need to handle noreturn function calls

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

inline bool isCall(Values V) {
  switch (V) {
  case FunctionCall:
  case IndirectCall:
  case IndirectTailCall:
    return true;

  case Invalid:
  case DirectBranch:
  case FakeFunctionCall:
  case FakeFunctionReturn:
  case Return:
  case BrokenReturn:
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
  static void enumeration(T &IO, model::FunctionEdgeType::Values &V) {
    using namespace model::FunctionEdgeType;
    IO.enumCase(V, "Invalid", Invalid);
    IO.enumCase(V, "DirectBranch", DirectBranch);
    IO.enumCase(V, "FakeFunctionCall", FakeFunctionCall);
    IO.enumCase(V, "FakeFunctionReturn", FakeFunctionReturn);
    IO.enumCase(V, "FunctionCall", FunctionCall);
    IO.enumCase(V, "IndirectCall", IndirectCall);
    IO.enumCase(V, "Return", Return);
    IO.enumCase(V, "BrokenReturn", BrokenReturn);
    IO.enumCase(V, "IndirectTailCall", IndirectTailCall);
    IO.enumCase(V, "LongJmp", LongJmp);
    IO.enumCase(V, "Killer", Killer);
    IO.enumCase(V, "Unreachable", Unreachable);
  }
};
} // namespace llvm::yaml

namespace model::FunctionEdgeType {

inline llvm::StringRef getName(Values V) {
  return getNameFromYAMLEnumScalar(V);
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
  using Key = std::pair<MetaAddress, model::FunctionEdgeType::Values>;

public:
  /// Edge target. If invalid, it's an indirect edge
  MetaAddress Destination;
  FunctionEdgeType::Values Type;

public:
  FunctionEdge() :
    Destination(MetaAddress::invalid()), Type(FunctionEdgeType::Invalid) {}
  FunctionEdge(MetaAddress Destination, FunctionEdgeType::Values Type) :
    Destination(Destination), Type(Type) {}

public:
  bool operator==(const FunctionEdge &Other) const = default;

  bool operator<(const FunctionEdge &Other) const {
    auto ThisTie = std::tie(Destination, Type);
    auto OtherTie = std::tie(Other.Destination, Other.Type);
    return ThisTie < OtherTie;
  }

public:
  static constexpr const char *Tag = "!FunctionEdge";
  static bool classof(const FunctionEdge *A) {
    return not FunctionEdgeType::isCall(A->Type);
  }

  bool verify() const debug_function;
};
INTROSPECTION_NS(model, FunctionEdge, Destination, Type);

class model::CallEdge : public model::FunctionEdge {
public:
  using Key = std::pair<MetaAddress, model::FunctionEdgeType::Values>;

public:
  SortedVector<model::FunctionABIRegister> Registers;

public:
  CallEdge() :
    FunctionEdge(MetaAddress::invalid(), FunctionEdgeType::FunctionCall) {}
  CallEdge(MetaAddress Destination, FunctionEdgeType::Values Type) :
    FunctionEdge(Destination, Type) {
    revng_assert(FunctionEdgeType::isCall(Type));
  }

public:
  static constexpr const char *Tag = "!CallEdge";
  static bool classof(const FunctionEdge *A) {
    return FunctionEdgeType::isCall(A->Type);
  }

public:
  bool verify() const debug_function;
};
INTROSPECTION_NS(model, CallEdge, Destination, Type, Registers);

template<>
struct concrete_types_traits<model::FunctionEdge> {
  using type = std::tuple<model::CallEdge, model::FunctionEdge>;
};

template<>
class llvm::yaml::MappingTraits<UpcastablePointer<model::FunctionEdge>>
  : public PolymorphicMappingTraits<UpcastablePointer<model::FunctionEdge>> {};

template<>
struct llvm::yaml::MappingTraits<model::FunctionEdge>
  : public TupleLikeMappingTraits<model::FunctionEdge> {};

template<>
struct llvm::yaml::MappingTraits<model::CallEdge>
  : public TupleLikeMappingTraits<model::CallEdge> {};

template<>
struct llvm::yaml::ScalarTraits<model::FunctionEdge::Key>
  : public CompositeScalar<model::FunctionEdge::Key, '-'> {};

template<>
struct KeyedObjectTraits<model::FunctionEdge> {
  using Key = model::FunctionEdge::Key;
  static Key key(const model::FunctionEdge &Obj) {
    return { Obj.Destination, Obj.Type };
  }

  static model::FunctionEdge fromKey(const Key &Obj) {
    return model::FunctionEdge{ Obj.first, Obj.second };
  }
};

template<>
struct KeyedObjectTraits<UpcastablePointer<model::FunctionEdge>> {
  using Key = model::FunctionEdge::Key;
  static Key key(const UpcastablePointer<model::FunctionEdge> &Obj) {
    return { Obj->Destination, Obj->Type };
  }

  static UpcastablePointer<model::FunctionEdge> fromKey(const Key &Obj) {
    using ResultType = UpcastablePointer<model::FunctionEdge>;
    if (model::FunctionEdgeType::isCall(Obj.second)) {
      return ResultType(new model::CallEdge(Obj.first, Obj.second));
    } else {
      return ResultType(new model::FunctionEdge(Obj.first, Obj.second));
    }
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
  SortedVector<UpcastablePointer<model::FunctionEdge>> Successors;

public:
  BasicBlock(const MetaAddress &Start) : Start(Start) {}
  bool operator==(const model::BasicBlock &Other) const = default;
};
INTROSPECTION_NS(model, BasicBlock, Start, End, Name, Successors);

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
  SortedVector<model::FunctionABIRegister> Registers;

public:
  Function(const MetaAddress &Entry) : Entry(Entry) {}
  bool operator==(const model::Function &Other) const = default;

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

static_assert(IsKeyedObjectContainer<MutableSet<model::Function>>);

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

static_assert(validateTupleTree<model::Binary>(IsYamlizable),
              "All elements of the model must be YAMLizable");

constexpr auto OnlyKOC = [](auto *K) {
  using type = std::remove_pointer_t<decltype(K)>;
  if constexpr (IsContainer<type>) {
    if constexpr (IsKeyedObjectContainer<type>) {
      using value_type = typename type::value_type;
      using KOT = KeyedObjectTraits<value_type>;
      using KeyType = decltype(KOT::key(std::declval<value_type>()));
      return Yamlizable<KeyType>;
    } else {
      return false;
    }
  } else {
    return true;
  }
};
static_assert(validateTupleTree<model::Binary>(OnlyKOC),
              "Only SortedVectors and MutableSets with YAMLizable keys are "
              "allowed");
