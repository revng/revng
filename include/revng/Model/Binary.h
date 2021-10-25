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
#include "revng/Model/Type.h"
#include "revng/Support/MetaAddress.h"
#include "revng/Support/MetaAddress/YAMLTraits.h"
#include "revng/Support/YAMLTraits.h"

// Forward declarations
namespace model {
class VerifyHelper;
class Function;
class DynamicFunction;
class Binary;
class FunctionEdge;
class CallEdge;
class BasicBlock;
class Segment;
} // namespace model

// TODO: Prevent changing the keys. Currently we need them to be public and
//       non-const for serialization purposes.

template<>
struct KeyedObjectTraits<MetaAddress>
  : public IdentityKeyedObjectTraits<MetaAddress> {};

//
// FunctionAttribute
//

/// Attributes for functions
///
/// \note These attributes can be applied both to functions and call sites
namespace model::FunctionAttribute {
enum Values {
  /// Invalid value
  Invalid,
  /// The function does not return
  NoReturn,
  Count
};
} // namespace model::FunctionAttribute

namespace llvm::yaml {
template<>
struct ScalarEnumerationTraits<model::FunctionAttribute::Values>
  : public NamedEnumScalarTraits<model::FunctionAttribute::Values> {};
} // namespace llvm::yaml

template<>
struct KeyedObjectTraits<model::FunctionAttribute::Values>
  : public IdentityKeyedObjectTraits<model::FunctionAttribute::Values> {};

namespace model::FunctionAttribute {

inline llvm::StringRef getName(Values V) {
  switch (V) {
  case Invalid:
    return "Invalid";
  case NoReturn:
    return "NoReturn";
  case Count:
    revng_abort();
    break;
  }
}

inline Values fromName(llvm::StringRef Name) {
  if (Name == "Invalid") {
    return Invalid;
  } else if (Name == "NoReturn") {
    return NoReturn;
  } else {
    revng_abort();
  }
}

} // namespace model::FunctionAttribute

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
  Unreachable,
  Count
};

inline bool isCall(Values V) {
  switch (V) {
  case Count:
    revng_abort();
    break;

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
struct ScalarEnumerationTraits<model::FunctionEdgeType::Values>
  : public NamedEnumScalarTraits<model::FunctionEdgeType::Values> {};
} // namespace llvm::yaml

namespace model::FunctionEdgeType {

inline llvm::StringRef getName(Values V) {
  switch (V) {
  case Invalid:
    return "Invalid";
  case DirectBranch:
    return "DirectBranch";
  case FakeFunctionCall:
    return "FakeFunctionCall";
  case FakeFunctionReturn:
    return "FakeFunctionReturn";
  case FunctionCall:
    return "FunctionCall";
  case IndirectCall:
    return "IndirectCall";
  case Return:
    return "Return";
  case BrokenReturn:
    return "BrokenReturn";
  case IndirectTailCall:
    return "IndirectTailCall";
  case LongJmp:
    return "LongJmp";
  case Killer:
    return "Killer";
  case Unreachable:
    return "Unreachable";
  case Count:
    revng_abort();
    break;
  }
}

inline Values fromName(llvm::StringRef Name) {
  if (Name == "Invalid") {
    return Invalid;
  } else if (Name == "DirectBranch") {
    return DirectBranch;
  } else if (Name == "FakeFunctionCall") {
    return FakeFunctionCall;
  } else if (Name == "FakeFunctionReturn") {
    return FakeFunctionReturn;
  } else if (Name == "FunctionCall") {
    return FunctionCall;
  } else if (Name == "IndirectCall") {
    return IndirectCall;
  } else if (Name == "Return") {
    return Return;
  } else if (Name == "BrokenReturn") {
    return BrokenReturn;
  } else if (Name == "IndirectTailCall") {
    return IndirectTailCall;
  } else if (Name == "LongJmp") {
    return LongJmp;
  } else if (Name == "Killer") {
    return Killer;
  } else if (Name == "Unreachable") {
    return Unreachable;
  } else {
    revng_abort();
  }
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
  /// Target of the CFG edge
  ///
  /// If invalid, it's an indirect edge such as a return instruction or an
  /// indirect function call.
  /// If valid, it's either the address of the basic block in case of a direct
  /// branch, or, in case of a function call, the address of the callee.
  // TODO: switch to TupleTreeReference
  MetaAddress Destination;

  /// Type of the CFG edge
  FunctionEdgeType::Values Type = FunctionEdgeType::Invalid;

public:
  FunctionEdge() : Destination(MetaAddress::invalid()) {}
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
  bool verify(bool Assert) const debug_function;
  bool verify(VerifyHelper &VH) const;
  void dump() const debug_function;
};
INTROSPECTION_NS(model, FunctionEdge, Destination, Type);

/// A CFG edge to represent function calls (direct, indirect and tail calls)
class model::CallEdge : public model::FunctionEdge {
public:
  using Key = std::pair<MetaAddress, model::FunctionEdgeType::Values>;

public:
  /// Path to the prototype for this call site
  ///
  /// In case of a direct function call, it has to be invalid.
  TypePath Prototype;

  /// Name of the dynamic function being called, or empty if not a dynamic call
  std::string DynamicFunction;

  /// Attributes for this function
  ///
  /// \note To have the effective list of attributes for this call site, you
  ///       have to add attributes on the called function.
  // TODO: switch to std::set
  MutableSet<model::FunctionAttribute::Values> Attributes;

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
  bool verify(bool Assert) const debug_function;
  bool verify(VerifyHelper &VH) const;
  void dump() const debug_function;
};
INTROSPECTION_NS(model,
                 CallEdge,
                 Destination,
                 Type,
                 Prototype,
                 DynamicFunction,
                 Attributes);

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
  : public TupleLikeMappingTraits<model::CallEdge,
                                  Fields<model::CallEdge>::DynamicFunction,
                                  Fields<model::CallEdge>::Attributes> {};

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
  /// An invalid entry
  Invalid,
  /// A function with at least one return instruction
  Regular,
  /// A function that never returns
  NoReturn,
  /// A function with at least one non-proper return instruction
  ///
  /// This typically represents outlined function prologues.
  // TODO: this should not be a function type, but a list of entry points, or
  //       maybe nothing at all
  Fake,
  Count
};

inline llvm::StringRef getName(Values V) {
  switch (V) {
  case Invalid:
    return "Invalid";
  case Regular:
    return "Regular";
  case NoReturn:
    return "NoReturn";
  case Fake:
    return "Fake";
  default:
    revng_abort();
  }
}

} // namespace model::FunctionType

namespace llvm::yaml {
template<>
struct ScalarEnumerationTraits<model::FunctionType::Values>
  : public NamedEnumScalarTraits<model::FunctionType::Values> {};
} // namespace llvm::yaml

/// The basic block of a function
class model::BasicBlock {
public:
  /// Start address of the basic block
  MetaAddress Start;

  /// End address of the basic block, i.e., the address where the last
  /// instruction ends
  MetaAddress End;

  /// Optional custom name
  Identifier CustomName;

  /// List of successor edges
  SortedVector<UpcastablePointer<model::FunctionEdge>> Successors;

public:
  BasicBlock() : Start(MetaAddress::invalid()) {}
  BasicBlock(const MetaAddress &Start) : Start(Start) {}

  bool operator==(const model::BasicBlock &Other) const = default;

public:
  Identifier name() const;

public:
  bool verify() const debug_function;
  bool verify(bool Assert) const debug_function;
  bool verify(VerifyHelper &VH) const;
  void dump() const debug_function;
};
INTROSPECTION_NS(model, BasicBlock, Start, End, CustomName, Successors);

template<>
struct llvm::yaml::MappingTraits<model::BasicBlock>
  : public TupleLikeMappingTraits<model::BasicBlock,
                                  Fields<model::BasicBlock>::CustomName> {};

template<>
struct KeyedObjectTraits<model::BasicBlock> {
  static MetaAddress key(const model::BasicBlock &Obj) { return Obj.Start; }

  static model::BasicBlock fromKey(const MetaAddress &Obj) {
    return model::BasicBlock(Obj);
  }
};

/// A function
class model::Function {
public:
  /// The address of the entry point
  ///
  /// \note This does not necessarily correspond to the address of the basic
  ///       block with the lowest address.
  MetaAddress Entry;

  /// An optional custom name
  Identifier CustomName;

  /// Type of the function
  FunctionType::Values Type = FunctionType::Invalid;

  /// List of basic blocks, which represent the CFG
  SortedVector<model::BasicBlock> CFG;

  /// The prototype of the function
  TypePath Prototype;

  /// Attributes for this call site
  // TODO: switch to std::set
  MutableSet<model::FunctionAttribute::Values> Attributes;

public:
  Function(const MetaAddress &Entry) : Entry(Entry) {}
  bool operator==(const model::Function &Other) const = default;

public:
  Identifier name() const;

public:
  bool verify() const debug_function;
  bool verify(bool Assert) const debug_function;
  bool verify(VerifyHelper &VH) const;
  void dump() const debug_function;

public:
  void dumpCFG() const debug_function;
};
INTROSPECTION_NS(model,
                 Function,
                 Entry,
                 CustomName,
                 Type,
                 CFG,
                 Prototype,
                 Attributes)

template<>
struct llvm::yaml::MappingTraits<model::Function>
  : public TupleLikeMappingTraits<model::Function,
                                  Fields<model::Function>::CustomName,
                                  Fields<model::Function>::CFG,
                                  Fields<model::Function>::Prototype,
                                  Fields<model::Function>::Attributes> {};

template<>
struct KeyedObjectTraits<model::Function> {
  static MetaAddress key(const model::Function &F) { return F.Entry; }
  static model::Function fromKey(const MetaAddress &Key) {
    return model::Function(Key);
  };
};

/// Function defined in a dynamic library
class model::DynamicFunction {
public:
  /// The name of the symbol for this dynamic function
  std::string SymbolName;

  /// An optional custom name
  Identifier CustomName;

  /// The prototype of the function
  TypePath Prototype;

  /// Function attributes
  MutableSet<model::FunctionAttribute::Values> Attributes;

  // TODO: DefiningLibrary

public:
  DynamicFunction() {}
  DynamicFunction(const std::string &SymbolName) : SymbolName(SymbolName) {}
  bool operator==(const model::DynamicFunction &Other) const = default;

public:
  Identifier name() const;

public:
  bool verify() const debug_function;
  bool verify(bool Assert) const debug_function;
  bool verify(VerifyHelper &VH) const;
  void dump() const debug_function;
};

INTROSPECTION_NS(model,
                 DynamicFunction,
                 SymbolName,
                 CustomName,
                 Prototype,
                 Attributes)

template<>
struct llvm::yaml::MappingTraits<model::DynamicFunction>
  : public TupleLikeMappingTraits<model::DynamicFunction,
                                  Fields<model::DynamicFunction>::CustomName,
                                  Fields<model::DynamicFunction>::Attributes> {
};

template<>
struct KeyedObjectTraits<model::DynamicFunction> {

  static auto key(const model::DynamicFunction &F) { return F.SymbolName; }

  static model::DynamicFunction fromKey(const std::string &Key) {
    return model::DynamicFunction(Key);
  }
};

static_assert(validateTupleTree<model::DynamicFunction>(IsYamlizable));

class model::Segment {
public:
  using Key = std::pair<MetaAddress, MetaAddress>;

public:
  MetaAddress StartAddress;
  MetaAddress EndAddress;

  uint64_t StartOffset = 0;
  uint64_t EndOffset = 0;

  bool IsReadable = false;
  bool IsWriteable = false;
  bool IsExecutable = false;

  Identifier CustomName;

public:
  Segment() {}
  Segment(const Key &K) : StartAddress(K.first), EndAddress(K.second) {}
  bool operator==(const model::Segment &Other) const = default;

public:
  Identifier name() const;

public:
  bool verify() const debug_function;
  bool verify(bool Assert) const debug_function;
  bool verify(VerifyHelper &VH) const;
  void dump() const debug_function;
};

INTROSPECTION_NS(model,
                 Segment,
                 StartAddress,
                 EndAddress,
                 StartOffset,
                 EndOffset,
                 IsReadable,
                 IsWriteable,
                 IsExecutable,
                 CustomName)

template<>
struct llvm::yaml::MappingTraits<model::Segment>
  : public TupleLikeMappingTraits<model::Segment,
                                  Fields<model::Segment>::CustomName> {};

template<>
struct KeyedObjectTraits<model::Segment> {

  static model::Segment::Key key(const model::Segment &F) {
    return { F.StartAddress, F.EndAddress };
  }

  static model::Segment fromKey(const model::Segment::Key &K) {
    return model::Segment(K);
  }
};

template<>
struct llvm::yaml::ScalarTraits<model::Segment::Key>
  : public CompositeScalar<model::Segment::Key, '-'> {};

static_assert(validateTupleTree<model::Segment>(IsYamlizable));

/// Data structure representing the whole binary
class model::Binary {
public:
  /// List of the functions within the binary
  SortedVector<model::Function> Functions;

  /// List of the functions within the binary
  SortedVector<model::DynamicFunction> ImportedDynamicFunctions;

  /// Binary architecture
  model::Architecture::Values Architecture = model::Architecture::Invalid;

  /// List of segments in the original binary
  SortedVector<model::Segment> Segments;

  /// Program entry point
  MetaAddress EntryPoint;

  /// The type system
  SortedVector<UpcastablePointer<model::Type>> Types;

public:
  model::TypePath getTypePath(const model::Type *T) {
    return TypePath::fromString(this,
                                "/Types/" + getNameFromYAMLScalar(T->key()));
  }

  TypePath recordNewType(UpcastablePointer<Type> &&T);

  model::TypePath
  getPrimitiveType(PrimitiveTypeKind::Values V, uint8_t ByteSize);

  bool verifyTypes() const debug_function;
  bool verifyTypes(bool Assert) const debug_function;
  bool verifyTypes(VerifyHelper &VH) const;

public:
  bool verify() const debug_function;
  bool verify(bool Assert) const debug_function;
  bool verify(VerifyHelper &VH) const;
  void dump() const debug_function;
  std::string toString() const debug_function;
};
INTROSPECTION_NS(model,
                 Binary,
                 Functions,
                 ImportedDynamicFunctions,
                 Types,
                 Architecture,
                 Segments)

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

inline model::TypePath
getPrototype(const model::Binary &Binary, const model::CallEdge &Edge) {
  if (Edge.Type == model::FunctionEdgeType::FunctionCall) {
    if (not Edge.DynamicFunction.empty()) {
      // Get the dynamic function prototype
      return Binary.ImportedDynamicFunctions.at(Edge.DynamicFunction).Prototype;
    } else if (Edge.Destination.isValid()) {
      // Get the function prototype
      return Binary.Functions.at(Edge.Destination).Prototype;
    } else {
      revng_abort();
    }
  } else {
    return Edge.Prototype;
  }
}

inline bool hasAttribute(const model::Binary &Binary,
                         const model::CallEdge &Edge,
                         model::FunctionAttribute::Values Attribute) {
  using namespace model;

  if (Edge.Attributes.count(Attribute) != 0)
    return true;

  if (Edge.Type == FunctionEdgeType::FunctionCall) {
    const MutableSet<FunctionAttribute::Values> *CalleeAttributes = nullptr;
    if (not Edge.DynamicFunction.empty()) {
      const auto &F = Binary.ImportedDynamicFunctions.at(Edge.DynamicFunction);
      CalleeAttributes = &F.Attributes;
    } else if (Edge.Destination.isValid()) {
      CalleeAttributes = &Binary.Functions.at(Edge.Destination).Attributes;
    } else {
      revng_abort();
    }

    return CalleeAttributes->count(Attribute) != 0;
  }

  return false;
}
