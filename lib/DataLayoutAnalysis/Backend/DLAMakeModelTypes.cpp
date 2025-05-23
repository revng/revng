//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <algorithm>
#include <cstdint>
#include <limits>
#include <variant>

#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/ADT/FilteredGraphTraits.h"
#include "revng/DataLayoutAnalysis/DLATypeSystem.h"
#include "revng/Model/Binary.h"
#include "revng/Model/TypeDefinition.h"
#include "revng/Support/Assert.h"
#include "revng/Support/Debug.h"

#include "DLAMakeModelTypes.h"

using namespace model;
using namespace dla;

static Logger<> Log("dla-make-model");
static Logger<> ModelLog("dla-dump-model");
static Logger<> TypeMapLog("dla-type-map");

using model::UpcastableType;
using LTSN = LayoutTypeSystemNode;

/// Create a single-field struct to wrap a given type
static UpcastableType makeStructWrapper(UpcastableType &&T,
                                        TupleTree<model::Binary> &Model,
                                        uint64_t Offset = 0ULL,
                                        uint64_t WrapperSize = 0ULL) {
  revng_log(Log, "makeStructWrapper");
  LoggerIndent Indent{ Log };

  // Create struct
  auto &&[Struct, NewType] = Model->makeStructDefinition();

  // Create and insert field in struct
  model::StructField &Field = Struct.addField(Offset, std::move(T));
  Struct.Size() = Offset + *Field.Type()->size();
  revng_log(Log, "Created wrapper struct type with Size: " << Struct.Size());

  // If `WrapperSize` is explicitly specified by the caller, ensure that it is
  // not smaller than the sum of the wrapped type's size with its starting
  // offset.
  if (WrapperSize) {
    revng_assert(WrapperSize >= Struct.Size());
    Struct.Size() = WrapperSize;
    revng_log(Log, "Adjusted size to WrapperSize: " << Struct.Size());
  }

  return NewType;
}

/// A fixed-size map that maps each ID of a LayoutTypeSystemNodes in a
/// LayoutTypeSystem onto a model::UpcastableType.
///
/// It exposes an API that makes it look like we're using LayoutTypeSystemNodes
/// as a key but it's actually using their IDs under the hood.
///
/// The map has a fixed size because the number of LayoutTypeSystemNodes is
/// known ahead of time and never changes during its operations.
///
/// Each UpcastableType is initialized as empty. When an ID is mapped onto an
/// empty UpcastableType it means that it's "missing" in the map.
///
/// LayoutTypeSystemNodes in a LayoutTypeSystem have equivalence classes
/// representehd by an instance of dla::EqClasses.
/// These are integer equivalence classes, grouping together the IDs of
/// LayoutTypeSystemNodes that were originally distinct and were merged by DLA
/// during its normal operation.
///
class NodeToUpcastableTypeMap {
public:
  /// Constructor, to create a NodeToUpcastableTypeMap from a LayoutTypeSystem.
  /// The size of the created map will be equal to the number of nodes in the
  /// LayoutTypeSystem itself.
  NodeToUpcastableTypeMap(const dla::LayoutTypeSystem &TS) :
    NodeIDEquivalenceClasses{ TS.getEqClasses() },
    UpcastableTypes{ TS.getEqClasses().getNumClasses() } {}

  UpcastableType &operator[](const LayoutTypeSystemNode *Node) {
    std::optional<unsigned> MaybeTypeIndex = NodeIDEquivalenceClasses
                                               .getEqClassID(Node->ID);
    revng_assert(MaybeTypeIndex.has_value());
    revng_assert(MaybeTypeIndex.value() < UpcastableTypes.size());
    return UpcastableTypes[MaybeTypeIndex.value()];
  }

  const UpcastableType &at(const LayoutTypeSystemNode *Node) const {
    std::optional<unsigned> MaybeTypeIndex = NodeIDEquivalenceClasses
                                               .getEqClassID(Node->ID);
    revng_assert(MaybeTypeIndex.has_value());
    revng_assert(MaybeTypeIndex.value() < UpcastableTypes.size());
    revng_assert(not UpcastableTypes[MaybeTypeIndex.value()].isEmpty());
    return UpcastableTypes[MaybeTypeIndex.value()];
  }

  std::vector<UpcastableType> &types() { return UpcastableTypes; }

private:
  const dla::VectEqClasses &NodeIDEquivalenceClasses;
  std::vector<UpcastableType> UpcastableTypes;
};

static bool representsTypeDefinition(const LayoutTypeSystemNode *Node) {
  if (isPointerNode(Node)) {
    return false;
  } else if (isLeaf(Node)) {
    revng_assert(Node->Size);
    // TODO: strictly speaking, we generate a StructDefinition also from a
    // scalar leaf node whenever its size doesn't match a valid size for
    // Generic. But we can get away not considering that here, since that struct
    // will have no fields and will just be created later opportunistically.
    return Node->NonScalar;
  } else if (isStructNode(Node) or isUnionNode(Node)) {
    return true;
  } else {
    revng_abort("Invalid LayoutTypeSystemNode");
  }
  return false;
}

/// Follow a pointer edge from the pointer to the pointee.
static const LTSN *getPointeeNode(const LTSN *PointerNode) {
  revng_assert(isPointerNode(PointerNode));
  return PointerNode->Successors.begin()->first;
}

static UpcastableType makeStructOrUnion(TupleTree<model::Binary> &Model,
                                        const LTSN *Node) {
  revng_log(Log,
            "Creating struct or union type for node with ID: " << Node->ID);
  LoggerIndent Indent{ Log };

  revng_assert(representsTypeDefinition(Node));
  revng_assert(Node->Size);

  model::UpcastableType NewStructOrUnion;
  if (isLeaf(Node) or isStructNode(Node)) {
    auto &&[Struct, NewType] = Model->makeStructDefinition(Node->Size);
    NewStructOrUnion = NewType;
    revng_log(Log, "StructDefinition with model ID: " << Struct.ID());
  } else if (isUnionNode(Node)) {
    auto &&[Union, NewType] = Model->makeUnionDefinition();
    NewStructOrUnion = NewType;
    revng_log(Log, "UnionDefinition with model ID: " << Union.ID());
  } else {
    revng_abort("Invalid LayoutTypeSystemNode");
  }

  revng_assert(not NewStructOrUnion.isEmpty());

  return NewStructOrUnion;
}

static RecursiveCoroutine<UpcastableType>
getOrCreateUpcastableScalarType(const LTSN *ScalarNode,
                                NodeToUpcastableTypeMap &NodesToTypes,
                                TupleTree<model::Binary> &Model) {

  UpcastableType &Result = NodesToTypes[ScalarNode];
  if (not Result.isEmpty())
    rc_return Result;

  revng_assert(not representsTypeDefinition(ScalarNode));

  if (isPointerNode(ScalarNode)) {
    const LTSN *PointeeNode = getPointeeNode(ScalarNode);
    UpcastableType PointeeType = rc_recur
      getOrCreateUpcastableScalarType(PointeeNode, NodesToTypes, Model);
    Result = model::PointerType::make(std::move(PointeeType),
                                      Model->Architecture());

  } else if (isLeaf(ScalarNode)) {

    Result = model::PrimitiveType::make(model::PrimitiveKind::Generic,
                                        ScalarNode->Size);
    if (not Result->verify()) {
      // If we do not support this primitive type, use a struct instead.
      Result = makeStructOrUnion(Model, ScalarNode);
      revng_assert(Result->verify());
    }

  } else {
    revng_abort("Node does not represent type definition, but it's neither a "
                "pointer nor a scalar");
  }

  revng_assert(not Result.isEmpty());
  revng_assert(not NodesToTypes[ScalarNode].isEmpty());

  rc_return Result;
}

static RecursiveCoroutine<UpcastableType>
makeArrayWrapper(UpcastableType &&FieldType,
                 const OffsetExpression &OE,
                 NodeToUpcastableTypeMap &NodesToTypes,
                 TupleTree<model::Binary> &Model,
                 size_t Index = 0) {
  revng_log(Log, "makeArrayWrapper");
  LoggerIndent IndentWrapper{ Log };

  size_t NumNestedArrays = OE.Strides.size();
  if (Index >= NumNestedArrays) {
    revng_log(Log, "no wrapping arrays left");
    rc_return FieldType;
  }

  // Recur until the innermost array.
  FieldType = rc_recur makeArrayWrapper(std::move(FieldType),
                                        OE,
                                        NodesToTypes,
                                        Model,
                                        Index + 1);

  uint64_t Stride = OE.Strides[Index];

  std::optional<uint64_t> TripCount = OE.TripCounts[Index];
  uint64_t NumElements = TripCount.value_or(1);

  revng_log(Log,
            "Stride " << Stride << "  Trip Count "
                      << (TripCount ? *TripCount : 0));

  // Strides in OE go from bigger to smaller.
  revng_assert(not Index or OE.Strides[--Index] > Stride);

  size_t FieldSize = FieldType->size().value();
  revng_assert(FieldSize <= Stride);

  // If the trip count is none or equal to one, we don't emit an array but a
  // single element.
  if (NumElements < 2)
    rc_return FieldType;

  // If the stride is larger than the size of the inner element, we need to
  // reserve space after each element, hence we wrap it in a struct.
  if (Stride > FieldSize) {
    revng_log(Log, "Creating wrapper");
    // Create a wrapper to hold each element (except for the last one)
    // together with its trailing padding
    UpcastableType ElemWrapper = makeStructWrapper(FieldType.copy(),
                                                   Model,
                                                   /*offset*/ 0,
                                                   /*size*/ Stride);

    // Now make this an array.
    UpcastableType WrappedArray = model::ArrayType::make(std::move(ElemWrapper),
                                                         (NumElements - 1));

    const uint64_t LastElemOffset = Stride * (NumElements - 1);
    const uint64_t ArrayWrapperSize = LastElemOffset + FieldSize;

    // Create a wrapper to hold the array + the last element, which does not
    // need trailing padding.
    UpcastableType ArrayWrapper = makeStructWrapper(std::move(WrappedArray),
                                                    Model,
                                                    /*offset*/ 0,
                                                    /*size*/ ArrayWrapperSize);

    // Insert the last element
    model::StructDefinition &Struct = ArrayWrapper->toStruct();

    // Insert the rest of the array
    Struct.addField(LastElemOffset, std::move(FieldType));
    Struct.Size() = ArrayWrapperSize;

    FieldType = std::move(ArrayWrapper);
  } else {
    revng_log(Log, "Wrapping it into an array with no struct wrappers");
    FieldType = model::ArrayType::make(std::move(FieldType), NumElements);
  }

  rc_return FieldType;
}

enum class FieldKind {
  StructField,
  UnionField
};

/// Generate a field's type from an offset expression, generating also
/// intermediate array and pointer types if needed.
template<FieldKind K>
static UpcastableType makeFieldType(const LTSN *FieldNode,
                                    const OffsetExpression &OE,
                                    NodeToUpcastableTypeMap &NodesToTypes,
                                    TupleTree<model::Binary> &Model) {
  if (Log.isEnabled()) {
    std::string OEStr;
    {
      llvm::raw_string_ostream OS{ OEStr };
      OE.print(OS);
    }
    revng_log(Log, "makeFieldType: FieldNode ID: " << FieldNode->ID << OEStr);
  }
  LoggerIndent Indent{ Log };

  UpcastableType FieldType = getOrCreateUpcastableScalarType(FieldNode,
                                                             NodesToTypes,
                                                             Model);
  revng_assert(OE.Strides.size() == OE.TripCounts.size());
  FieldType = makeArrayWrapper(std::move(FieldType), OE, NodesToTypes, Model);
  if (K == FieldKind::UnionField and OE.Offset) {
    FieldType = makeStructWrapper(std::move(FieldType), Model, OE.Offset);
  }
  return FieldType;
}

static void logEntry(const LayoutTypeSystem &TS,
                     TupleTree<model::Binary> &Model) {
  if (Log.isEnabled())
    TS.dumpDotOnFile("before-make-model.dot");

  if (ModelLog.isEnabled())
    Model->dump("model-before-dla.yaml");

  if (VerifyLog.isEnabled()) {
    revng_assert(Model->verify(true));
    revng_assert(TS.verifyPointerDAG() and TS.verifyDAG()
                 and TS.verifyUnions());
  }
}

static void logExit(const LayoutTypeSystem &TS,
                    TupleTree<model::Binary> &Model) {
  if (VerifyLog.isEnabled()) {
    revng_assert(TS.verifyPointerDAG() and TS.verifyDAG()
                 and TS.verifyUnions());
    revng_assert(Model->verify(true));
  }

  if (Log.isEnabled())
    TS.dumpDotOnFile("after-make-model.dot");

  if (ModelLog.isEnabled())
    Model->dump("model-after-dla.yaml");
}

/// Build a map between LLVM values and types using equivalence classes.
static TypeMapT
mapLLVMValuesToModelTypes(const LayoutTypeSystem &TS,
                          const LayoutTypePtrVect &Values,
                          const std::vector<UpcastableType> &Types) {
  const dla::VectEqClasses &EqClasses = TS.getEqClasses();

  TypeMapT ValMap;
  revng_log(TypeMapLog, "LayoutTypePtr; ModelType");

  for (auto &&[ValueIdx, Val] : llvm::enumerate(Values)) {
    // If the value is in the class that represents nodes removed from the
    // graph, or the value itself does not represent an llvm::Value, we can just
    // skip it.
    if (EqClasses.isRemoved(ValueIdx) or Val.isEmpty())
      continue;

    auto TypeIdx = EqClasses.getEqClassID(ValueIdx);
    revng_assert(TypeIdx.has_value());

    const UpcastableType &T = Types[TypeIdx.value()];
    revng_log(TypeMapLog, Val.toString() << ";" << T->toString());

    bool New = ValMap.emplace(Val, T.copy()).second;
    revng_assert(New);
  }

  return ValMap;
}

static void makeTypeDefinitions(const dla::LayoutTypeSystem &TS,
                                NodeToUpcastableTypeMap &NodesToTypes,
                                TupleTree<model::Binary> &Model) {

  revng_log(Log, "makeTypeDefinitions");
  LoggerIndent Indent{ Log };
  for (const LTSN *Node : llvm::nodes(&TS)) {

    if (not representsTypeDefinition(Node))
      continue;

    UpcastableType NewStructOrUnion = makeStructOrUnion(Model, Node);
    revng_log(Log,
              "Mapped Node with ID: "
                << Node->ID << " to TypeDefinition with ID: "
                << NewStructOrUnion->tryGetAsDefinition()->ID());

    NodesToTypes[Node] = NewStructOrUnion;
  }
}

static void populateStructFields(const LayoutTypeSystemNode *StructNode,
                                 model::StructDefinition *Struct,
                                 const LayoutTypeSystem &TS,
                                 NodeToUpcastableTypeMap &NodesToTypes,
                                 TupleTree<model::Binary> &Model) {
  LoggerIndent Indent{ Log };
  for (const auto &[FieldNode, Edge] : StructNode->Successors) {
    revng_log(Log, "Field ID: " << FieldNode->ID);
    LoggerIndent FieldIndent{ Log };

    revng_assert(TypeLinkTag::LK_Instance == Edge->getKind());

    dla::OffsetExpression OE = Edge->getOffsetExpr();
    UpcastableType
      FieldType = makeFieldType<FieldKind::StructField>(FieldNode,
                                                        OE,
                                                        NodesToTypes,
                                                        Model);
    Struct->addField(OE.Offset, std::move(FieldType));
  }
}

static RecursiveCoroutine<void>
populateUnionFields(const LayoutTypeSystemNode *Node,
                    model::UnionDefinition *Union,
                    const LayoutTypeSystem &TS,
                    NodeToUpcastableTypeMap &NodesToTypes,
                    TupleTree<model::Binary> &Model) {
  LoggerIndent Indent{ Log };
  for (const auto &[FieldNode, Edge] : Node->Successors) {

    revng_log(Log, "Field ID: " << FieldNode->ID);
    LoggerIndent FieldIndent{ Log };

    revng_assert(TypeLinkTag::LK_Instance == Edge->getKind());

    // Recur on fields that are unions themselves and that don't have a size,
    // because otherwise we can't compute their sizes.
    if (isUnionNode(FieldNode)) {
      UpcastableType UnionFieldType = NodesToTypes.at(FieldNode);
      model::TypeDefinition *Definition = UnionFieldType->tryGetAsDefinition();

      if (Definition and llvm::isa<model::UnionDefinition>(Definition)) {
        auto *NestedUnion = llvm::cast<model::UnionDefinition>(Definition);

        if (not NestedUnion->size().has_value())
          rc_recur populateUnionFields(FieldNode,
                                       NestedUnion,
                                       TS,
                                       NodesToTypes,
                                       Model);
      }
    }

    dla::OffsetExpression OE = Edge->getOffsetExpr();
    UpcastableType
      FieldType = makeFieldType<FieldKind::UnionField>(FieldNode,
                                                       OE,
                                                       NodesToTypes,
                                                       Model);
    Union->addField(std::move(FieldType));
  }
  rc_return;
}

static void populateFields(const LayoutTypeSystemNode *Node,
                           const LayoutTypeSystem &TS,
                           NodeToUpcastableTypeMap &NodesToTypes,
                           TupleTree<model::Binary> &Model) {
  revng_log(Log, "Populating fields of Node ID: " << Node->ID);
  model::TypeDefinition *Definition = NodesToTypes[Node]->tryGetAsDefinition();
  revng_assert(Definition);
  if (auto *Struct = llvm::dyn_cast<model::StructDefinition>(Definition)) {
    populateStructFields(Node, Struct, TS, NodesToTypes, Model);
  } else if (auto *Union = llvm::dyn_cast<model::UnionDefinition>(Definition)) {
    if (not Union->size().has_value())
      populateUnionFields(Node, Union, TS, NodesToTypes, Model);
  } else {
    revng_abort("Node should be mapped to a struct or union definition");
  }
}

static void makeScalars(const LayoutTypeSystem &TS,
                        NodeToUpcastableTypeMap &NodesToTypes,
                        TupleTree<model::Binary> &Model) {
  revng_log(Log, "makeScalars");
  LoggerIndent Indent{ Log };

  for (const LTSN *Node : llvm::nodes(&TS))
    if (not representsTypeDefinition(Node))
      getOrCreateUpcastableScalarType(Node, NodesToTypes, Model);
}

TypeMapT dla::makeModelTypes(const LayoutTypeSystem &TS,
                             const LayoutTypePtrVect &Values,
                             TupleTree<model::Binary> &Model) {
  logEntry(TS, Model);

  NodeToUpcastableTypeMap NodesToTypes{ TS };
  makeTypeDefinitions(TS, NodesToTypes, Model);
  makeScalars(TS, NodesToTypes, Model);

  {
    revng_log(Log, "Populating fields");
    LoggerIndent Indent{ Log };

    // We populate fields of unions first, because otherwise union types have
    // no way of knowing their own size, since they have no fields.
    {
      revng_log(Log, "Populating fields of unions");
      LoggerIndent MoreIndent{ Log };
      for (const LTSN *Node : llvm::nodes(&TS))
        if (isUnionNode(Node))
          populateFields(Node, TS, NodesToTypes, Model);
    }

    // At this point we can populate all the fields of structs too.
    {
      revng_log(Log, "Populating fields of structs");
      LoggerIndent MoreIndent{ Log };
      for (const LTSN *Node : llvm::nodes(&TS))
        if (not isUnionNode(Node) and representsTypeDefinition(Node))
          populateFields(Node, TS, NodesToTypes, Model);
    }
  }

  logExit(TS, Model);

  return mapLLVMValuesToModelTypes(TS, Values, NodesToTypes.types());
}
