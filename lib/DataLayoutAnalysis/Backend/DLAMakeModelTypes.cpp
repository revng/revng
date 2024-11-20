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

using LTSN = LayoutTypeSystemNode;
using ConstNonPointerFilterT = EdgeFilteredGraph<const LTSN *,
                                                 isNotPointerEdge>;

using TypeVect = std::vector<model::UpcastableType>;

/// Maps a pointer node to all the types that are associated to that pointer.
using PtrFieldsMap = std::map<const LTSN *,
                              llvm::SmallPtrSet<model::Type *, 8>>;

/// Create a single-field struct to wrap a given type
static model::UpcastableType
createStructWrapper(const LTSN *N,
                    UpcastableType &&T,
                    PtrFieldsMap &PointerFieldsToUpdate,
                    TupleTree<model::Binary> &Model,
                    uint64_t Offset = 0ULL,
                    uint64_t WrapperSize = 0ULL) {
  // Create struct
  auto [Struct, NewType] = Model->makeStructDefinition();

  // Create and insert field in struct
  model::StructField &Field = Struct.addField(Offset, std::move(T));

  // If the field has pointer type, save the corresponding type, since we'll
  // want to update it later.
  if (Field.Type()->isPointer()) {
    bool New = PointerFieldsToUpdate[N].insert(Field.Type().get()).second;
    revng_assert(New);
    revng_log(Log,
              "Wrapped pointer node "
                << N->ID << " inside wrapper struct (ID: " << Struct.ID()
                << "):" << Field.Type()->toString());
  }

  Struct.Size() = Offset + *Field.Type()->size();

  // If `WrapperSize` is explicitly specified by the caller, ensure that it is
  // not smaller than the sum of the wrapped type's size with its starting
  // offset.
  if (WrapperSize) {
    revng_assert(WrapperSize >= Struct.Size());
    Struct.Size() = WrapperSize;
  }

  return NewType;
}

/// Retrieve the model type associated to TypeSystem \a Node, if any
static model::UpcastableType getNodeType(TupleTree<model::Binary> &Model,
                                         const LTSN *Node,
                                         const TypeVect &Types,
                                         const VectEqClasses &EqClasses) {
  auto TypeIndex = EqClasses.getEqClassID(Node->ID);
  revng_assert(TypeIndex.has_value());
  revng_assert(TypeIndex.value() < Types.size());
  revng_assert(not Types[TypeIndex.value()].isEmpty());
  return Types[TypeIndex.value()].copy();
}

/// Generate a type from an offset expression, generating also intermediate
/// array wrappers if needed
static model::UpcastableType
makeInstanceType(const LTSN *N,
                 model::UpcastableType &&Inner,
                 const OffsetExpression &OE,
                 PtrFieldsMap &PointerFieldsToUpdate,
                 TupleTree<model::Binary> &Model) {
  model::UpcastableType Result = std::move(Inner);
  revng_assert(OE.Strides.size() == OE.TripCounts.size());
  if (OE.Strides.empty())
    return Result;

  // Create an array for each level of the offset expression, if needed.
  // The Strides in OE go from bigger to smaller by construction, so we have to
  // iterate in reverse, to build the instance type bottom-up.
  auto NestedArrayLevels = llvm::reverse(llvm::zip(OE.TripCounts, OE.Strides));
  uint64_t PrevStride = 0ULL;
  for (const auto &[TC, Stride] : NestedArrayLevels) {
    revng_log(Log, "Stride " << Stride << "  Trip Count " << (TC ? *TC : 0U));
    auto InnerSize = Result->size().value();
    revng_assert(not PrevStride or PrevStride < Stride);
    revng_assert(Stride >= 0);
    uint64_t UStride = static_cast<uint64_t>(Stride);
    revng_assert(UStride >= 0 and InnerSize and UStride >= InnerSize);

    // If the trip count is none or equal to one, we don't emit an array but a
    // single element.
    const uint64_t NumElems = TC.value_or(1);
    if (NumElems < 2)
      continue;

    // If the stride is larger than the size of the inner element, we need to
    // reserve space after each element, hence we wrap it in a struct.
    if (UStride > InnerSize) {
      revng_log(Log, "Creating wrapper");
      // Create a wrapper to hold each element (a part from the last one)
      // together with its trailing padding
      auto ElemWrapper = createStructWrapper(N,
                                             Result.copy(),
                                             PointerFieldsToUpdate,
                                             Model,
                                             /*offset*/ 0,
                                             /*size*/ UStride);

      // Now make this an array.
      auto WrappedArray = model::ArrayType::make(std::move(ElemWrapper),
                                                 (NumElems - 1));

      const uint64_t LastElemOffset = UStride * (NumElems - 1);
      const uint64_t ArrayWrapperSize = LastElemOffset + InnerSize;

      // Create a wrapper to hold the array + the last element, which does not
      // need trailing padding.
      auto ArrayWrapper = createStructWrapper(N,
                                              std::move(WrappedArray),
                                              PointerFieldsToUpdate,
                                              Model,
                                              /*offset*/ 0,
                                              /*size*/ ArrayWrapperSize);

      // Insert the last element
      model::StructDefinition &Struct = ArrayWrapper->toStruct();

      // Insert the rest of the array
      Struct.addField(LastElemOffset, std::move(Result));
      Struct.Size() = ArrayWrapperSize;

      Result = std::move(ArrayWrapper);
    } else {
      revng_log(Log, "Wrapping it into an array with no struct wrappers");

      Result = model::ArrayType::make(std::move(Result), NumElems);
    }
  }

  return Result;
}

/// Create a struct type from a TypeSystem node. For pointer members,
/// omit the pointee type for now.
static model::UpcastableType
makeStructFromNode(const LTSN *N,
                   TypeVect &Types,
                   PtrFieldsMap &PointerFieldsToUpdate,
                   TupleTree<model::Binary> &Model,
                   const VectEqClasses &EqClasses) {
  // Create struct
  revng_log(Log, "Creating struct type for node " << N->ID);
  LoggerIndent StructIndent{ Log };
  auto [Struct, NewType] = Model->makeStructDefinition();
  Struct.Size() = N->Size;

  // This holds the struct fields in the same order as in the model, so we can
  // later insert them in the model already in order, without invalidating
  // iterators, so we can take their address in case they are pointers that need
  // to be fixed up at the end.
  std::map<model::StructField, const LTSN *> Fields;

  // Create the fields.
  for (auto &[SuccNode, SuccEdge] : N->Successors) {
    revng_log(Log, "Child " << SuccNode->ID);

    auto FieldType = getNodeType(Model, SuccNode, Types, EqClasses);

    revng_assert(TypeLinkTag::LK_Instance == SuccEdge->getKind());
    const OffsetExpression &OE = SuccEdge->getOffsetExpr();
    revng_assert(OE.Offset >= 0U);
    uint64_t FieldOffset = OE.Offset;
    FieldType = makeInstanceType(SuccNode,
                                 std::move(FieldType),
                                 OE,
                                 PointerFieldsToUpdate,
                                 Model);

    model::StructField Field{ FieldOffset, {}, {}, {}, std::move(FieldType) };
    bool Inserted = Fields.insert({ std::move(Field), SuccNode }).second;
    revng_assert(Inserted);
  }

  // Reserve the fields, since we're passing around pointers to them, we don't
  // want them to be reallocated on insert.
  Struct.Fields().reserve(N->Successors.size());

  for (auto &[Field, SuccNode] : Fields) {
    const auto &[FieldIt, Inserted] = Struct.Fields().insert(std::move(Field));
    revng_assert(Inserted);

    // If the field is a pointer, save the type: we want to update it later
    if (FieldIt->Type()->isPointer()) {
      bool New = PointerFieldsToUpdate[SuccNode]
                   .insert(FieldIt->Type().get())
                   .second;
      revng_assert(New);
      revng_log(Log,
                "Pointer node "
                  << SuccNode->ID << " inside struct with Model ID: "
                  << Struct.ID() << " at address " << &FieldIt->Type());
      if (Log.isEnabled()) {
        std::string S;
        llvm::raw_string_ostream OS{ S };
        serialize(OS, FieldIt->Type());
        OS.flush();
        revng_log(Log, "Pointer: " << S);
      }
    }
  }

  return NewType;
}

/// Create a union type from a TypeSystem node. For pointer members,
/// keep the pointee type undefined for now.
static model::UpcastableType
makeUnionFromNode(const LTSN *N,
                  TypeVect &Types,
                  PtrFieldsMap &PointerFieldsToUpdate,
                  TupleTree<model::Binary> &Model,
                  const VectEqClasses &EqClasses) {
  // Create union
  revng_log(Log, "Creating union type for node " << N->ID);
  auto [Union, NewType] = Model->makeUnionDefinition();

  LoggerIndent StructIndent{ Log };

  // This holds the union fields in the same order as in the model, so we can
  // later insert them in the model already in order, without invalidating
  // iterators, so we can take their address in case they are pointers that need
  // to be fixed up at the end.
  std::map<UnionField, const LTSN *> Fields;

  for (auto &[FieldIndex, Successor] : llvm::enumerate(N->Successors)) {
    auto &[SuccNode, SuccEdge] = Successor;
    revng_log(Log, "Child " << SuccNode->ID);

    auto FieldType = getNodeType(Model, SuccNode, Types, EqClasses);

    revng_assert(TypeLinkTag::LK_Instance == SuccEdge->getKind());
    const OffsetExpression &OE = SuccEdge->getOffsetExpr();
    revng_assert(OE.Offset >= 0U);
    uint64_t FieldOffset = OE.Offset;
    FieldType = makeInstanceType(SuccNode,
                                 std::move(FieldType),
                                 OE,
                                 PointerFieldsToUpdate,
                                 Model);

    if (FieldOffset)
      FieldType = createStructWrapper(SuccNode,
                                      std::move(FieldType),
                                      PointerFieldsToUpdate,
                                      Model,
                                      FieldOffset);

    UnionField Field{ FieldIndex, {}, {}, {}, FieldType };
    bool Inserted = Fields.insert({ std::move(Field), SuccNode }).second;
    revng_assert(Inserted);
  }

  // Reserve the fields, since we're passing around pointers to them, we don't
  // want them to be reallocated on insert.
  Union.Fields().reserve(N->Successors.size());

  for (auto &[Field, SuccNode] : Fields) {
    // Insert field in union
    const auto &[NewF, Inserted] = Union.Fields().insert(std::move(Field));
    revng_assert(Inserted);

    // If the field is a pointer, save the type: we want to update it later
    if (NewF->Type()->isPointer()) {
      auto [_, S] = PointerFieldsToUpdate[SuccNode].insert(NewF->Type().get());
      revng_assert(S);
      revng_log(Log,
                "Pointer node " << SuccNode->ID
                                << " inside union (ID: " << Union.ID()
                                << "): " << NewF->Type()->toString());
    }
  }

  return NewType;
}

static model::UpcastableType
createNodeType(TupleTree<model::Binary> &Model,
               const LTSN *Node,
               TypeVect &Types,
               const VectEqClasses &EqClasses,
               PtrFieldsMap &PointerFieldsToUpdate) {

  auto TypeIndex = EqClasses.getEqClassID(Node->ID);
  revng_assert(TypeIndex.has_value());

  model::UpcastableType &Result = Types[TypeIndex.value()];
  revng_assert(Result.isEmpty());

  if (isPointerNode(Node)) {
    // All pointer nodes are created as `void *` and they will be backpatched
    // later, to point to the correct type instead of void.
    // This dance is necessary since there's no way to guarantee that the
    // pointee have been visited when we're looking at the pointer.
    Result = model::PointerType::make(model::PrimitiveType::makeVoid(),
                                      Model->Architecture());

    PointerFieldsToUpdate[Node].insert(Result.get());
    revng_log(Log,
              "Found root pointer node " << Node->ID << " at address "
                                         << Result.get());

  } else if (isLeaf(Node)) {
    revng_assert(Node->Size);
    if (Node->NonScalar) {
      Result = makeStructFromNode(Node,
                                  Types,
                                  PointerFieldsToUpdate,
                                  Model,
                                  EqClasses);
    } else {
      Result = model::PrimitiveType::make(model::PrimitiveKind::Generic,
                                          Node->Size);
      if (!Result->verify()) {
        // If we do not support this primitive type, use a struct instead.
        Result = makeStructFromNode(Node,
                                    Types,
                                    PointerFieldsToUpdate,
                                    Model,
                                    EqClasses);
      }
    }
  } else if (isStructNode(Node)) {
    Result = makeStructFromNode(Node,
                                Types,
                                PointerFieldsToUpdate,
                                Model,
                                EqClasses);
  } else if (isUnionNode(Node)) {
    Result = makeUnionFromNode(Node,
                               Types,
                               PointerFieldsToUpdate,
                               Model,
                               EqClasses);
  } else {
    revng_abort("Illegal DLA node encountered when generating model "
                "types.");
  }

  revng_assert(!Result.isEmpty());
  return Result;
}

/// Follow a pointer edge from the pointer to the pointee.
static const LTSN *getPointeeNode(const LTSN *PointerNode) {
  revng_assert(isPointerNode(PointerNode));
  return PointerNode->Successors.begin()->first;
}

/// Create the right type for a pointer node by following pointer
/// edges until you find a pointee.
static std::pair<size_t, const LTSN *>
getNumPointersAndPointee(const LTSN *Node) {
  revng_assert(isPointerNode(Node));
  const LTSN *PointerNode = Node;
  size_t NumPointers = 0;
  while (isPointerNode(PointerNode)) {
    revng_log(Log, "Following Pointer " << PointerNode->ID);
    ++NumPointers;
    PointerNode = getPointeeNode(PointerNode);
  }
  return { NumPointers, PointerNode };
}

static void logEntry(const LayoutTypeSystem &TS,
                     TupleTree<model::Binary> &Model) {
  if (Log.isEnabled())
    TS.dumpDotOnFile("before-make-model.dot");

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
static TypeMapT mapLLVMValuesToModelTypes(const LayoutTypeSystem &TS,
                                          const LayoutTypePtrVect &Values,
                                          const TypeVect &Types) {
  const dla::VectEqClasses &EqClasses = TS.getEqClasses();

  TypeMapT ValMap;
  revng_log(TypeMapLog, "LayoutTypePtr; ModelType");

  for (auto [ValueIdx, Val] : llvm::enumerate(Values)) {
    // If the value is in the class that represents nodes removed from the
    // graph, or the value itself does not represent an llvm::Value, we can just
    // skip it.
    if (EqClasses.isRemoved(ValueIdx) or Val.isEmpty())
      continue;

    auto TypeIdx = EqClasses.getEqClassID(ValueIdx);
    revng_assert(TypeIdx.has_value());

    const model::UpcastableType &T = Types[TypeIdx.value()];
    revng_log(TypeMapLog, Val.toString() << ";" << T->toString());

    bool New = ValMap.emplace(Val, T.copy()).second;
    revng_assert(New);
  }

  return ValMap;
}

TypeMapT dla::makeModelTypes(const LayoutTypeSystem &TS,
                             const LayoutTypePtrVect &Values,
                             TupleTree<model::Binary> &Model) {
  logEntry(TS, Model);

  const dla::VectEqClasses &EqClasses = TS.getEqClasses();
  TypeVect Types;
  Types.resize(EqClasses.getNumClasses());
  PtrFieldsMap PointerFieldsToUpdate;

  // Create nodes for anything that is not a pointer
  llvm::SmallPtrSet<const LTSN *, 16> Visited;
  for (const LTSN *Root : llvm::nodes(&TS)) {
    revng_assert(Root != nullptr);
    if (not isRoot(Root))
      continue;

    for (const LTSN *N : llvm::post_order(ConstNonPointerFilterT(Root))) {

      if (bool New = Visited.insert(N).second; not New)
        continue;

      auto T = createNodeType(Model,
                              N,
                              Types,
                              EqClasses,
                              PointerFieldsToUpdate);

      revng_log(Log,
                "Assigned type " << T->toString() << " to index "
                                 << EqClasses.getEqClassID(N->ID).value());
    }
  }

  // Fix pointers
  // TODO: possible optimization: explore in bfs the pointer edges backwards
  // from each pointee, and update all the parents of the pointer node
  // encountered. In this way, each tree of pointer edges is visited only once.
  revng_log(Log, "Fixing pointer fields");
  LoggerIndent Indent{ Log };
  for (auto [PointerNode, PointerTypes] : PointerFieldsToUpdate) {
    revng_log(Log,
              "Updating " << PointerTypes.size()
                          << " pointer types associated to ptr node: "
                          << PointerNode->ID);
    LoggerIndent MoreIndent{ Log };

    const auto [NumPointers,
                PointeeNode] = getNumPointersAndPointee(PointerNode);
    revng_log(Log, "NumPointers: " << NumPointers);

    auto FinalPointeeType = getNodeType(Model, PointeeNode, Types, EqClasses);
    revng_assert(llvm::isa<model::DefinedType>(FinalPointeeType.get())
                 || llvm::isa<model::PrimitiveType>(FinalPointeeType.get()));

    revng_log(Log, "Final pointee type: " << FinalPointeeType->toString());

    LoggerIndent EvenMoreIndent{ Log };
    for (model::Type *PointerToUpdate : PointerTypes) {
      revng_log(Log,
                "Updating pointer type at address "
                  << PointerToUpdate << ": "
                  << PointerToUpdate->toPointer().toString());

      // Fix the pointee type
      PointerToUpdate->toPointer().PointeeType() = FinalPointeeType.copy();
      revng_log(Log, "NEW: " << PointerToUpdate->toPointer().toString());
    }
  }

  logExit(TS, Model);

  return mapLLVMValuesToModelTypes(TS, Values, Types);
}
