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
#include "revng/Model/Binary.h"
#include "revng/Model/TypeDefinition.h"
#include "revng/Support/Assert.h"
#include "revng/Support/Debug.h"

#include "revng-c/DataLayoutAnalysis/DLATypeSystem.h"

#include "DLAMakeModelTypes.h"

using namespace model;
using namespace dla;

static Logger<> Log("dla-make-model");
static Logger<> ModelLog("dla-dump-model");
static Logger<> TypeMapLog("dla-type-map");

using LTSN = LayoutTypeSystemNode;
using ConstNonPointerFilterT = EdgeFilteredGraph<const LTSN *,
                                                 isNotPointerEdge>;

using TypeVect = std::vector<std::optional<model::QualifiedType>>;

using model::Qualifier;
using model::PrimitiveKind::Generic;
using model::QualifierKind::Array;
using model::QualifierKind::Pointer;

/// Maps a pointer node to all the qualified types that are associated to
/// that pointer.
using PtrFieldsMap = std::map<const LTSN *,
                              llvm::SmallPtrSet<QualifiedType *, 8>>;

/// Create a single-field struct to wrap a given type
static QualifiedType createStructWrapper(const LTSN *N,
                                         const QualifiedType &T,
                                         PtrFieldsMap &PointerFieldsToUpdate,
                                         TupleTree<model::Binary> &Model,
                                         uint64_t Offset = 0ULL,
                                         uint64_t WrapperSize = 0ULL) {
  // Create struct
  auto [Struct, Path] = Model->makeTypeDefinition<model::StructDefinition>();

  // Create and insert field in struct
  StructField Field{ Offset, {}, {}, {}, T };
  const auto &[FieldIt, Inserted] = Struct.Fields().insert(Field);
  revng_assert(Inserted);

  // If the field has pointer type, save the corresponding qualified type for
  // updating it later
  if (T.isPointer()) {
    bool New = PointerFieldsToUpdate[N].insert(&FieldIt->Type()).second;
    revng_assert(New);
    revng_log(Log,
              "Wrapped pointer node "
                << N->ID << " inside wrapper struct with Model ID: "
                << Struct.ID() << " at address " << &FieldIt->Type());
    if (Log.isEnabled()) {
      std::string S;
      llvm::raw_string_ostream OS{ S };
      serialize(OS, FieldIt->Type());
      OS.flush();
      revng_log(Log, "Pointer: " << S);
    }
  }

  // If WrapperSize == 0ULL we use the size of T to set the size of the
  // generated wrapper struct, otherwise we take WrapperSize, but in that case
  // it must be larger or equal than T.size() + Offset;
  Struct.Size() = WrapperSize ? WrapperSize : (Offset + *T.size());

  revng_assert(Struct.Size() and T.size().has_value()
               and Struct.Size() >= (T.size().value() + Offset));
  return QualifiedType{ Path, {} };
}

/// Retrieve the model type associated to TypeSystem \a Node, if any
static const QualifiedType &getNodeType(TupleTree<model::Binary> &Model,
                                        const LTSN *Node,
                                        const TypeVect &Types,
                                        const VectEqClasses &EqClasses) {
  auto TypeIndex = EqClasses.getEqClassID(Node->ID);
  revng_assert(TypeIndex.has_value());
  auto &MaybeResult = Types[TypeIndex.value()];
  revng_assert(MaybeResult.has_value());
  return MaybeResult.value();
}

/// Generate a qualified type from an offset expression, generating also
/// intermediate array wrappers if needed
static QualifiedType
makeInstanceQualifiedType(const LTSN *N,
                          const QualifiedType &Inner,
                          const OffsetExpression &OE,
                          PtrFieldsMap &PointerFieldsToUpdate,
                          TupleTree<model::Binary> &Model) {
  QualifiedType Result = Inner;
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
    auto InnerSize = Result.size().value();
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
      QualifiedType ElemWrapper = createStructWrapper(N,
                                                      Result,
                                                      PointerFieldsToUpdate,
                                                      Model,
                                                      /*offset*/ 0,
                                                      /*size*/ UStride);
      // Now make this an array.
      revng_assert(ElemWrapper.Qualifiers().empty());
      ElemWrapper.Qualifiers().push_back({ Array, (NumElems - 1) });

      const uint64_t LastElemOffset = UStride * (NumElems - 1);
      const uint64_t ArrayWrapperSize = LastElemOffset + InnerSize;

      // Create a wrapper to hold the array + the last element, which does not
      // need trailing padding.
      QualifiedType
        ArrayWrapper = createStructWrapper(N,
                                           ElemWrapper,
                                           PointerFieldsToUpdate,
                                           Model,
                                           /*offset*/ 0,
                                           /*size*/ ArrayWrapperSize);

      // Insert the last element
      revng_assert(ArrayWrapper.Qualifiers().empty());
      auto *UnqualifiedWrapper = ArrayWrapper.UnqualifiedType().get();
      auto *Struct = llvm::cast<model::StructDefinition>(UnqualifiedWrapper);

      // Insert the rest of the array
      StructField TrailingElem{ LastElemOffset, {}, {}, {}, Result };
      Struct->Fields().insert(TrailingElem);
      Struct->Size() = ArrayWrapperSize;

      Result = ArrayWrapper;
    } else {
      revng_log(Log, "Adding array qualifier with no wrappers");

      Result.Qualifiers().push_back({ Array, NumElems });
    }
  }

  return Result;
}

/// Create a struct type from a TypeSystem node. For pointer members,
/// only populate the field's qualifiers and omit the type for now.
static QualifiedType makeStructFromNode(const LTSN *N,
                                        TypeVect &Types,
                                        PtrFieldsMap &PointerFieldsToUpdate,
                                        TupleTree<model::Binary> &Model,
                                        const VectEqClasses &EqClasses) {
  // Create struct
  revng_log(Log, "Creating struct type for node " << N->ID);
  LoggerIndent StructIndent{ Log };
  auto [Struct, Path] = Model->makeTypeDefinition<model::StructDefinition>();
  Struct.Size() = N->Size;

  // This holds the struct fields in the same order as in the model, so we can
  // later insert them in the model already in order, without invalidating
  // iterators, so we can take their address in case they are pointers that need
  // to be fixed up at the end.
  std::map<StructField, const LTSN *> Fields;

  // Create the fields.
  for (auto &[SuccNode, SuccEdge] : N->Successors) {
    revng_log(Log, "Child " << SuccNode->ID);

    QualifiedType FieldType = getNodeType(Model, SuccNode, Types, EqClasses);

    revng_assert(TypeLinkTag::LK_Instance == SuccEdge->getKind());
    const OffsetExpression &OE = SuccEdge->getOffsetExpr();
    revng_assert(OE.Offset >= 0U);
    uint64_t FieldOffset = OE.Offset;
    FieldType = makeInstanceQualifiedType(SuccNode,
                                          FieldType,
                                          OE,
                                          PointerFieldsToUpdate,
                                          Model);

    StructField Field{ FieldOffset, {}, {}, {}, FieldType };
    bool Inserted = Fields.insert({ std::move(Field), SuccNode }).second;
    revng_assert(Inserted);
  }

  // Reserve the fields, since we're passing around pointers to them, we don't
  // want them to be reallocated on insert.
  Struct.Fields().reserve(N->Successors.size());

  for (auto &[Field, SuccNode] : Fields) {
    const auto &[FieldIt, Inserted] = Struct.Fields().insert(std::move(Field));
    revng_assert(Inserted);

    // If the field is a pointer, save the corresponding qualified type
    if (FieldIt->Type().isPointer()) {
      bool
        New = PointerFieldsToUpdate[SuccNode].insert(&FieldIt->Type()).second;
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

  return QualifiedType{ Path, {} };
}

/// Create a union type from a TypeSystem node. For pointer members,
/// only populate the field's qualifiers.
static QualifiedType makeUnionFromNode(const LTSN *N,
                                       TypeVect &Types,
                                       PtrFieldsMap &PointerFieldsToUpdate,
                                       TupleTree<model::Binary> &Model,
                                       const VectEqClasses &EqClasses) {
  // Create union
  revng_log(Log, "Creating union type for node " << N->ID);
  auto [Union, Path] = Model->makeTypeDefinition<model::UnionDefinition>();

  LoggerIndent StructIndent{ Log };

  // This holds the union fields in the same order as in the model, so we can
  // later insert them in the model already in order, without invalidating
  // iterators, so we can take their address in case they are pointers that need
  // to be fixed up at the end.
  std::map<UnionField, const LTSN *> Fields;

  for (auto &Group : llvm::enumerate(N->Successors)) {
    auto &[SuccNode, SuccEdge] = Group.value();
    revng_log(Log, "Child " << SuccNode->ID);

    QualifiedType FieldType = getNodeType(Model, SuccNode, Types, EqClasses);

    revng_assert(TypeLinkTag::LK_Instance == SuccEdge->getKind());
    const OffsetExpression &OE = SuccEdge->getOffsetExpr();
    revng_assert(OE.Offset >= 0U);
    uint64_t FieldOffset = OE.Offset;
    FieldType = makeInstanceQualifiedType(SuccNode,
                                          FieldType,
                                          OE,
                                          PointerFieldsToUpdate,
                                          Model);

    if (FieldOffset)
      FieldType = createStructWrapper(SuccNode,
                                      FieldType,
                                      PointerFieldsToUpdate,
                                      Model,
                                      FieldOffset);

    auto FieldIndex = Group.index();
    UnionField Field{ FieldIndex, {}, {}, {}, FieldType };
    bool Inserted = Fields.insert({ std::move(Field), SuccNode }).second;
    revng_assert(Inserted);
  }

  // Reserve the fields, since we're passing around pointers to them, we don't
  // want them to be reallocated on insert.
  Union.Fields().reserve(N->Successors.size());

  for (auto &[Field, SuccNode] : Fields) {
    // Insert field in union
    const auto &[FieldIt, Inserted] = Union.Fields().insert(std::move(Field));
    revng_assert(Inserted);

    // If the field is a pointer, save the corresponding qualified type
    if (FieldIt->Type().isPointer()) {
      bool
        New = PointerFieldsToUpdate[SuccNode].insert(&FieldIt->Type()).second;
      revng_assert(New);
      revng_log(Log,
                "Pointer node " << SuccNode->ID
                                << " inside union with Model ID: " << Union.ID()
                                << " at address " << &FieldIt->Type());
      if (Log.isEnabled()) {
        std::string S;
        llvm::raw_string_ostream OS{ S };
        serialize(OS, FieldIt->Type());
        OS.flush();
        revng_log(Log, "Pointer: " << S);
      }
    }
  }

  return QualifiedType{ Path, {} };
}

static QualifiedType &createNodeType(TupleTree<model::Binary> &Model,
                                     const LTSN *Node,
                                     TypeVect &Types,
                                     const VectEqClasses &EqClasses,
                                     PtrFieldsMap &PointerFieldsToUpdate) {

  auto TypeIndex = EqClasses.getEqClassID(Node->ID);
  revng_assert(TypeIndex.has_value());

  auto &MaybeResult = Types[TypeIndex.value()];
  revng_assert(not MaybeResult.has_value());

  if (isPointerNode(Node)) {
    // All pointer nodes are created as `void *` and they will be backpatched
    // later, to point to the correct type instead of void.
    // This dance is necessary since there's no way to guarantee that the
    // pointee have been visited when we're looking at the pointer.
    MaybeResult = QualifiedType{
      Model->getPrimitiveType(model::PrimitiveKind::Void, 0),
      { Qualifier::createPointer(Model->Architecture()) }
    };

    if (MaybeResult.value().isPointer()) {
      PointerFieldsToUpdate[Node].insert(&MaybeResult.value());
      revng_log(Log,
                "Found root pointer node " << Node->ID << " at address "
                                           << &MaybeResult.value());
    }

  } else if (isLeaf(Node)) {
    revng_assert(Node->Size);
    if (Node->NonScalar) {
      MaybeResult = makeStructFromNode(Node,
                                       Types,
                                       PointerFieldsToUpdate,
                                       Model,
                                       EqClasses);
    } else {
      const auto &IsValidPrimitiveSize = [](uint64_t Size) {
        if (Size > std::numeric_limits<uint8_t>::max())
          return false;
        return model::PrimitiveDefinition{ Generic, static_cast<uint8_t>(Size) }
          .verify();
      };

      if (IsValidPrimitiveSize(Node->Size)) {
        MaybeResult = QualifiedType{
          Model->getPrimitiveType(Generic, Node->Size), {}
        };
      } else {
        MaybeResult = makeStructFromNode(Node,
                                         Types,
                                         PointerFieldsToUpdate,
                                         Model,
                                         EqClasses);
      }
    }
  } else if (isStructNode(Node)) {
    MaybeResult = makeStructFromNode(Node,
                                     Types,
                                     PointerFieldsToUpdate,
                                     Model,
                                     EqClasses);
  } else if (isUnionNode(Node)) {
    MaybeResult = makeUnionFromNode(Node,
                                    Types,
                                    PointerFieldsToUpdate,
                                    Model,
                                    EqClasses);
  } else {
    revng_abort("Illegal DLA node encountered when generating model "
                "types.");
  }

  revng_assert(MaybeResult.has_value());
  return MaybeResult.value();
}

/// Follow a pointer edge from the pointer to the pointee.
static const LTSN *getPointeeNode(const LTSN *PointerNode) {
  revng_assert(isPointerNode(PointerNode));
  return PointerNode->Successors.begin()->first;
}

/// Create the right QualifiedType for a pointer node, following pointer
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
    writeToFile(Model->toString(), "model-after-dla.yaml");
}

/// Build a map between LLVM values and types using equivalence classes.
static TypeMapT mapLLVMValuesToModelTypes(const LayoutTypeSystem &TS,
                                          const LayoutTypePtrVect &Values,
                                          const TypeVect &Types) {
  const dla::VectEqClasses &EqClasses = TS.getEqClasses();

  TypeMapT ValMap;
  revng_log(TypeMapLog, "LayoutTypePtr; ModelType");

  for (auto LTP : llvm::enumerate(Values)) {
    unsigned ValueIdx = LTP.index();
    auto &Val = LTP.value();
    // If the value is in the class that represents nodes removed from the
    // graph, or the value itself does not represent an llvm::Value, we can just
    // skip it.
    if (EqClasses.isRemoved(ValueIdx) or Val.isEmpty())
      continue;

    auto TypeIdx = EqClasses.getEqClassID(ValueIdx);
    revng_assert(TypeIdx.has_value());

    auto &T = Types[TypeIdx.value()].value();
    if (TypeMapLog.isEnabled()) {
      std::string S;
      llvm::raw_string_ostream OS{ S };
      serialize(OS, T);
      OS.flush();
      revng_log(TypeMapLog, Val.toString() << ";" << S);
    }

    bool New = ValMap.insert(std::make_pair(Val, T)).second;
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

      QualifiedType &T = createNodeType(Model,
                                        N,
                                        Types,
                                        EqClasses,
                                        PointerFieldsToUpdate);

      if (Log.isEnabled()) {
        std::string S;
        llvm::raw_string_ostream OS{ S };
        serialize(OS, T);
        OS.flush();
        revng_log(Log,
                  "Assigned type " << S << " to index "
                                   << EqClasses.getEqClassID(N->ID).value());
      }
    }
  }

  // Fix pointers
  // TODO: possible optimization: explore in bfs the pointer edges backwards
  // from each pointee, and update all the parents of the pointer node
  // encountered. In this way, each tree of pointer edges is visited only once.
  revng_log(Log, "Fixing pointer fields");
  LoggerIndent Indent{ Log };
  for (auto [PointerNode, PointerQTypes] : PointerFieldsToUpdate) {
    revng_log(Log,
              "Updating " << PointerQTypes.size()
                          << " pointer types associated to ptr node: "
                          << PointerNode->ID);
    LoggerIndent MoreIndent{ Log };

    const auto [NumPointers,
                PointeeNode] = getNumPointersAndPointee(PointerNode);
    revng_log(Log, "NumPointers: " << NumPointers);

    auto FinalPointeeType = getNodeType(Model, PointeeNode, Types, EqClasses);
    revng_assert(FinalPointeeType.Qualifiers().empty());

    if (Log.isEnabled()) {
      std::string S;
      llvm::raw_string_ostream OS{ S };
      serialize(OS, FinalPointeeType);
      OS.flush();
      revng_log(Log, "Final pointee type: " << S);
    }

    LoggerIndent EvenMoreIndent{ Log };
    for (QualifiedType *QTypeToUpdate : PointerQTypes) {

      if (Log.isEnabled()) {
        revng_log(Log,
                  "Updating pointer QualifiedType at address "
                    << QTypeToUpdate);
        std::string S;
        llvm::raw_string_ostream OS{ S };
        serialize(OS, *QTypeToUpdate);
        OS.flush();
        revng_log(Log, "OLD: " << S);
      }

      // Fixup the pointee type
      QTypeToUpdate->UnqualifiedType() = FinalPointeeType.UnqualifiedType();

      // If it's a pointer to pointer, add the missing pointer qualifiers.
      revng_assert(Qualifier::isPointer(QTypeToUpdate->Qualifiers().front()));
      auto PtrQual = Qualifier::createPointer(Model->Architecture());
      QTypeToUpdate->Qualifiers().insert(QTypeToUpdate->Qualifiers().begin(),
                                         NumPointers - 1,
                                         PtrQual);
      if (Log.isEnabled()) {
        std::string S;
        llvm::raw_string_ostream OS{ S };
        serialize(OS, *QTypeToUpdate);
        OS.flush();
        revng_log(Log, "NEW: " << S);
      }
    }
  }

  logExit(TS, Model);

  return mapLLVMValuesToModelTypes(TS, Values, Types);
}
