//
// Copyright (c) rev.ng Srls. See LICENSE.md for details.
//

#include <algorithm>
#include <cstdint>
#include <variant>

#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/ADT/FilteredGraphTraits.h"
#include "revng/Model/Binary.h"
#include "revng/Model/Type.h"
#include "revng/Support/Assert.h"
#include "revng/Support/Debug.h"

#include "revng-c/DataLayoutAnalysis/DLATypeSystem.h"

#include "../DLAHelpers.h"
#include "DLAMakeModelTypes.h"

using namespace model;
using namespace dla;

static Logger<> Log("dla-make-model");
static Logger<> ModelLog("dla-dump-model");
static Logger<> TypeMapLog("dla-type-map");

using LTSN = LayoutTypeSystemNode;
using ConstNonPointerFilterT = EdgeFilteredGraph<const LTSN *,
                                                 isNotPointerEdge>;

using WritableModelT = TupleTree<model::Binary>;

using model::PrimitiveTypeKind::Number;
using model::QualifierKind::Array;
using model::QualifierKind::Pointer;
using model::TypeKind::Struct;
using model::TypeKind::Typedef;
using model::TypeKind::Union;

///\brief Uniquely identifies a field inside a struct or union.
/// This is needed because the pointer to the qualified type itself is not
/// stable, since fields are stored in a SortedVector.
using StructOrUnionFieldId = std::pair<TypePath, /*offset or id=*/uint64_t>;

///\brief Stores a stable identifier to a QualifiedType.
using QTypeId = std::variant<QualifiedType *, StructOrUnionFieldId>;
///\brief Maps a pointer node to all the qualified types that are associated to
/// that pointer.
using PtrFieldsMap = std::map<const LTSN *, llvm::SmallVector<QTypeId, 8>>;

///\brief Get a union or struct field given its parent and the offset/id.
static QualifiedType *
getFieldFromId(TypePath &Path, uint64_t ID, WritableModelT &Model) {
  if (Path.get()->Kind == Typedef) {
    revng_assert(ID == 0);
    return &llvm::cast<TypedefType>(Path.get())->UnderlyingType;
  }

  if (Path.get()->Kind == Struct) {
    auto &Fields = llvm::cast<StructType>(Path.get())->Fields;
    auto &Field = Fields.at(ID);
    return &(Field.Type);
  }

  if (Path.get()->Kind == Union) {
    auto &Fields = llvm::cast<UnionType>(Path.get())->Fields;
    auto &Field = Fields.at(ID);
    return &(Field.Type);
  }

  revng_abort("Expecting a struct or union here");
}

///\brief Descend all wrapper structures until you find an empty type.
static QualifiedType *unwrapUntilEmpty(QualifiedType *Q) {
  QualifiedType *Field = Q;
  while (Field->UnqualifiedType.Root != nullptr) {
    revng_log(Log,
              "Descending the wrappers: " << Field->UnqualifiedType.get()->ID);

    auto *T = Field->UnqualifiedType.get();
    auto &Fields = llvm::cast<StructType>(T)->Fields;
    revng_assert(Fields.size() == 1);
    Field = &Fields.begin()->Type;
  }

  return Field;
}

///\brief Get a QualifiedType* from its identifier.
static QualifiedType *getQTypeFromId(QTypeId &ID, WritableModelT &Model) {
  // For stable QualifiedTypes, i.e. the ones created outside of the model, the
  // id is the pointer itself.
  if (std::holds_alternative<QualifiedType *>(ID))
    return std::get<QualifiedType *>(ID);

  // For QualifiedTypes that identify a field in a struct or union of the model,
  // since the pointers are not stable, we have to find the corresponding
  // QualifiedTypes starting from the parent struct or union.
  if (std::holds_alternative<StructOrUnionFieldId>(ID)) {
    auto &[Path, Id] = std::get<StructOrUnionFieldId>(ID);
    QualifiedType *FieldQType = getFieldFromId(Path, Id, Model);

    // The field could be wrapped inside a wrapper produced when creating
    // arrays: return the innermost type
    return unwrapUntilEmpty(FieldQType);
  }

  revng_abort("QTypeID can only be a pointer or a StructOrUnionFieldId");
}

///\brief Retrieve the model type associated to TypeSystem \a Node, if any
static QualifiedType *
getNodeType(const LTSN *Node, TypeVect &Types, const VectEqClasses &EqClasses) {
  auto TypeIndex = EqClasses.getEqClassID(Node->ID);
  if (not TypeIndex.has_value())
    return nullptr;

  QualifiedType *T = &Types[TypeIndex.value()];
  return T;
}

///\brief Check if an OffsetExpression has a StrideSize that is smaller that
///       it's inner element size.
static bool
hasValidStrides(const OffsetExpression &OE, const uint64_t InitialSize) {
  long InnerSize = InitialSize;
  revng_assert(OE.TripCounts.size() == OE.Strides.size());
  auto NestedArrays = llvm::reverse(llvm::zip(OE.TripCounts, OE.Strides));

  bool HasTC = true;
  for (const auto &Level : llvm::enumerate(NestedArrays)) {
    auto &[TC, S] = Level.value();
    if (S < InnerSize)
      return false;

    // Check that you never have an inner type with an empty TC and an outer
    // type with a defined TC
    if (TC.has_value())
      revng_assert(HasTC);
    else
      HasTC = false;

    uint64_t NumElems = TC.value_or(1);
    InnerSize = S * NumElems;
  }

  return true;
}

///\brief Create a single-field struct to wrap a given type
static TypePath createStructWrapper(const TypePath &T,
                                    const uint64_t Offset,
                                    const uint64_t ElementSize,
                                    WritableModelT &Model,
                                    const uint64_t ID) {
  StructField F = StructField{ Offset };
  F.Type.UnqualifiedType = T;

  TypePath StructPath = Model->recordNewType(makeType<StructType>());
  auto *Struct = llvm::cast<StructType>(StructPath.get());
  Struct->Fields.insert(F);
  Struct->Size = ElementSize + Offset;

  if (VerifyLog.isEnabled()) {
    std::string S;
    llvm::raw_string_ostream SS(S);
    SS << "Wrapper_" << ID << "__" << Struct->ID;
    Struct->CustomName = SS.str();
  }

  return StructPath;
}

///\brief Generate a qualified type from an offset expression, generating also
/// intermediate array wrappers if needed
static std::pair<QualifiedType, uint64_t>
makeFieldFromInstanceEdge(const TypePath &T,
                          const uint64_t Size,
                          const OffsetExpression &OE,
                          WritableModelT &Model,
                          const uint64_t ID) {
  QualifiedType Inner;
  Inner.UnqualifiedType = T;
  long InnerSize = Size;

  // Bail out if the stride sizes are incompatible with their inner elements
  // TODO: check the stride validity during Type construction instead of before
  if (not hasValidStrides(OE, Size))
    return { Inner, 0U };

  // Create an array for each level of the offset expression, if needed
  InnerSize = Size;
  revng_assert(OE.Strides.size() == OE.TripCounts.size());
  auto NestedArrayLevels = llvm::reverse(llvm::zip(OE.TripCounts, OE.Strides));

  for (const auto &[TC, Stride] : NestedArrayLevels) {
    revng_log(Log, "Stride " << Stride << "  Trip Count " << (TC ? *TC : 0U));
    revng_assert(InnerSize and Stride >= InnerSize);

    // If the trip count is none or equal to one, we don't emit an array but a
    // single element.
    // TODO: inflate arrays with no TripCount
    const uint64_t NumElems = TC.value_or(1);
    if (NumElems < 2)
      continue;

    // If the stride is larger than the size of the inner element, we need to
    // reserve space after each element, hence we wrap it in a struct.
    if (Stride > InnerSize) {
      revng_log(Log, "Creating wrapper");
      // Create a wrapper to hold each element (a part from the last one)
      // together with its trailing padding
      auto ElemWrapper = createStructWrapper(Inner.UnqualifiedType,
                                             /*offset=*/0U,
                                             /*size=*/Stride,
                                             Model,
                                             ID);

      const uint64_t LastElemOffset = Stride * (NumElems - 1);
      const uint64_t ArrayWrapperSize = LastElemOffset + InnerSize;
      // Create a wrapper to hold the array + the last element, which does not
      // need trailing padding.
      // Insert the last element during creation.
      auto ArrayWrapper = createStructWrapper(Inner.UnqualifiedType,
                                              /*offset=*/LastElemOffset,
                                              /*size=*/ArrayWrapperSize,
                                              Model,
                                              ID);
      // Insert the rest of the array
      StructField ArrayField = StructField{ /*offset=*/0U };
      ArrayField.Type.UnqualifiedType = ElemWrapper;
      auto *ArrayWrapperStruct = llvm::cast<StructType>(ArrayWrapper.get());
      ArrayField.Type.Qualifiers.push_back({ Array, (NumElems - 1) });
      ArrayWrapperStruct->Fields.insert(ArrayField);
      ArrayWrapperStruct->Size = ArrayWrapperSize;

      Inner.UnqualifiedType = ArrayWrapper;
      InnerSize = ArrayWrapperSize;
    } else {
      revng_log(Log, "Adding array qualifier with no wrappers");

      Inner.Qualifiers.push_back({ Array, NumElems });
      InnerSize = Stride * NumElems;
    }
  }

  return { Inner, InnerSize };
}

///\brief Create a struct type from a TypeSystem node. For pointer members,
/// only populate the field's qualifiers and omit the type for now.
static TypePath makeStructFromNode(const LTSN *N,
                                   TypeVect &Types,
                                   PtrFieldsMap &PtrFields,
                                   WritableModelT &Model,
                                   const VectEqClasses &EqClasses) {
  // Create struct
  revng_log(Log, "Creating struct type for node " << N->ID);
  TypePath StructPath = Model->recordNewType(makeType<StructType>());
  auto *Struct = llvm::cast<StructType>(StructPath.get());
  Struct->Size = N->Size;

  bool InheritsFromOther = false;
  for (auto &[SuccNode, SuccEdge] : N->Successors) {
    revng_log(Log, "Child " << SuccNode->ID);
    StructField Field{ /*offset=*/0U };

    if (not isPointerNode(SuccNode)) {
      const QualifiedType *T = getNodeType(SuccNode, Types, EqClasses);
      revng_assert(T);
      Field.Type.UnqualifiedType = T->UnqualifiedType;
    } else {
      // Don't specify the UnqualifiedType of pointer fields yet
      revng_log(Log, "Pointer");
    }

    switch (SuccEdge->getKind()) {
    case TypeLinkTag::LK_Instance: {
      revng_log(Log, " Instance");

      const OffsetExpression &OE = SuccEdge->getOffsetExpr();
      revng_assert(OE.Offset >= 0U);
      Field.Offset = OE.Offset;
      // Create a field inspecting the OffsetExpression of the instance edge
      auto [ChildType,
            ChildSize] = makeFieldFromInstanceEdge(Field.Type.UnqualifiedType,
                                                   SuccNode->Size,
                                                   OE,
                                                   Model,
                                                   N->ID);
      if (ChildSize == 0U)
        continue;

      Field.Type = ChildType;
    } break;

    case TypeLinkTag::LK_Inheritance: {
      revng_log(Log, " Inheritance");
      // Check that there is only one inheritance edge.
      revng_assert(not InheritsFromOther);
      InheritsFromOther = true;
    } break;

    case TypeLinkTag::LK_Pointer: {
      revng_abort("A struct node cannot have outgoing pointer edges.");
    } break;

    default:
      revng_unreachable("Unexpected edge tag");
    }

    // Insert field in struct
    auto [Elem, Inserted] = Struct->Fields.insert(Field);
    revng_assert(Inserted);

    // If the field is a pointer, save the corresponding qualified type
    if (isPointerNode(SuccNode)) {
      StructOrUnionFieldId StructFieldId(StructPath, Elem->Offset);
      PtrFields[SuccNode].push_back(StructFieldId);
    }
  }

  return StructPath;
}

///\brief Create a union type from a TypeSystem node. For pointer members,
/// only populate the field's qualifiers.
static TypePath makeUnionFromNode(const LTSN *N,
                                  TypeVect &Types,
                                  PtrFieldsMap &PtrFields,
                                  WritableModelT &Model,
                                  const VectEqClasses &EqClasses) {
  // Create union
  revng_log(Log, "Creating union type for node " << N->ID);
  TypePath UnionPath = Model->recordNewType(makeType<UnionType>());
  auto *Union = llvm::cast<UnionType>(UnionPath.get());

  bool InheritsFromOther = false;
  uint64_t Index = 0;
  for (auto &[SuccNode, SuccEdge] : N->Successors) {
    revng_log(Log, "Child " << SuccNode->ID);
    UnionField Field(Index);

    if (not isPointerNode(SuccNode)) {
      const QualifiedType *T = getNodeType(SuccNode, Types, EqClasses);
      revng_assert(T);
      Field.Type.UnqualifiedType = T->UnqualifiedType;
    } else {
      revng_log(Log, "Pointer");
    }

    switch (SuccEdge->getKind()) {
    case TypeLinkTag::LK_Instance: {
      revng_log(Log, "Instance");

      const OffsetExpression &OE = SuccEdge->getOffsetExpr();
      revng_assert(OE.Offset >= 0LL);

      // Create a field inspecting the OffsetExpression of the instance edge
      auto [ChType,
            ChSize] = makeFieldFromInstanceEdge(Field.Type.UnqualifiedType,
                                                SuccNode->Size,
                                                OE,
                                                Model,
                                                SuccNode->ID);
      if (ChSize == 0U)
        continue;

      // Wrap in a struct if there's padding before the field
      if (OE.Offset > 0U)
        ChType.UnqualifiedType = createStructWrapper(ChType.UnqualifiedType,
                                                     OE.Offset,
                                                     ChSize,
                                                     Model,
                                                     SuccNode->ID);
      Field.Type = ChType;

    } break;

    case TypeLinkTag::LK_Inheritance: {
      revng_log(Log, "Inheritance");
      // Check that there is only one inheritance edge.
      revng_assert(not InheritsFromOther);
      InheritsFromOther = true;
    } break;

    case TypeLinkTag::LK_Pointer: {
      revng_abort("A union node cannot have outgoing pointer edges.");
    } break;

    default:
      revng_unreachable("Unexpected edge tag");
    }

    // Insert field in struct
    auto [Elem, Inserted] = Union->Fields.insert(Field);
    revng_assert(Inserted);

    // If the field is a pointer, save the corresponding qualified type
    if (isPointerNode(SuccNode)) {
      StructOrUnionFieldId UnionFieldId(UnionPath, Elem->Index);
      PtrFields[SuccNode].push_back(UnionFieldId);
    }

    ++Index;
  }

  return UnionPath;
}

///\brief Follow a pointer edge from the pointer to the pointee.
static const LTSN *getPointeeNode(const LTSN *PointerNode) {
  if (VerifyLog.isEnabled())
    revng_assert(llvm::count_if(PointerNode->Successors, isPointerEdge) == 1
                 and PointerNode->Successors.size() == 1);
  return PointerNode->Successors.begin()->first;
}

///\brief Create the right QualifiedType for a pointer node, following pointer
/// edges until you find a pointee.
static QualifiedType makePtrTypeFromNode(const LTSN *Node,
                                         TypeVect &Types,
                                         const VectEqClasses &EqClasses) {
  revng_assert(isPointerNode(Node));
  llvm::SmallPtrSet<const LTSN *, 16> Visited;
  QualifiedType QType;

  const LTSN *PointeeNode = Node;
  while (isPointerNode(PointeeNode)) {
    revng_log(Log, "Following Pointer " << PointeeNode->ID);
    // Check that there are no loops
    if (VerifyLog.isEnabled()) {
      revng_assert(not Visited.contains(PointeeNode));
      Visited.insert(PointeeNode);
    }

    QType.Qualifiers.push_back({ Pointer, PointeeNode->Size });
    PointeeNode = getPointeeNode(PointeeNode);
  }

  const QualifiedType *T = getNodeType(PointeeNode, Types, EqClasses);
  revng_assert(T);
  QType.UnqualifiedType = T->UnqualifiedType;

  revng_log(Log,
            "Pointee node " << PointeeNode->ID
                            << ", type id:" << QType.UnqualifiedType.get()->ID);

  return QType;
}

static void logEntry(const LayoutTypeSystem &TS, WritableModelT &Model) {
  if (Log.isEnabled())
    TS.dumpDotOnFile("before-make-model.dot");
  if (VerifyLog.isEnabled()) {
    revng_assert(Model->verify(true));
    revng_assert(TS.verifyDAG() and TS.verifyInheritanceTree()
                 and TS.verifyUnions());
  }
}

static void logExit(const LayoutTypeSystem &TS, WritableModelT &Model) {
  if (VerifyLog.isEnabled()) {
    revng_assert(TS.verifyDAG() and TS.verifyInheritanceTree()
                 and TS.verifyUnions());
    revng_assert(Model->verify(true));
  }
  if (Log.isEnabled())
    TS.dumpDotOnFile("after-make-model.dot");
  if (ModelLog.isEnabled())
    writeToFile(Model->toString(), "model-after-dla.yaml");
}

/// \brief Build a map between LLVM values and types using equivalence classes.
static TypeMapT mapLLVMValuesToModelTypes(const LayoutTypeSystem &TS,
                                          const LayoutTypePtrVect &Values,
                                          const TypeVect &Types) {
  const dla::VectEqClasses &EqClasses = TS.getEqClasses();

  TypeMapT ValMap;
  revng_log(TypeMapLog, "LayoutTypePtr; ModelType");

  for (auto LTP : llvm::enumerate(Values)) {
    unsigned ValueIdx = LTP.index();
    auto &Val = LTP.value();
    if (EqClasses.isRemoved(ValueIdx) or Val.isEmpty())
      continue;

    auto TypeIdx = EqClasses.getEqClassID(ValueIdx);
    if (TypeIdx.has_value()) {
      auto &T = Types[TypeIdx.value()];
      revng_log(TypeMapLog,
                Val.toString() << ";" << T.UnqualifiedType.toString());

      auto NewPair = std::make_pair(Val, T);
      bool New = ValMap.insert(NewPair).second;
      revng_assert(New);
    }
  }

  return ValMap;
}

TypeMapT dla::makeModelTypes(const LayoutTypeSystem &TS,
                             const LayoutTypePtrVect &Values,
                             WritableModelT &Model) {
  logEntry(TS, Model);

  const dla::VectEqClasses &EqClasses = TS.getEqClasses();
  TypeVect Types;
  Types.resize(EqClasses.getNumClasses());
  PtrFieldsMap FieldsToUpdate;

  // Create nodes for anything that is not a pointer
  llvm::SmallPtrSet<const LTSN *, 16> Visited;
  for (const LTSN *Root : llvm::nodes(&TS)) {
    revng_assert(Root != nullptr);
    if (not isRoot(Root))
      continue;

    for (const LTSN *N : llvm::post_order(ConstNonPointerFilterT(Root))) {
      if (Visited.contains(N))
        continue;
      Visited.insert(N);

      QualifiedType *T = getNodeType(N, Types, EqClasses);
      revng_assert(T);
      if (isPointerNode(N)) {
        // Save the qualified type corresponding to this pointer for later
        FieldsToUpdate[N].push_back(T);
        revng_log(Log,
                  "Found root pointer node "
                    << N->ID << ", saved type at index "
                    << EqClasses.getEqClassID(N->ID).value());
        continue;
      }

      if (isLeaf(N)) {
        revng_assert(N->Size);
        // Note: pointers have already been skipped
        T->UnqualifiedType = Model->getPrimitiveType(Number, N->Size);
        if (hasInheritanceParent(N))
          T->UnqualifiedType = createStructWrapper(T->UnqualifiedType,
                                                   /*offset=*/0,
                                                   /*size=*/N->Size,
                                                   Model,
                                                   N->ID);
      } else if (isStructNode(N)) {
        T->UnqualifiedType = makeStructFromNode(N,
                                                Types,
                                                FieldsToUpdate,
                                                Model,
                                                EqClasses);
      } else if (isUnionNode(N)) {
        T->UnqualifiedType = makeUnionFromNode(N,
                                               Types,
                                               FieldsToUpdate,
                                               Model,
                                               EqClasses);
      } else {
        revng_abort("Illegal DLA node encountered when generating model "
                    "types.");
      }

      revng_log(Log,
                "Assigned type " << T->UnqualifiedType.toString()
                                 << " to index "
                                 << EqClasses.getEqClassID(N->ID).value());
    }
  }

  // Fix pointers
  // TODO: possible optimization: explore in bfs the pointer edges backwards
  // from each pointee, and update all the parents of the pointer node
  // encountered. In this way, each tree of pointer edges is visited only once.
  revng_log(Log, "Fixing pointer fields");
  llvm::SmallPtrSet<QualifiedType *, 8> VisitedQTypes;
  for (auto [PointerNode, PointerQTypes] : FieldsToUpdate) {
    QualifiedType FinalQType = makePtrTypeFromNode(PointerNode,
                                                   Types,
                                                   EqClasses);

    for (auto &ID : PointerQTypes) {
      QualifiedType *QTypeToModify = getQTypeFromId(ID, Model);

      auto [_, Inserted] = VisitedQTypes.insert(QTypeToModify);
      if (not Inserted)
        continue;

      revng_log(Log,
                "Updating "
                  << " (" << QTypeToModify->UnqualifiedType.Root << ") "
                  << " with pointer to type "
                  << FinalQType.UnqualifiedType.get()->ID);

      QTypeToModify->UnqualifiedType = FinalQType.UnqualifiedType;
      QTypeToModify->Qualifiers.insert(QTypeToModify->Qualifiers.end(),
                                       FinalQType.Qualifiers.begin(),
                                       FinalQType.Qualifiers.end());
    }
  }

  logExit(TS, Model);

  return mapLLVMValuesToModelTypes(TS, Values, Types);
}