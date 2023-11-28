//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

#include <unordered_map>

#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/ADT/GenericGraph.h"
#include "revng/Model/Binary.h"
#include "revng/Model/Helpers.h"
#include "revng/Model/Type.h"
#include "revng/Pipeline/Location.h"
#include "revng/Support/Assert.h"
#include "revng/Support/Debug.h"
#include "revng/Support/YAMLTraits.h"
#include "revng/Yield/PTML.h"

#include "revng-c/HeadersGeneration/ModelToHeader.h"
#include "revng-c/Pipes/Ranks.h"
#include "revng-c/Support/ModelHelpers.h"
#include "revng-c/Support/PTMLC.h"
#include "revng-c/TypeNames/ModelToPTMLTypeHelpers.h"
#include "revng-c/TypeNames/ModelTypeNames.h"

using QualifiedTypeNameMap = std::map<model::QualifiedType, std::string>;
using TypeSet = std::set<const model::Type *>;
using TypeToNumOfRefsMap = std::unordered_map<const model::Type *, unsigned>;
using GraphInfo = TypeInlineHelper::GraphInfo;
using Node = TypeInlineHelper::Node;
using StackTypesMap = std::unordered_map<const model::Function *,
                                         std::set<const model::Type *>>;

TypeInlineHelper::TypeInlineHelper(const model::Binary &Model) {
  // Create graph that represents type system.
  TypeGraph = buildTypeGraph(Model);
  TypeToNumOfRefs = calculateNumOfOccurences(Model);
  TypesToInline = findTypesToInline(Model, TypeGraph);
}

const GraphInfo &TypeInlineHelper::getTypeGraph() const {
  return TypeGraph;
}
const TypeSet &TypeInlineHelper::getTypesToInline() const {
  return TypesToInline;
}
const TypeToNumOfRefsMap &TypeInlineHelper::getTypeToNumOfRefs() const {
  return TypeToNumOfRefs;
}

/// Collect candidates for emitting inline types.
TypeSet TypeInlineHelper::findTypesToInline(const model::Binary &Model,
                                            const GraphInfo &TypeGraph) {
  std::unordered_map<const model::Type *, uint64_t> Candidates;
  std::set<const model::Type *> ShouldIgnore;

  // We may find a struct that represents stack type that is being used exactly
  // once somewhere else in Types:, but we do not want to inline it if that is
  // the case.
  for (auto &Function : Model.Functions()) {
    if (not Function.StackFrameType().empty()) {
      const model::Type *StackT = Function.StackFrameType().getConst();
      ShouldIgnore.insert(StackT);
    }
  }

  for (const UpcastablePointer<model::Type> &T : Model.Types()) {
    for (const model::QualifiedType &QT : T->edges()) {
      auto *DependantType = QT.UnqualifiedType().get();
      if (llvm::isa<model::RawFunctionType>(T.get())
          or llvm::isa<model::CABIFunctionType>(T.get())
          or llvm::isa<model::TypedefType>(T.get())) {
        // Used as typename.
        ShouldIgnore.insert(DependantType);
      } else if (isCandidateForInline(DependantType)) {
        // If it comes from a Type other than a function, consider that we are
        // interested for the type, or if it was referenced from a type other
        // than itself.
        Candidates[DependantType]++;

        // To inline a pointer type we need to know the sizes of all nested
        // types, which may not be the case at the moment of inlining, so we
        // avoid inlining it for now. In addition, we avoid inlining the types
        // pointing to itself.
        if (QT.isPointer() or T.get()->key() == DependantType->key()) {
          ShouldIgnore.insert(DependantType);
        } else if (isReachableFromRootType(T.get(), DependantType, TypeGraph)) {
          // Or the type could point to itself on a nested level.
          ShouldIgnore.insert(T.get());
          ShouldIgnore.insert(DependantType);
        }
      }
    }
  }

  // A candidate for inline is the type IFF it was referenced only once.
  std::set<const model::Type *> Result;
  using TypeReferences = const pair<const model::Type *, uint64_t>;
  for_each(Candidates.begin(),
           Candidates.end(),
           [&Result, &ShouldIgnore](TypeReferences &TheType) {
             if (TheType.second == 1
                 and not ShouldIgnore.contains(TheType.first)) {
               Result.insert(TheType.first);
             }
           });
  return Result;
}

GraphInfo TypeInlineHelper::buildTypeGraph(const model::Binary &Model) {
  GraphInfo Result;

  for (const UpcastablePointer<model::Type> &T : Model.Types()) {
    Result.TypeToNode[T.get()] = Result.TypeGraph.addNode(NodeData{ T.get() });
  }

  // Create type system edges.
  for (const UpcastablePointer<model::Type> &T : Model.Types()) {
    for (const model::QualifiedType &QT : T->edges()) {
      auto *UType = QT.UnqualifiedType().get();
      Result.TypeToNode.at(T.get())->addSuccessor(Result.TypeToNode.at(UType));
    }
  }

  return Result;
}

TypeToNumOfRefsMap
TypeInlineHelper::calculateNumOfOccurences(const model::Binary &Model) {
  TypeToNumOfRefsMap Result;
  for (const UpcastablePointer<model::Type> &T : Model.Types()) {
    for (const model::QualifiedType &QT : T->edges()) {
      auto *DependantType = QT.UnqualifiedType().get();
      Result[DependantType]++;
    }
  }
  return Result;
}

StackTypesMap
TypeInlineHelper::findStackTypesPerFunction(const model::Binary &Model) const {
  StackTypesMap Result;

  for (auto &Function : Model.Functions()) {
    if (not Function.StackFrameType().empty()) {
      const model::Type *StackT = Function.StackFrameType().getConst();
      // Do not inline stack types that are being used somewhere else.
      auto TheTypeToNumOfRefs = TypeToNumOfRefs.find(StackT);
      if (TheTypeToNumOfRefs != TypeToNumOfRefs.end()
          and TheTypeToNumOfRefs->second != 0)
        continue;

      revng_assert(StackT->Kind() == model::TypeKind::StructType);
      Result[&Function].insert(StackT);
      auto AllNestedTypes = getTypesToInlineInTypeTy(Model, StackT);
      Result[&Function].merge(AllNestedTypes);
    }
  }

  return Result;
}

TypeSet TypeInlineHelper::collectStackTypes(const model::Binary &Model) const {
  TypeSet Result;
  for (auto &Function : Model.Functions()) {
    if (not Function.StackFrameType().empty()) {
      const model::Type *StackT = Function.StackFrameType().getConst();
      revng_assert(StackT->Kind() == model::TypeKind::StructType);

      // Do not inline stack types that are being used somewhere else.
      auto TheTypeToNumOfRefs = TypeToNumOfRefs.find(StackT);
      if (TheTypeToNumOfRefs != TypeToNumOfRefs.end()
          and TheTypeToNumOfRefs->second != 0)
        continue;

      revng_assert(StackT != nullptr);
      Result.insert(StackT);
      auto AllNestedTypes = getTypesToInlineInTypeTy(Model, StackT);
      Result.merge(AllNestedTypes);
    }
  }

  return Result;
}

bool declarationIsDefinition(const model::Type *T) {
  return not llvm::isa<model::StructType>(T)
         and not llvm::isa<model::UnionType>(T)
         and not llvm::isa<model::EnumType>(T);
}

static ptml::Tag getTypeKeyword(const model::Type &T,
                                const ptml::PTMLCBuilder &B) {

  switch (T.Kind()) {

  case model::TypeKind::EnumType: {
    return B.getKeyword(ptml::PTMLCBuilder::Keyword::Enum);
  }

  case model::TypeKind::StructType: {
    return B.getKeyword(ptml::PTMLCBuilder::Keyword::Struct);
  }

  case model::TypeKind::UnionType: {
    return B.getKeyword(ptml::PTMLCBuilder::Keyword::Union);
  }

  default:
    revng_abort("unexpected type kind");
  }
}

void printForwardDeclaration(const model::Type &T,
                             ptml::PTMLIndentedOstream &Header,
                             ptml::PTMLCBuilder &B) {
  if (declarationIsDefinition(&T))
    Header << B.getModelComment(T);

  auto TypeNameReference = B.getLocationReference(T);
  Header << B.getKeyword(ptml::PTMLCBuilder::Keyword::Typedef) << " "
         << getTypeKeyword(T, B) << " " << B.getAttributePacked() << " "
         << TypeNameReference << " " << TypeNameReference << ";\n";
}

static void printDefinition(const model::EnumType &E,
                            ptml::PTMLIndentedOstream &Header,
                            ptml::PTMLCBuilder &B,
                            const TypeSet &TypesToInline,
                            llvm::StringRef NameOfInlineInstance,
                            const std::vector<model::Qualifier> &Qualifiers,
                            bool ForEditing) {
  // We have to make the enum of the correct size of the underlying type
  auto ByteSize = *E.size();
  revng_assert(ByteSize <= 8);
  size_t FullMask = std::numeric_limits<size_t>::max();
  size_t MaxBitPatternInEnum = (ByteSize == 8) ?
                                 FullMask :
                                 ((FullMask) xor (FullMask << (8 * ByteSize)));

  Header
    << B.getModelComment(E) << B.getKeyword(ptml::PTMLCBuilder::Keyword::Enum)
    << " "
    << B.getAnnotateEnum(E.UnderlyingType().UnqualifiedType().get()->name())
    << " " << B.getAttributePacked() << " " << B.getLocationDefinition(E)
    << " ";

  {
    Scope Scope(Header);

    using PTMLOperator = ptml::PTMLCBuilder::Operator;
    for (const auto &Entry : E.Entries()) {
      Header << B.getModelComment(Entry) << B.getLocationDefinition(E, Entry)
             << " " << B.getOperator(PTMLOperator::Assign) << " "
             << B.getHex(Entry.Value()) << ",\n";
    }

    if (not ForEditing) {
      // This ensures the enum is large exactly like the Underlying type
      Header << B.tokenTag(("_enum_max_value_" + E.name()).str(),
                           ptml::c::tokens::Field)
             << " " + B.getOperator(PTMLOperator::Assign) + " "
             << B.getHex(MaxBitPatternInEnum) << ",\n";
    }
  }

  if (not NameOfInlineInstance.empty())
    Header << " " << NameOfInlineInstance << ";\n";
  else
    Header << ";\n";
}

void printDefinition(Logger<> &Log,
                     const model::StructType &S,
                     ptml::PTMLIndentedOstream &Header,
                     ptml::PTMLCBuilder &B,
                     const model::Binary &Model,
                     QualifiedTypeNameMap &AdditionalNames,
                     const TypeSet &TypesToInline,
                     llvm::StringRef NameOfInlineInstance,
                     const std::vector<model::Qualifier> &Qualifiers) {

  Header << B.getModelComment(S)
         << B.getKeyword(ptml::PTMLCBuilder::Keyword::Struct) << " "
         << B.getAttributePacked() << " ";
  Header << B.getLocationDefinition(S) << " ";
  {
    Scope Scope(Header, ptml::c::scopes::StructBody);

    size_t NextOffset = 0ULL;
    for (const auto &Field : S.Fields()) {
      if (NextOffset < Field.Offset()) {
        Header << B.tokenTag("uint8_t", ptml::c::tokens::Type) << " "
               << B.tokenTag(StructPaddingPrefix + std::to_string(NextOffset),
                             ptml::c::tokens::Field)
               << "[" << B.getNumber(Field.Offset() - NextOffset) << "];\n";
      }

      auto TheType = Field.Type().UnqualifiedType().get();
      if (not TypesToInline.contains(TheType)) {
        auto F = B.getLocationDefinition(S, Field);
        Header << B.getModelComment(Field)
               << getNamedCInstance(Field.Type(), F, B) << ";\n";
      } else {
        auto Qualifiers = Field.Type().Qualifiers();
        printDefinition(Log,
                        *TheType,
                        Header,
                        B,
                        Model,
                        AdditionalNames,
                        TypesToInline,
                        Field.name().str(),
                        Qualifiers);
      }

      NextOffset = Field.Offset() + Field.Type().size().value();
    }

    if (NextOffset < S.Size())
      Header << B.tokenTag("uint8_t", ptml::c::tokens::Type) << " "
             << B.tokenTag(StructPaddingPrefix + std::to_string(NextOffset),
                           ptml::c::tokens::Field)
             << "[" << B.getNumber(S.Size() - NextOffset) << "];\n";
  }
  if (not NameOfInlineInstance.empty()) {
    if (Qualifiers.empty())
      Header << " " << NameOfInlineInstance;
    else
      Header << getNamedCInstance("", Qualifiers, NameOfInlineInstance, B);
  }
  Header << ";\n";
}

static void printDefinition(Logger<> &Log,
                            const model::UnionType &U,
                            ptml::PTMLIndentedOstream &Header,
                            ptml::PTMLCBuilder &B,
                            const model::Binary &Model,
                            QualifiedTypeNameMap &AdditionalTypeNames,
                            const TypeSet &TypesToInline,
                            llvm::StringRef NameOfInlineInstance,
                            const std::vector<model::Qualifier> &Qualifiers) {
  Header << B.getModelComment(U)
         << B.getKeyword(ptml::PTMLCBuilder::Keyword::Union) << " "
         << B.getAttributePacked() << " ";
  Header << B.getLocationDefinition(U) << " ";

  {
    Scope Scope(Header, ptml::c::scopes::UnionBody);
    for (const auto &Field : U.Fields()) {
      auto TheType = Field.Type().UnqualifiedType().get();
      if (not TypesToInline.contains(TheType)) {
        auto F = B.getLocationDefinition(U, Field);
        Header << B.getModelComment(Field)
               << getNamedCInstance(Field.Type(), F, B) << ";\n";
      } else {
        std::string Name = Field.name().str().str();

        auto Qualifiers = Field.Type().Qualifiers();
        printDefinition(Log,
                        *TheType,
                        Header,
                        B,
                        Model,
                        AdditionalTypeNames,
                        TypesToInline,
                        llvm::StringRef(Name.c_str()),
                        Qualifiers);
      }
    }
  }

  if (not NameOfInlineInstance.empty()) {
    if (Qualifiers.empty())
      Header << " " << NameOfInlineInstance;
    else
      Header << getNamedCInstance("", Qualifiers, NameOfInlineInstance, B);
  }
  Header << ";\n";
}

void printDeclaration(const model::TypedefType &TD,
                      ptml::PTMLIndentedOstream &Header,
                      ptml::PTMLCBuilder &B) {
  if (declarationIsDefinition(&TD))
    Header << B.getModelComment(TD);

  auto Type = B.getLocationDefinition(TD);
  Header << B.getKeyword(ptml::PTMLCBuilder::Keyword::Typedef) << " "
         << getNamedCInstance(TD.UnderlyingType(), Type, B) << ";\n";
}

/// Generate the definition of a new struct type that wraps all the
///        return values of \a F. The name of the struct type is provided by the
///        caller.
static void generateReturnValueWrapper(Logger<> &Log,
                                       const model::RawFunctionType &F,
                                       ptml::PTMLIndentedOstream &Header,
                                       ptml::PTMLCBuilder &B,
                                       const model::Binary &Model) {
  revng_assert(F.ReturnValues().size() > 1);
  if (Log.isEnabled())
    Header << B.getLineComment("definition the of return type "
                               "needed");

  Header << B.getKeyword(ptml::PTMLCBuilder::Keyword::Typedef) << " "
         << B.getKeyword(ptml::PTMLCBuilder::Keyword::Struct) << " "
         << B.getAttributePacked() << " ";

  {
    Scope Scope(Header, ptml::c::scopes::StructBody);
    for (auto &Group : llvm::enumerate(F.ReturnValues())) {
      const model::QualifiedType &RetTy = Group.value().Type();
      Header << getNamedCInstance(RetTy,
                                  B.tokenTag(Group.value().name(),
                                             ptml::c::tokens::Field)
                                    .serialize(),
                                  B)
             << ";\n";
    }
  }

  Header << " " << getReturnTypeName(F, B, true) << ";\n";
}

/// If the function has more than one return value, generate a wrapper
///        struct that contains them.
static void printRawFunctionWrappers(Logger<> &Log,
                                     const model::RawFunctionType *F,
                                     ptml::PTMLIndentedOstream &Header,
                                     ptml::PTMLCBuilder &B,
                                     const model::Binary &Model) {
  if (F->ReturnValues().size() > 1)
    generateReturnValueWrapper(Log, *F, Header, B, Model);

  for (auto &Arg : F->Arguments())
    revng_assert(Arg.Type().isScalar());
}

/// Print a typedef for a RawFunctionType, that can be used when you have
///        a variable that is a pointer to a function.
static void printDeclaration(Logger<> &Log,
                             const model::RawFunctionType &F,
                             ptml::PTMLIndentedOstream &Header,
                             ptml::PTMLCBuilder &B,
                             const model::Binary &Model) {
  printRawFunctionWrappers(Log, &F, Header, B, Model);

  Header << B.getModelComment(F)
         << B.getKeyword(ptml::PTMLCBuilder::Keyword::Typedef) << " ";
  // In this case, we are defining a type for the function, not the function
  // itself, so the token right before the parenthesis is the name of the type.
  printFunctionTypeDeclaration(F, Header, B, Model);
  Header << ";\n";
}

/// Generate the definition of a new struct type that wraps \a ArrayType.
///        This is used to wrap array arguments or array return values of
///        CABIFunctionTypes.
static void generateArrayWrapper(const model::QualifiedType &ArrayType,
                                 ptml::PTMLIndentedOstream &Header,
                                 ptml::PTMLCBuilder &B,
                                 QualifiedTypeNameMap &NamesCache) {
  revng_assert(ArrayType.isArray());
  auto WrapperName = getArrayWrapper(ArrayType, B);

  // Check if the wrapper was already added
  bool IsNew = NamesCache.emplace(ArrayType, WrapperName).second;
  if (not IsNew)
    return;

  Header << B.getKeyword(ptml::PTMLCBuilder::Keyword::Typedef) << " "
         << B.getKeyword(ptml::PTMLCBuilder::Keyword::Struct) << " "
         << B.getAttributePacked() << " ";
  {
    Scope Scope(Header, ptml::c::scopes::StructBody);
    Header << getNamedCInstance(ArrayType,
                                ArtificialTypes::ArrayWrapperFieldName,
                                B)
           << ";\n";
  }
  Header << " " << B.tokenTag(WrapperName, ptml::c::tokens::Type) << ";\n";
}

/// If the return value or any of the arguments is an array, generate
///        a wrapper struct for each of them, if it's not already in the cache.
static void printCABIFunctionWrappers(const model::CABIFunctionType *F,
                                      ptml::PTMLIndentedOstream &Header,
                                      ptml::PTMLCBuilder &B,
                                      QualifiedTypeNameMap &NamesCache) {
  if (F->ReturnType().isArray())
    generateArrayWrapper(F->ReturnType(), Header, B, NamesCache);

  for (auto &Arg : F->Arguments())
    if (Arg.Type().isArray())
      generateArrayWrapper(Arg.Type(), Header, B, NamesCache);
}

/// Print a typedef for a CABIFunctionType, that can be used when you
///        have a variable that is a pointer to a function.
static void printDeclaration(const model::CABIFunctionType &F,
                             ptml::PTMLIndentedOstream &Header,
                             ptml::PTMLCBuilder &B,
                             QualifiedTypeNameMap &NamesCache,
                             const model::Binary &Model) {
  printCABIFunctionWrappers(&F, Header, B, NamesCache);

  Header << B.getModelComment(F)
         << B.getKeyword(ptml::PTMLCBuilder::Keyword::Typedef) << " ";
  // In this case, we are defining a type for the function, not the function
  // itself, so the token right before the parenthesis is the name of the type.
  printFunctionTypeDeclaration(F, Header, B, Model);
  Header << ";\n";
}

void printDeclaration(Logger<> &Log,
                      const model::Type &T,
                      ptml::PTMLIndentedOstream &Header,
                      ptml::PTMLCBuilder &B,
                      const model::Binary &Model,
                      QualifiedTypeNameMap &AdditionalNames,
                      const TypeSet &TypesToInline,
                      llvm::StringRef NameOfInlineInstance,
                      const std::vector<model::Qualifier> &Qualifiers,
                      bool ForEditing) {
  if (Log.isEnabled()) {
    auto Scope = helpers::LineComment(Header, B.isGenerateTagLessPTML());
    Header << "Declaration of " << getNameFromYAMLScalar(T.key());
  }

  revng_log(Log, "Declaring " << getNameFromYAMLScalar(T.key()));

  switch (T.Kind()) {

  case model::TypeKind::Invalid: {
    if (Log.isEnabled())
      Header << B.getLineComment("invalid");
  } break;

  case model::TypeKind::PrimitiveType: {
    // Do nothing. Primitive type declarations are all present in
    // revng-primitive-types.h
  } break;

  case model::TypeKind::EnumType: {
    printForwardDeclaration(llvm::cast<model::EnumType>(T), Header, B);
  } break;

  case model::TypeKind::StructType: {
    printForwardDeclaration(llvm::cast<model::StructType>(T), Header, B);
  } break;

  case model::TypeKind::UnionType: {
    printForwardDeclaration(llvm::cast<model::UnionType>(T), Header, B);
  } break;

  case model::TypeKind::TypedefType: {
    printDeclaration(llvm::cast<model::TypedefType>(T), Header, B);
  } break;

  case model::TypeKind::RawFunctionType: {
    printDeclaration(Log,
                     llvm::cast<model::RawFunctionType>(T),
                     Header,
                     B,
                     Model);
  } break;

  case model::TypeKind::CABIFunctionType: {
    printDeclaration(llvm::cast<model::CABIFunctionType>(T),
                     Header,
                     B,
                     AdditionalNames,
                     Model);
  } break;
  default:
    revng_abort();
  }
}

void printDefinition(Logger<> &Log,
                     const model::Type &T,
                     ptml::PTMLIndentedOstream &Header,
                     ptml::PTMLCBuilder &B,
                     const model::Binary &Model,
                     QualifiedTypeNameMap &AdditionalNames,
                     const TypeSet &TypesToInline,
                     llvm::StringRef NameOfInlineInstance,
                     const std::vector<model::Qualifier> &Qualifiers,
                     bool ForEditing) {
  if (Log.isEnabled())
    Header << B.getLineComment("Definition of "
                               + getNameFromYAMLScalar(T.key()));

  revng_log(Log, "Defining " << getNameFromYAMLScalar(T.key()));
  if (declarationIsDefinition(&T)) {
    printDeclaration(Log,
                     T,
                     Header,
                     B,
                     Model,
                     AdditionalNames,
                     TypesToInline,
                     NameOfInlineInstance,
                     Qualifiers,
                     ForEditing);
  } else {
    switch (T.Kind()) {

    case model::TypeKind::Invalid: {
      if (Log.isEnabled())
        Header << B.getLineComment("invalid");
    } break;

    case model::TypeKind::StructType: {
      printDefinition(Log,
                      llvm::cast<model::StructType>(T),
                      Header,
                      B,
                      Model,
                      AdditionalNames,
                      TypesToInline,
                      NameOfInlineInstance,
                      Qualifiers);
    } break;

    case model::TypeKind::UnionType: {
      printDefinition(Log,
                      llvm::cast<model::UnionType>(T),
                      Header,
                      B,
                      Model,
                      AdditionalNames,
                      TypesToInline,
                      NameOfInlineInstance,
                      Qualifiers);
    } break;

    case model::TypeKind::EnumType: {
      printDefinition(llvm::cast<model::EnumType>(T),
                      Header,
                      B,
                      TypesToInline,
                      NameOfInlineInstance,
                      Qualifiers,
                      ForEditing);
    } break;

    default:
      revng_abort();
    }
  }
}

bool isCandidateForInline(const model::Type *T) {
  return llvm::isa<model::StructType>(T) or llvm::isa<model::UnionType>(T)
         or llvm::isa<model::EnumType>(T);
}

bool TypeInlineHelper::isReachableFromRootType(const model::Type *Type,
                                               const model::Type *RootType,
                                               const GraphInfo &TypeGraph) {
  auto TheTypeToNode = TypeGraph.TypeToNode;

  // Visit all the nodes reachable from RootType.
  llvm::df_iterator_default_set<Node *> Visited;
  for ([[maybe_unused]] Node *N :
       depth_first_ext(TheTypeToNode.at(RootType), Visited))
    ;

  return Visited.contains(TheTypeToNode.at(Type));
}

using UPtrTy = UpcastablePointer<model::Type>;
TypeSet TypeInlineHelper::getNestedTypesToInline(const model::Type *RootType,
                                                 const UPtrTy &NestedTy) const {
  model::Type *CurrentTy = NestedTy.get();
  TypeSet Result;
  do {
    Result.insert(CurrentTy);
    auto
      ParentNode = TypeGraph.TypeToNode.at(CurrentTy)->predecessors().begin();
    if ((*ParentNode)->data().T == RootType) {
      return Result;
    } else if (TypesToInline.contains((*ParentNode)->data().T)) {
      CurrentTy = (*ParentNode)->data().T;
    } else {
      return {};
    }
  } while (CurrentTy);

  return {};
}

TypeSet
TypeInlineHelper::getTypesToInlineInTypeTy(const model::Binary &Model,
                                           const model::Type *RootType) const {
  TypeSet Result;
  auto TheTypeToNode = TypeGraph.TypeToNode;

  // Visit all the nodes reachable from RootType.
  llvm::df_iterator_default_set<Node *> Visited;
  for ([[maybe_unused]] Node *N :
       depth_first_ext(TheTypeToNode.at(RootType), Visited))
    ;

  for (auto &Type : Model.Types()) {
    if (Visited.contains(TheTypeToNode.at(Type.get()))
        and TypesToInline.contains(Type.get())
        and TheTypeToNode.at(Type.get())->predecessorCount() == 1) {
      auto ParentNode = TheTypeToNode.at(Type.get())->predecessors().begin();
      // In the case the parent is stack type itself, just insert the type.
      if ((*ParentNode)->data().T == RootType) {
        Result.insert(Type.get());
      } else if (TypesToInline.contains((*ParentNode)->data().T)) {
        // In the case the parent type is not the type RootType itself, make
        // sure that the parent is inlinable into the type RootType. NOTE: This
        // goes as further as possible in opposite direction in order to find
        // all types that we should inline into the type RootType.
        auto NestedTypesToInline = getNestedTypesToInline(RootType, Type);
        Result.merge(NestedTypesToInline);
      }
    }
  }

  return Result;
}
