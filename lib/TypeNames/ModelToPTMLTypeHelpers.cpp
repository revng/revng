//
// This file is distributed under the MIT License. See LICENSE.md for details.
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
#include "revng/Model/TypeDefinition.h"
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
using TypeSet = std::set<const model::TypeDefinition *>;
using GraphInfo = TypeInlineHelper::GraphInfo;
using Node = TypeInlineHelper::Node;
using StackTypesMap = std::unordered_map<const model::Function *, TypeSet>;

/// Collect candidates for emitting inline types.
static TypeSet findTypesToInline(const model::Binary &Model) {

  using NumTypeRefMap = std::unordered_map<const model::TypeDefinition *,
                                           uint64_t>;
  NumTypeRefMap NumberOfRefsPerType;
  TypeSet TypesWithBannedReferences;

  for (const UpcastablePointer<model::TypeDefinition> &T :
       Model.TypeDefinitions()) {
    const model::TypeDefinition *TheType = T.get();
    NumberOfRefsPerType.insert({ TheType, 0 });

    bool ParentDeclIsDefinition = declarationIsDefinition(TheType);
    for (const model::QualifiedType &QT : T->edges()) {
      const model::TypeDefinition *Dependency = QT.UnqualifiedType().get();
      NumberOfRefsPerType[Dependency]++;

      // If the parent type has a declaration that is also a definition, we
      // cannot inline it there, since we only allow inlining inside types whose
      // full definition is separate from declaration
      if (ParentDeclIsDefinition)
        TypesWithBannedReferences.insert(Dependency);

      // To inline an array or pointer type, we should basically inline the
      // array element or the pointee type.
      // At the moment we don't try to do this, and just prevent them to be
      // inlined. We might try and do better in the future.
      if (QT.isPointer() or QT.isArray())
        TypesWithBannedReferences.insert(Dependency);
    }
  }

  for (auto &Function : Model.Functions())
    if (not Function.StackFrameType().empty())
      NumberOfRefsPerType[Function.StackFrameType().getConst()]++;

  // TODO: In principle we should do this for segments to, to enable inlining
  // their type definition directly in the declaration of the global variable
  // representing the segment.
  // This is not urgent now though, and it would require more tweaks to
  // ModelToHeader that are low-priority now.
  //
  // for (auto &Segment : Model.Segments())
  //  if (not Segment.Type().empty())
  //    NumberOfRefsPerType[Segment.Type().getConst()]++;

  const auto BanFromInlining =
    [&NumberOfRefsPerType,
     &TypesWithBannedReferences](const model::TypeDefinition *T) {
      // If T's forward declaration cannot be separated by its full definition,
      // ban it from inlining.
      if (declarationIsDefinition(T))
        return true;

      // If T has banned references, ban it from inlining.
      if (TypesWithBannedReferences.contains(T))
        return true;

      // If T has more than 1 other type referring to it, ban it from inlining.
      return NumberOfRefsPerType.at(T) != 1;
    };

  TypeSet Result;
  llvm::for_each(Model.TypeDefinitions(),
                 [&BanFromInlining,
                  &Result](const model::UpcastableTypeDefinition &T) {
                   auto *TheType = T.get();
                   if (not BanFromInlining(TheType))
                     Result.insert(TheType);
                 });

  return Result;
}

static GraphInfo buildTypeGraph(const model::Binary &Model) {

  GraphInfo Result;

  using NodeData = TypeInlineHelper::NodeData;
  for (const UpcastablePointer<model::TypeDefinition> &T :
       Model.TypeDefinitions()) {
    Result.TypeToNode[T.get()] = Result.TypeGraph.addNode(NodeData{ T.get() });
  }

  // Create type system edges.
  for (const model::UpcastableTypeDefinition &T : Model.TypeDefinitions()) {
    for (const model::QualifiedType &QT : T->edges()) {
      auto *UType = QT.UnqualifiedType().get();
      Result.TypeToNode.at(T.get())->addSuccessor(Result.TypeToNode.at(UType));
    }
  }

  return Result;
}

TypeInlineHelper::TypeInlineHelper(const model::Binary &TheModel) :
  Model(TheModel),
  TypeGraph(buildTypeGraph(Model)),
  TypesToInline(findTypesToInline(Model)) {
}

const TypeSet &TypeInlineHelper::getTypesToInline() const {
  return TypesToInline;
}

/// Returns a set of types that are referred to by at least one other type in
/// the \a Model. It does not take into consideration other references to the
/// types that are not cross-references among types, like e.g. stack frame types
/// that refer to model::Types from model::Functions.
static TypeSet getCrossReferencedTypes(const model::Binary &Model) {
  TypeSet Result;

  for (const model::UpcastableTypeDefinition &T : Model.TypeDefinitions())
    for (const model::QualifiedType &QT : T->edges())
      Result.insert(QT.UnqualifiedType().get());

  return Result;
}

StackTypesMap TypeInlineHelper::findTypesToInlineInStacks() const {

  TypeSet CrossReferencedTypes = getCrossReferencedTypes(Model);

  StackTypesMap Result;
  for (auto &Function : Model.Functions()) {
    if (not Function.StackFrameType().empty()) {
      const auto *StackT = Function.StackFrameType().getConst();

      using TDK = model::TypeDefinitionKind::Values;
      revng_assert(StackT and StackT->Kind() == TDK::StructDefinition);

      // Do not inline stack types that are used by at least one other type.
      if (CrossReferencedTypes.contains(StackT))
        continue;

      Result[&Function].insert(StackT);
      auto AllNestedTypes = getTypesToInlineInTypeTy(StackT);
      Result[&Function].merge(AllNestedTypes);
    }
  }

  return Result;
}

TypeSet TypeInlineHelper::collectTypesInlinableInStacks() const {
  StackTypesMap TypesToInlineInStacks = findTypesToInlineInStacks();

  TypeSet Result;
  for (auto [Function, TypesToInlineInStack] : TypesToInlineInStacks)
    Result.merge(std::move(TypesToInlineInStack));

  return Result;
}

using UPtrTy = UpcastablePointer<model::TypeDefinition>;
TypeSet
TypeInlineHelper::getNestedTypesToInline(const model::TypeDefinition *RootType,
                                         const UPtrTy &NestedTy) const {
  model::TypeDefinition *CurrentTy = NestedTy.get();
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

TypeSet TypeInlineHelper::getTypesToInlineInTypeTy(const model::TypeDefinition
                                                     *RootType) const {
  TypeSet Result;
  auto TheTypeToNode = TypeGraph.TypeToNode;

  // Visit all the nodes reachable from RootType.
  llvm::df_iterator_default_set<Node *> Visited;
  for ([[maybe_unused]] Node *N :
       depth_first_ext(TheTypeToNode.at(RootType), Visited))
    ;

  for (auto &Type : Model.TypeDefinitions()) {
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

bool declarationIsDefinition(const model::TypeDefinition *T) {
  return not llvm::isa<model::StructDefinition>(T)
         and not llvm::isa<model::UnionDefinition>(T)
         and not llvm::isa<model::EnumDefinition>(T);
}

static ptml::Tag getTypeKeyword(const model::TypeDefinition &T,
                                const ptml::PTMLCBuilder &B) {

  switch (T.Kind()) {

  case model::TypeDefinitionKind::EnumDefinition: {
    return B.getKeyword(ptml::PTMLCBuilder::Keyword::Enum);
  }

  case model::TypeDefinitionKind::StructDefinition: {
    return B.getKeyword(ptml::PTMLCBuilder::Keyword::Struct);
  }

  case model::TypeDefinitionKind::UnionDefinition: {
    return B.getKeyword(ptml::PTMLCBuilder::Keyword::Union);
  }

  default:
    revng_abort("unexpected type kind");
  }
}

void printForwardDeclaration(const model::TypeDefinition &T,
                             ptml::PTMLIndentedOstream &Header,
                             ptml::PTMLCBuilder &B) {
  if (declarationIsDefinition(&T))
    Header << B.getModelComment(T);

  auto TypeNameReference = B.getLocationReference(T);
  Header << B.getKeyword(ptml::PTMLCBuilder::Keyword::Typedef) << " "
         << getTypeKeyword(T, B) << " " << B.getAttributePacked() << " "
         << TypeNameReference << " " << TypeNameReference << ";\n";
}

static void printDefinition(const model::EnumDefinition &E,
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
                     const model::StructDefinition &S,
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
                            const model::UnionDefinition &U,
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

void printDeclaration(const model::TypedefDefinition &TD,
                      ptml::PTMLIndentedOstream &Header,
                      ptml::PTMLCBuilder &B) {
  if (declarationIsDefinition(&TD))
    Header << B.getModelComment(TD);

  auto Type = B.getLocationDefinition(TD);
  Header << B.getKeyword(ptml::PTMLCBuilder::Keyword::Typedef) << " "
         << getNamedCInstance(TD.UnderlyingType(), Type, B) << ";\n";
}

/// Generate the definition of a new struct type that wraps all the return
/// values of \a F. The name of the struct type is provided by the caller.
static void generateReturnValueWrapper(Logger<> &Log,
                                       const model::RawFunctionDefinition &F,
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
      const model::NamedTypedRegister &RetVal = Group.value();
      const model::QualifiedType &RetTy = Group.value().Type();

      using pipeline::serializedLocation;
      std::string
        ActionLocation = serializedLocation(revng::ranks::ReturnRegister,
                                            F.key(),
                                            RetVal.key());

      std::string
        FieldString = B.tokenTag(RetVal.name(), ptml::c::tokens::Field)
                        .addAttribute(ptml::attributes::ActionContextLocation,
                                      ActionLocation)
                        .serialize();
      Header << getNamedCInstance(RetTy, FieldString, B) << ";\n";
    }
  }

  Header << " " << getReturnTypeName(F, B, true) << ";\n";
}

/// If the function has more than one return value, generate a wrapper struct
/// that contains them.
static void printRawFunctionWrappers(Logger<> &Log,
                                     const model::RawFunctionDefinition *F,
                                     ptml::PTMLIndentedOstream &Header,
                                     ptml::PTMLCBuilder &B,
                                     const model::Binary &Model) {
  if (F->ReturnValues().size() > 1)
    generateReturnValueWrapper(Log, *F, Header, B, Model);

  for (auto &Arg : F->Arguments())
    revng_assert(Arg.Type().isScalar());
}

/// Print a typedef for a RawFunctionDefinition, that can be used when you have
/// a variable that is a pointer to a function.
static void printDeclaration(Logger<> &Log,
                             const model::RawFunctionDefinition &F,
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

/// Generate the definition of a new struct type that wraps \a ArrayType. This
/// is used to wrap array arguments or array return values of CABIFunctionTypes.
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

/// If the return value or any of the arguments is an array, generate a wrapper
/// struct for each of them, if it's not already in the cache.
static void printCABIFunctionWrappers(const model::CABIFunctionDefinition *F,
                                      ptml::PTMLIndentedOstream &Header,
                                      ptml::PTMLCBuilder &B,
                                      QualifiedTypeNameMap &NamesCache) {
  if (F->ReturnType().isArray())
    generateArrayWrapper(F->ReturnType(), Header, B, NamesCache);

  for (auto &Arg : F->Arguments())
    if (Arg.Type().isArray())
      generateArrayWrapper(Arg.Type(), Header, B, NamesCache);
}

/// Print a typedef for a CABIFunctionType, that can be used when you have
/// a variable that is a pointer to a function.
static void printDeclaration(const model::CABIFunctionDefinition &F,
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
                      const model::TypeDefinition &T,
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
    Header << "Declaration of " << getNameFromYAMLScalar(T.key()) << "\n";
  }

  revng_log(Log, "Declaring " << getNameFromYAMLScalar(T.key()));

  switch (T.Kind()) {

  case model::TypeDefinitionKind::Invalid: {
    if (Log.isEnabled())
      Header << B.getLineComment("invalid");
  } break;

  case model::TypeDefinitionKind::PrimitiveDefinition: {
    // Do nothing. Primitive type declarations are all present in
    // revng-primitive-types.h
  } break;

  case model::TypeDefinitionKind::EnumDefinition: {
    printForwardDeclaration(llvm::cast<model::EnumDefinition>(T), Header, B);
  } break;

  case model::TypeDefinitionKind::StructDefinition: {
    printForwardDeclaration(llvm::cast<model::StructDefinition>(T), Header, B);
  } break;

  case model::TypeDefinitionKind::UnionDefinition: {
    printForwardDeclaration(llvm::cast<model::UnionDefinition>(T), Header, B);
  } break;

  case model::TypeDefinitionKind::TypedefDefinition: {
    printDeclaration(llvm::cast<model::TypedefDefinition>(T), Header, B);
  } break;

  case model::TypeDefinitionKind::RawFunctionDefinition: {
    printDeclaration(Log,
                     llvm::cast<model::RawFunctionDefinition>(T),
                     Header,
                     B,
                     Model);
  } break;

  case model::TypeDefinitionKind::CABIFunctionDefinition: {
    printDeclaration(llvm::cast<model::CABIFunctionDefinition>(T),
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
                     const model::TypeDefinition &T,
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

    case model::TypeDefinitionKind::Invalid: {
      if (Log.isEnabled())
        Header << B.getLineComment("invalid");
    } break;

    case model::TypeDefinitionKind::StructDefinition: {
      printDefinition(Log,
                      llvm::cast<model::StructDefinition>(T),
                      Header,
                      B,
                      Model,
                      AdditionalNames,
                      TypesToInline,
                      NameOfInlineInstance,
                      Qualifiers);
    } break;

    case model::TypeDefinitionKind::UnionDefinition: {
      printDefinition(Log,
                      llvm::cast<model::UnionDefinition>(T),
                      Header,
                      B,
                      Model,
                      AdditionalNames,
                      TypesToInline,
                      NameOfInlineInstance,
                      Qualifiers);
    } break;

    case model::TypeDefinitionKind::EnumDefinition: {
      printDefinition(llvm::cast<model::EnumDefinition>(T),
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
