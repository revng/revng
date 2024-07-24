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
#include "revng/ADT/STLExtras.h"
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

using TypeNameMap = std::map<model::UpcastableType, std::string>;
using DefinitionSet = std::set<const model::TypeDefinition *>;
using GraphInfo = TypeInlineHelper::GraphInfo;
using Node = TypeInlineHelper::Node;
using StackTypesMap = std::unordered_map<const model::Function *,
                                         DefinitionSet>;

/// Collect candidates for emitting inline types.
static DefinitionSet findTypesToInline(const model::Binary &Model) {
  using NumTypeRefMap = std::unordered_map<const model::TypeDefinition *,
                                           uint64_t>;
  NumTypeRefMap NumberOfRefsPerType;
  DefinitionSet TypesWithBannedReferences;

  for (const model::UpcastableTypeDefinition &T : Model.TypeDefinitions()) {
    NumberOfRefsPerType.insert({ T.get(), 0 });

    bool ParentDeclarationIsDefinition = declarationIsDefinition(*T);
    for (const model::Type *Edge : T->edges()) {
      const model::TypeDefinition *Dependency = Edge->skipToDefinition();
      if (Dependency == nullptr)
        continue; // Skip types without definitions (only primitives as of now)

      NumberOfRefsPerType[Dependency]++;

      // If the parent type has a declaration that is also a definition, we
      // cannot inline it there, since we only allow inlining inside types whose
      // full definition is separate from declaration
      if (ParentDeclarationIsDefinition)
        TypesWithBannedReferences.insert(Dependency);

      // To inline an array or pointer type, we should basically inline the
      // array element or the pointee type.
      // At the moment we don't try to do this, and just prevent them to be
      // inlined. We might try and do better in the future.
      if (Edge->isPointer() or Edge->isArray())
        TypesWithBannedReferences.insert(Dependency);
    }
  }

  for (const model::Function &Function : Model.Functions())
    if (const model::StructDefinition *Stack = Function.stackFrameType())
      NumberOfRefsPerType[Stack]++;

  // TODO: In principle we should do this for segments to, to enable inlining
  // their type definition directly in the declaration of the global variable
  // representing the segment.
  // This is not urgent now though, and it would require more tweaks to
  // ModelToHeader that are low-priority now.
  //
  // for (const model::Segment &Segment : Model.Segments())
  //  if (const model::StructDefinition *Type = Function.type())
  //    NumberOfRefsPerType[Type]++;

  // A candidate for inline is the type IFF it was referenced only once.
  auto F = std::views::filter([&TypesWithBannedReferences](const auto &Pair) {
    auto [Definition, ReferenceCount] = Pair;
    return ReferenceCount == 1
           and not TypesWithBannedReferences.contains(Definition)
           and not declarationIsDefinition(*Definition);
  });
  return NumberOfRefsPerType | F | std::views::keys
         | revng::to<DefinitionSet>();
}

static GraphInfo buildTypeGraph(const model::Binary &Model) {

  GraphInfo Result;

  using NodeData = TypeInlineHelper::NodeData;
  for (const UpcastablePointer<model::TypeDefinition> &T :
       Model.TypeDefinitions()) {
    Result.TypeToNode[T.get()] = Result.TypeGraph.addNode(NodeData{ T.get() });
  }

  // Create type system edges.
  for (const model::UpcastableTypeDefinition &T : Model.TypeDefinitions())
    for (const model::Type *Edge : T->edges())
      if (const model::TypeDefinition *Def = Edge->skipToDefinition())
        Result.TypeToNode.at(T.get())->addSuccessor(Result.TypeToNode.at(Def));

  return Result;
}

TypeInlineHelper::TypeInlineHelper(const model::Binary &TheModel) :
  Model(TheModel),
  TypeGraph(buildTypeGraph(Model)),
  TypesToInline(findTypesToInline(Model)) {
}

const DefinitionSet &TypeInlineHelper::getTypesToInline() const {
  return TypesToInline;
}

/// Returns a set of types that are referred to by at least one other type in
/// the \a Model. It does not take into consideration other references to the
/// types that are not cross-references among types, like e.g. stack frame types
/// that refer to model::Types from model::Functions.
static DefinitionSet getCrossReferencedTypes(const model::Binary &Model) {
  DefinitionSet Result;

  for (const model::UpcastableTypeDefinition &T : Model.TypeDefinitions())
    for (const model::Type *Edge : T->edges())
      Result.insert(Edge->skipToDefinition());

  return Result;
}

StackTypesMap TypeInlineHelper::findTypesToInlineInStacks() const {
  StackTypesMap Result;

  DefinitionSet CrossReferencedTypes = getCrossReferencedTypes(Model);
  for (auto &Function : Model.Functions()) {
    if (const model::StructDefinition *Stack = Function.stackFrameType()) {
      // Do not inline stack types that are used by at least one other type.
      if (CrossReferencedTypes.contains(Stack))
        continue;

      Result[&Function].insert(Stack);
      auto AllNestedTypes = getTypesToInlineInTypeTy(*Stack);
      Result[&Function].merge(AllNestedTypes);
    }
  }

  return Result;
}

DefinitionSet TypeInlineHelper::collectTypesInlinableInStacks() const {
  DefinitionSet Result;

  for (auto [Function, TypesToInlineInStack] : findTypesToInlineInStacks())
    Result.merge(std::move(TypesToInlineInStack));

  return Result;
}

using TI = TypeInlineHelper;
DefinitionSet
TI::getNestedTypesToInline(const model::TypeDefinition &RootType,
                           const model::TypeDefinition &Nested) const {
  const model::TypeDefinition *Current = &Nested;
  DefinitionSet Result;
  do {
    Result.insert(Current);
    auto ParentNode = TypeGraph.TypeToNode.at(Current)->predecessors().begin();
    if ((*ParentNode)->data().T == &RootType) {
      return Result;
    } else if (TypesToInline.contains((*ParentNode)->data().T)) {
      Current = (*ParentNode)->data().T;
    } else {
      return {};
    }
  } while (Current != nullptr);

  return {};
}

DefinitionSet
TI::getTypesToInlineInTypeTy(const model::TypeDefinition &RootType) const {
  DefinitionSet Result;
  auto TheTypeToNode = TypeGraph.TypeToNode;

  // Visit all the nodes reachable from RootType.
  llvm::df_iterator_default_set<Node *> Visited;
  for ([[maybe_unused]] Node *N :
       depth_first_ext(TheTypeToNode.at(&RootType), Visited))
    ;

  for (const auto &Type : Model.TypeDefinitions()) {
    if (Visited.contains(TheTypeToNode.at(Type.get()))
        and TypesToInline.contains(Type.get())
        and TheTypeToNode.at(Type.get())->predecessorCount() == 1) {
      auto &ParentNode = *TheTypeToNode.at(Type.get())->predecessors().begin();
      // In the case the parent is stack type itself, just insert the type.
      if (ParentNode->data().T == &RootType) {
        Result.insert(Type.get());
      } else if (TypesToInline.contains(ParentNode->data().T)) {
        // In the case the parent type is not the type RootType itself, make
        // sure that the parent is inlinable into the type RootType. NOTE: This
        // goes as further as possible in opposite direction in order to find
        // all types that we should inline into the type RootType.
        auto NestedTypesToInline = getNestedTypesToInline(RootType, *Type);
        Result.merge(NestedTypesToInline);
      }
    }
  }

  return Result;
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
  revng_assert(not declarationIsDefinition(T));

  auto TypeNameReference = B.getLocationReference(T);
  Header << B.getKeyword(ptml::PTMLCBuilder::Keyword::Typedef) << " "
         << getTypeKeyword(T, B) << " " << B.getPackedAttribute() << " "
         << TypeNameReference << " " << TypeNameReference << ";\n";
}

static void printDefinition(const model::EnumDefinition &E,
                            ptml::PTMLIndentedOstream &Header,
                            ptml::PTMLCBuilder &B,
                            bool ForEditing,
                            std::string &&Suffix = "") {
  // We have to make the enum of the correct size of the underlying type
  auto ByteSize = *E.size();
  revng_assert(ByteSize <= 8);
  size_t FullMask = std::numeric_limits<size_t>::max();
  size_t MaxBitPatternInEnum = (ByteSize == 8) ?
                                 FullMask :
                                 ((FullMask) xor (FullMask << (8 * ByteSize)));

  Header << B.getModelComment(E)
         << B.getKeyword(ptml::PTMLCBuilder::Keyword::Enum) << " "
         << B.getEnumAnnotation(E.underlyingType().getCName()) << " "
         << B.getPackedAttribute() << " " << B.getLocationDefinition(E) << " ";

  {
    Scope Scope(Header);

    using PTMLOperator = ptml::PTMLCBuilder::Operator;
    for (const auto &Entry : E.Entries()) {
      Header << B.getModelComment(Entry) << B.getLocationDefinition(E, Entry)
             << " " << B.getOperator(PTMLOperator::Assign) << " "
             << B.getHex(Entry.Value()) << ",\n";
    }

    if (not ForEditing) {
      // This ensures the enum is exactly as large as the Underlying type
      Header << B.tokenTag(("_enum_max_value_" + E.name()).str(),
                           ptml::c::tokens::Field)
             << " " + B.getOperator(PTMLOperator::Assign) + " "
             << B.getHex(MaxBitPatternInEnum) << ",\n";
    }
  }

  Header << std::move(Suffix) << ";\n";
}

static void printPadding(ptml::PTMLIndentedOstream &Header,
                         ptml::PTMLCBuilder &B,
                         uint64_t FieldOffset,
                         uint64_t NextOffset,
                         bool ForEditing) {
  revng_assert(FieldOffset <= NextOffset);
  if (FieldOffset == NextOffset)
    return; // No padding is needed

  if (ForEditing) {
    Header << B.getStartAtAnnotation(NextOffset) << "\n";
  } else {
    Header << B.tokenTag("uint8_t", ptml::c::tokens::Type) << " "
           << B.tokenTag(StructPaddingPrefix + std::to_string(FieldOffset),
                         ptml::c::tokens::Field)
           << "[" << B.getNumber(NextOffset - FieldOffset) << "];\n";
  }
}

static void printDefinition(Logger<> &Log,
                            const model::StructDefinition &S,
                            ptml::PTMLIndentedOstream &Header,
                            ptml::PTMLCBuilder &B,
                            const model::Binary &Model,
                            TypeNameMap &AdditionalNames,
                            const DefinitionSet &TypesToInline,
                            bool ForEditing,
                            std::string &&Suffix = "") {

  Header << B.getModelComment(S)
         << B.getKeyword(ptml::PTMLCBuilder::Keyword::Struct) << " "
         << B.getPackedAttribute() << " ";

  if (S.CanContainCode())
    Header << B.getCanContainCodeAttribute() << " ";

  if (ForEditing)
    Header << B.getSizeAnnotation(S.Size()) << " ";

  Header << B.getLocationDefinition(S) << " ";

  {
    Scope Scope(Header, ptml::c::scopes::StructBody);

    size_t PreviousOffset = 0ULL;
    for (const auto &Field : S.Fields()) {
      printPadding(Header, B, PreviousOffset, Field.Offset(), ForEditing);

      auto *MaybeDefinition = Field.Type()->skipToDefinition();
      if (not MaybeDefinition or not TypesToInline.contains(MaybeDefinition)) {
        auto F = B.getLocationDefinition(S, Field);
        Header << B.getModelComment(Field)
               << getNamedCInstance(*Field.Type(), F, B) << ";\n";
      } else {
        printInlineDefinition(Log,
                              Field.name().str(),
                              *Field.Type(),
                              Header,
                              B,
                              Model,
                              AdditionalNames,
                              TypesToInline);
      }

      PreviousOffset = Field.Offset() + Field.Type()->size().value();
    }

    if (!ForEditing)
      printPadding(Header, B, PreviousOffset, S.Size(), false);
  }

  Header << std::move(Suffix) << ";\n";
}

static void printDefinition(Logger<> &Log,
                            const model::UnionDefinition &U,
                            ptml::PTMLIndentedOstream &Header,
                            ptml::PTMLCBuilder &B,
                            const model::Binary &Model,
                            TypeNameMap &AdditionalTypeNames,
                            const DefinitionSet &TypesToInline,
                            std::string &&Suffix = "") {
  Header << B.getModelComment(U)
         << B.getKeyword(ptml::PTMLCBuilder::Keyword::Union) << " "
         << B.getPackedAttribute() << " ";
  Header << B.getLocationDefinition(U) << " ";

  {
    Scope Scope(Header, ptml::c::scopes::UnionBody);
    for (const auto &Field : U.Fields()) {
      auto *MaybeDefinition = Field.Type()->skipToDefinition();
      if (not MaybeDefinition or not TypesToInline.contains(MaybeDefinition)) {
        auto F = B.getLocationDefinition(U, Field);
        Header << B.getModelComment(Field)
               << getNamedCInstance(*Field.Type(), F, B) << ";\n";
      } else {
        printInlineDefinition(Log,
                              Field.name().str(),
                              *Field.Type(),
                              Header,
                              B,
                              Model,
                              AdditionalTypeNames,
                              TypesToInline);
      }
    }
  }

  Header << std::move(Suffix) << ";\n";
}

void printDeclaration(const model::TypedefDefinition &TD,
                      ptml::PTMLIndentedOstream &Header,
                      ptml::PTMLCBuilder &B) {
  if (declarationIsDefinition(TD))
    Header << B.getModelComment(TD);

  auto Type = B.getLocationDefinition(TD);
  Header << B.getKeyword(ptml::PTMLCBuilder::Keyword::Typedef) << " "
         << getNamedCInstance(*TD.UnderlyingType(), Type, B) << ";\n";
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
         << B.getPackedAttribute() << " ";

  {
    Scope Scope(Header, ptml::c::scopes::StructBody);
    for (auto &[Index, ReturnValue] : llvm::enumerate(F.ReturnValues())) {
      using pipeline::serializedLocation;
      std::string
        ActionLocation = serializedLocation(revng::ranks::ReturnRegister,
                                            F.key(),
                                            ReturnValue.key());

      std::string
        FieldString = B.tokenTag(ReturnValue.name(), ptml::c::tokens::Field)
                        .addAttribute(ptml::attributes::ActionContextLocation,
                                      ActionLocation)
                        .serialize();
      Header << getNamedCInstance(*ReturnValue.Type(), FieldString, B) << ";\n";
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
    revng_assert(Arg.Type()->isScalar());
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

/// Generate the definition of a new struct type that wraps \a ArrayType.
/// This is used to wrap array arguments or array return values of
/// CABI functions.
static void generateArrayWrapper(const model::ArrayType &ArrayType,
                                 ptml::PTMLIndentedOstream &Header,
                                 ptml::PTMLCBuilder &B,
                                 TypeNameMap &NamesCache) {
  auto WrapperName = getArrayWrapper(ArrayType, B);

  // Check if the wrapper was already added
  auto [_, IsNew] = NamesCache.emplace(ArrayType, WrapperName);
  if (not IsNew)
    return;

  Header << B.getKeyword(ptml::PTMLCBuilder::Keyword::Typedef) << " "
         << B.getKeyword(ptml::PTMLCBuilder::Keyword::Struct) << " "
         << B.getPackedAttribute() << " ";
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
                                      TypeNameMap &NamesCache) {
  if (not F->ReturnType().isEmpty())
    if (auto *Array = F->ReturnType()->getArray())
      generateArrayWrapper(*Array, Header, B, NamesCache);

  for (auto &Arg : F->Arguments())
    if (auto *Array = Arg.Type()->getArray())
      generateArrayWrapper(*Array, Header, B, NamesCache);
}

/// Print a typedef for a CABI function, that can be used when you have
/// a variable that is a pointer to a function.
static void printDeclaration(const model::CABIFunctionDefinition &F,
                             ptml::PTMLIndentedOstream &Header,
                             ptml::PTMLCBuilder &B,
                             TypeNameMap &NamesCache,
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
                      TypeNameMap &AdditionalNames) {
  if (Log.isEnabled()) {
    auto Scope = helpers::LineComment(Header, B.isGenerateTagLessPTML());
    Header << "Declaration of " << getNameFromYAMLScalar(T.key()) << "\n";
  }

  revng_log(Log, "Declaring " << getNameFromYAMLScalar(T.key()));

  if (auto *Enum = llvm::dyn_cast<model::EnumDefinition>(&T))
    printForwardDeclaration(*Enum, Header, B);

  else if (auto *Struct = llvm::dyn_cast<model::StructDefinition>(&T))
    printForwardDeclaration(*Struct, Header, B);

  else if (auto *Union = llvm::dyn_cast<model::UnionDefinition>(&T))
    printForwardDeclaration(*Union, Header, B);

  else if (auto *Typedef = llvm::dyn_cast<model::TypedefDefinition>(&T))
    printDeclaration(*Typedef, Header, B);

  else if (auto *RFD = llvm::dyn_cast<model::RawFunctionDefinition>(&T))
    printDeclaration(Log, *RFD, Header, B, Model);

  else if (auto *CFD = llvm::dyn_cast<model::CABIFunctionDefinition>(&T))
    printDeclaration(*CFD, Header, B, AdditionalNames, Model);

  else
    revng_abort("Unsupported type definition.");
}

void printDefinition(Logger<> &Log,
                     const model::TypeDefinition &T,
                     ptml::PTMLIndentedOstream &Header,
                     ptml::PTMLCBuilder &B,
                     const model::Binary &Model,
                     TypeNameMap &AdditionalNames,
                     const DefinitionSet &TypesToInline,
                     bool ForEditing) {
  if (Log.isEnabled())
    Header << B.getLineComment("Definition of "
                               + getNameFromYAMLScalar(T.key()));

  revng_log(Log, "Defining " << getNameFromYAMLScalar(T.key()));
  if (declarationIsDefinition(T)) {
    printDeclaration(Log, T, Header, B, Model, AdditionalNames);
  } else if (auto *Struct = llvm::dyn_cast<model::StructDefinition>(&T)) {
    printDefinition(Log,
                    *Struct,
                    Header,
                    B,
                    Model,
                    AdditionalNames,
                    TypesToInline,
                    ForEditing);
  } else if (auto *Union = llvm::dyn_cast<model::UnionDefinition>(&T)) {
    printDefinition(Log,
                    *Union,
                    Header,
                    B,
                    Model,
                    AdditionalNames,
                    TypesToInline);
  } else if (auto *Enum = llvm::dyn_cast<model::EnumDefinition>(&T)) {
    printDefinition(llvm::cast<model::EnumDefinition>(T),
                    Header,
                    B,
                    ForEditing);
  } else {
    revng_abort("Unsupported type definition.");
  }
}

void printInlineDefinition(Logger<> &Log,
                           llvm::StringRef Name,
                           const model::Type &T,
                           ptml::PTMLIndentedOstream &Header,
                           ptml::PTMLCBuilder &B,
                           const model::Binary &Model,
                           std::map<model::UpcastableType, std::string>
                             &AdditionalNames,
                           const std::set<const model::TypeDefinition *>
                             &TypesToInline) {
  const model::TypeDefinition *Definition = T.skipToDefinition();
  revng_assert(Definition, "Primitives cannot be printed inline.");

  auto Suffix = getNamedCInstance(T, Name, B, {}, true).str().str();
  if (auto *Struct = llvm::dyn_cast<model::StructDefinition>(Definition)) {
    printDefinition(Log,
                    *Struct,
                    Header,
                    B,
                    Model,
                    AdditionalNames,
                    TypesToInline,
                    /* ForEditing = */ false,
                    std::move(Suffix));

  } else if (auto *Union = llvm::dyn_cast<model::UnionDefinition>(Definition)) {
    printDefinition(Log,
                    *Union,
                    Header,
                    B,
                    Model,
                    AdditionalNames,
                    TypesToInline,
                    std::move(Suffix));

  } else if (auto *Enum = llvm::dyn_cast<model::EnumDefinition>(Definition)) {
    printDefinition(*Enum, Header, B, false, std::move(Suffix));

  } else {
    revng_abort("Only enums, structs, and unions can be printed inline.");
  }
}

void printInlineDefinition(Logger<> &Log,
                           const model::StructDefinition &Struct,
                           ptml::PTMLIndentedOstream &Header,
                           ptml::PTMLCBuilder &B,
                           const model::Binary &Model,
                           TypeNameMap &AdditionalNames,
                           const DefinitionSet &TypesToInline,
                           std::string &&Suffix) {
  printDefinition(Log,
                  Struct,
                  Header,
                  B,
                  Model,
                  AdditionalNames,
                  TypesToInline,
                  /* ForEditing = */ false,
                  " " + std::move(Suffix));
}
