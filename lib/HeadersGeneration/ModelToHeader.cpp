//
// Copyright (c) rev.ng Labs Srl. See LICENSE.md for details.
//

#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/Model/Binary.h"
#include "revng/Model/Type.h"
#include "revng/PTML/ModelHelpers.h"
#include "revng/Pipeline/Location.h"
#include "revng/Support/Assert.h"
#include "revng/Support/Debug.h"
#include "revng/Support/YAMLTraits.h"

#include "revng-c/HeadersGeneration/ModelToHeader.h"
#include "revng-c/Pipes/Ranks.h"
#include "revng-c/Support/ModelHelpers.h"
#include "revng-c/Support/PTMLC.h"
#include "revng-c/TypeNames/ModelTypeNames.h"

#include "DependencyGraph.h"

using ArtificialTypes::ArrayWrapperFieldName;

using llvm::cast;
using llvm::isa;
using llvm::Twine;

using modelEditPath::getCustomNamePath;
using pipeline::serializedLocation;
using ptml::str;
using ptml::Tag;
namespace attributes = ptml::attributes;
namespace tokens = ptml::c::tokenTypes;
namespace ranks = revng::ranks;

static Logger<> Log{ "model-to-header" };

static bool declarationIsDefinition(const model::Type *T) {
  return not isa<model::StructType>(T) and not isa<model::UnionType>(T);
}

static void printDeclaration(const model::PrimitiveType &P,
                             ptml::PTMLIndentedOstream &Header) {
  switch (P.PrimitiveKind) {

  case model::PrimitiveTypeKind::Unsigned: {
    // If it's 16 byte wide we need a typedef, since uint128_t is not defined
    // by the language
    if (P.Size == 16)
      Header << keywords::Typedef << " "
             << tokenTag("__uint128_t", tokens::Type) << " "
             << tokenTag(P.name().str(), tokens::Type)
                  .addAttribute(attributes::LocationDefinition,
                                serializedLocation(ranks::Type, P.key()))
             << ";\n";
    else if (Log.isEnabled())
      Header << helpers::lineComment("not necessary, already in stdint.h");

  } break;

  case model::PrimitiveTypeKind::Signed: {
    if (P.Size == 16)
      Header << keywords::Typedef << " " << tokenTag("__int128_t", tokens::Type)
             << " "
             << tokenTag(P.name(), tokens::Type)
                  .addAttribute(attributes::LocationDefinition,
                                serializedLocation(ranks::Type, P.key()))
             << ";\n";
    else if (Log.isEnabled())
      Header << helpers::lineComment("not necessary, already in stdint.h");

  } break;

  case model::PrimitiveTypeKind::Void: {
    if (Log.isEnabled())
      Header << helpers::lineComment("not necessary, already in stdint.h");
  } break;

  case model::PrimitiveTypeKind::Float: {
    if (Log.isEnabled())
      Header << helpers::lineComment("not necessary, already in revngfloat.h");
  } break;

  case model::PrimitiveTypeKind::Number:
  case model::PrimitiveTypeKind::PointerOrNumber:
  case model::PrimitiveTypeKind::Generic: {
    std::string IntType = [&]() constexpr {
      switch (P.Size) {
      case 1:
        return "uint8_t";
      case 2:
        return "uint16_t";
      case 4:
        return "uint32_t";
      case 8:
        return "uint64_t";
      case 16:
        return "__uint128_t";
      default:
        return "";
      }
    }
    ();
    if (!IntType.empty())
      Header << keywords::Typedef << " " << tokenTag(IntType, tokens::Type)
             << " "
             << tokenTag(P.name().str(), tokens::Type)
                  .addAttribute(attributes::LocationDefinition,
                                serializedLocation(ranks::Type, P.key()))
             << ";\n";
  } break;

  default:
    if (Log.isEnabled())
      Header << helpers::lineComment("invalid primitive type");
    revng_abort("Invalid primitive type");
  }
}

static void
printDeclaration(const model::EnumType &E, ptml::PTMLIndentedOstream &Header) {
  // We have to make the enum of the correct size of the underlying type
  auto ByteSize = *E.size();
  revng_assert(ByteSize <= 8);
  size_t FullMask = std::numeric_limits<size_t>::max();
  size_t MaxBitPatternInEnum = (ByteSize == 8) ?
                                 FullMask :
                                 ((FullMask) xor (FullMask << (8 * ByteSize)));

  Header << keywords::Typedef << " " << keywords::Enum << " " << helpers::Packed
         << " ";

  {
    Scope Scope(Header);

    for (const auto &Entry : E.Entries) {
      if (not Entry.CustomName.empty()) {
        Header << tokenTag(E.name().str().str() + "_"
                             + Entry.CustomName.str().str(),
                           tokens::Field)
                    .addAttribute(attributes::LocationDefinition,
                                  serializedLocation(ranks::EnumEntry,
                                                     E.key(),
                                                     Entry.key()))
                    .addAttribute(attributes::ModelEditPath,
                                  getCustomNamePath(E, Entry))
               << " " << operators::Assign << " " << constants::hex(Entry.Value)
               << ",\n";
      }
    }

    // This ensures the enum is large exactly like the Underlying type
    Header << tokenTag((E.name() + "_max_held_value").str(), tokens::Field)
           << " " + operators::Assign + " "
           << constants::hex(MaxBitPatternInEnum) << ",\n";
  }

  Header << " "
         << tokenTag(E.name(), tokens::Type)
              .addAttribute(attributes::LocationDefinition,
                            serializedLocation(ranks::Type, E.key()))
              .addAttribute(attributes::ModelEditPath, getCustomNamePath(E))
         << ";\n";
}

static void printForwardDeclaration(const model::StructType &S,
                                    ptml::PTMLIndentedOstream &Header) {
  Tag TheTag = tokenTag(S.name(), tokens::Type)
                 .addAttribute(attributes::LocationReferences,
                               serializedLocation(ranks::Type, S.key()))
                 .addAttribute(attributes::ModelEditPath, getCustomNamePath(S));
  Header << keywords::Struct << " " << helpers::Packed << " " + TheTag + ";\n";
  Header << keywords::Typedef << " " << keywords::Struct << " "
         << helpers::Packed << " " << TheTag << " " << TheTag << ";\n";
}

static void
printDefinition(const model::StructType &S, ptml::PTMLIndentedOstream &Header) {
  Header << keywords::Struct << " " << helpers::Packed << " ";
  Header << tokenTag(S.name(), tokens::Type)
              .addAttribute(attributes::LocationDefinition,
                            serializedLocation(ranks::Type, S.key()))
              .addAttribute(attributes::ModelEditPath, getCustomNamePath(S))
         << " ";
  {
    Scope Scope(Header, scopeTags::Struct);

    size_t NextOffset = 0ULL;
    for (const auto &Field : S.Fields) {
      if (NextOffset < Field.Offset)
        Header << tokenTag("uint8_t", tokens::Type) << " "
               << tokenTag("padding_at_offset_" + std::to_string(NextOffset),
                           tokens::Field)
               << "[" << constants::number(Field.Offset - NextOffset) << "];\n";

      Tag FieldTag = tokenTag(Field.name().str(), tokens::Field)
                       .addAttribute(attributes::LocationDefinition,
                                     serializedLocation(ranks::StructField,
                                                        S.key(),
                                                        Field.key()))
                       .addAttribute(attributes::ModelEditPath,
                                     getCustomNamePath(S, Field));
      Header << getNamedCInstance(Field.Type, FieldTag.serialize()) << ";\n";

      NextOffset = Field.Offset + Field.Type.size().value();
    }

    if (NextOffset < S.Size)
      Header << tokenTag("uint8_t", tokens::Type) << " "
             << tokenTag("padding_at_offset_" + std::to_string(NextOffset),
                         tokens::Field)
             << "[" << constants::number(S.Size - NextOffset) << "];\n";
  }
  Header << ";\n";
}

static void printForwardDeclaration(const model::UnionType &U,
                                    ptml::PTMLIndentedOstream &Header) {
  Tag TheTag = tokenTag(U.name(), tokens::Type)
                 .addAttribute(attributes::LocationReferences,
                               serializedLocation(ranks::Type, U.key()))
                 .addAttribute(attributes::ModelEditPath, getCustomNamePath(U));
  Header << keywords::Union << " " << helpers::Packed << " " + TheTag + ";\n";
  Header << keywords::Typedef << " " << keywords::Union << " "
         << helpers::Packed << " " << TheTag << " " << TheTag << ";\n";
}

static void
printDefinition(const model::UnionType &U, ptml::PTMLIndentedOstream &Header) {
  Header << keywords::Union << " " << helpers::Packed << " ";
  Header << tokenTag(U.name(), tokens::Type)
              .addAttribute(attributes::LocationDefinition,
                            serializedLocation(ranks::Type, U.key()))
              .addAttribute(attributes::ModelEditPath, getCustomNamePath(U))
         << " ";

  {
    Scope Scope(Header, scopeTags::Union);
    for (const auto &Field : U.Fields) {
      Tag FieldTag = tokenTag(Field.name().str(), tokens::Field)
                       .addAttribute(attributes::LocationDefinition,
                                     serializedLocation(ranks::UnionField,
                                                        U.key(),
                                                        Field.key()))
                       .addAttribute(attributes::ModelEditPath,
                                     getCustomNamePath(U, Field));
      Header << getNamedCInstance(Field.Type, FieldTag.serialize()) << ";\n";
    }
  }

  Header << ";\n";
}

static void printDeclaration(const model::TypedefType &TD,
                             ptml::PTMLIndentedOstream &Header) {
  Tag NewType = tokenTag(TD.name(), tokens::Type)
                  .addAttribute(attributes::ModelEditPath,
                                getCustomNamePath(TD))
                  .addAttribute(attributes::LocationDefinition,
                                serializedLocation(ranks::Type, TD.key()));
  Header << keywords::Typedef << " "
         << getNamedCInstance(TD.UnderlyingType, NewType.serialize()) << ";\n";
}

static void printSegmentsTypes(const model::Segment &Segment,
                               ptml::PTMLIndentedOstream &Header) {
  auto SegmentTag = tokenTag(Segment.name(), tokens::Variable)
                      .addAttribute(attributes::ModelEditPath,
                                    getCustomNamePath(Segment))
                      .addAttribute(attributes::LocationDefinition,
                                    serializedLocation(ranks::Segment,
                                                       Segment.key()));
  Header << getNamedCInstance(Segment.Type, SegmentTag.serialize()) << ";\n";
}

/// Generate the definition of a new struct type that wraps all the
///        return values of \a F. The name of the struct type is provided by the
///        caller.
static void generateReturnValueWrapper(const model::RawFunctionType &F,
                                       ptml::PTMLIndentedOstream &Header) {
  revng_assert(F.ReturnValues.size() > 1);
  if (Log.isEnabled())
    Header << helpers::lineComment("definition the of return type needed");

  Header << keywords::Typedef << " " << keywords::Struct << " "
         << helpers::Packed << " ";

  {
    Scope Scope(Header, scopeTags::Struct);
    for (auto &Group : llvm::enumerate(F.ReturnValues)) {
      const model::QualifiedType &RetTy = Group.value().Type;
      const auto &FieldName = getReturnField(F, Group.index());
      Header
        << getNamedCInstance(RetTy,
                             tokenTag(FieldName, tokens::Field).serialize())
        << ";\n";
    }
  }

  Header << " " << getReturnTypeName(F) << ";\n";
}

/// If the function has more than one return value, generate a wrapper
///        struct that contains them.
static void printRawFunctionWrappers(const model::RawFunctionType *F,
                                     ptml::PTMLIndentedOstream &Header) {
  if (F->ReturnValues.size() > 1)
    generateReturnValueWrapper(*F, Header);

  for (auto &Arg : F->Arguments)
    revng_assert(Arg.Type.isScalar());
}

/// Print a typedef for a RawFunctionType, that can be used when you have
///        a variable that is a pointer to a function.
static void printDeclaration(const model::RawFunctionType &F,
                             ptml::PTMLIndentedOstream &Header,
                             const model::Binary &Model) {
  printRawFunctionWrappers(&F, Header);

  Header << keywords::Typedef << " ";
  // In this case, we are defining a type for the function, not the function
  // itself, so the token right before the parenthesis is the name of the type.
  printFunctionPrototype(F, getTypeName(F), Header, Model, true);
  Header << ";\n";
}

using QualifiedTypeNameMap = std::map<model::QualifiedType, std::string>;

/// Generate the definition of a new struct type that wraps \a ArrayType.
///        This is used to wrap array arguments or array return values of
///        CABIFunctionTypes.
static void generateArrayWrapper(const model::QualifiedType &ArrayType,
                                 ptml::PTMLIndentedOstream &Header,
                                 QualifiedTypeNameMap &NamesCache) {
  revng_assert(ArrayType.isArray());
  auto WrapperName = getArrayWrapper(ArrayType);

  // Check if the wrapper was already added
  bool IsNew = NamesCache.emplace(ArrayType, WrapperName).second;
  if (not IsNew)
    return;

  Header << keywords::Typedef << " " << keywords::Struct << " "
         << helpers::Packed << " ";
  {
    Scope Scope(Header, scopeTags::Struct);
    Header << getNamedCInstance(ArrayType, ArrayWrapperFieldName) << ";\n";
  }
  Header << " " << tokenTag(WrapperName, tokens::Type) << ";\n";
}

/// If the return value or any of the arguments is an array, generate
///        a wrapper struct for each of them, if it's not already in the cache.
static void printCABIFunctionWrappers(const model::CABIFunctionType *F,
                                      ptml::PTMLIndentedOstream &Header,
                                      QualifiedTypeNameMap &NamesCache) {
  if (F->ReturnType.isArray())
    generateArrayWrapper(F->ReturnType, Header, NamesCache);

  for (auto &Arg : F->Arguments)
    if (Arg.Type.isArray())
      generateArrayWrapper(Arg.Type, Header, NamesCache);
}

/// Print a typedef for a CABIFunctionType, that can be used when you
///        have a variable that is a pointer to a function.
static void printDeclaration(const model::CABIFunctionType &F,
                             ptml::PTMLIndentedOstream &Header,
                             QualifiedTypeNameMap &NamesCache,
                             const model::Binary &Model) {
  printCABIFunctionWrappers(&F, Header, NamesCache);

  Header << keywords::Typedef << " ";
  // In this case, we are defining a type for the function, not the function
  // itself, so the token right before the parenthesis is the name of the type.
  printFunctionPrototype(F, getTypeName(F), Header, Model, true);
  Header << ";\n";
}

static void printDeclaration(const model::Type &T,
                             ptml::PTMLIndentedOstream &Header,
                             QualifiedTypeNameMap &AdditionalTypeNames,
                             const model::Binary &Model) {
  if (Log.isEnabled()) {
    auto Scope = helpers::LineComment(Header);
    Header << "Declaration of " << getNameFromYAMLScalar(T.key());
  }

  revng_log(Log, "Declaring " << getNameFromYAMLScalar(T.key()));

  switch (T.Kind) {

  case model::TypeKind::Invalid: {
    if (Log.isEnabled())
      Header << helpers::lineComment("invalid");
  } break;

  case model::TypeKind::PrimitiveType: {
    printDeclaration(cast<model::PrimitiveType>(T), Header);
  } break;

  case model::TypeKind::EnumType: {
    printDeclaration(cast<model::EnumType>(T), Header);
  } break;

  case model::TypeKind::StructType: {
    printForwardDeclaration(cast<model::StructType>(T), Header);
  } break;

  case model::TypeKind::UnionType: {
    printForwardDeclaration(cast<model::UnionType>(T), Header);
  } break;

  case model::TypeKind::TypedefType: {
    printDeclaration(cast<model::TypedefType>(T), Header);
  } break;

  case model::TypeKind::RawFunctionType: {
    printDeclaration(cast<model::RawFunctionType>(T), Header, Model);
  } break;

  case model::TypeKind::CABIFunctionType: {
    printDeclaration(cast<model::CABIFunctionType>(T),
                     Header,
                     AdditionalTypeNames,
                     Model);
  } break;
  default:
    revng_abort();
  }
}

static void printDefinition(const model::Type &T,
                            ptml::PTMLIndentedOstream &Header,
                            QualifiedTypeNameMap &AdditionalTypeNames,
                            const model::Binary &Model) {
  if (Log.isEnabled())
    Header << helpers::lineComment("Definition of "
                                   + getNameFromYAMLScalar(T.key()));

  revng_log(Log, "Defining " << getNameFromYAMLScalar(T.key()));

  if (declarationIsDefinition(&T)) {
    printDeclaration(T, Header, AdditionalTypeNames, Model);
  } else {
    switch (T.Kind) {

    case model::TypeKind::Invalid: {
      if (Log.isEnabled())
        Header << helpers::lineComment("invalid");
    } break;

    case model::TypeKind::StructType: {
      printDefinition(cast<model::StructType>(T), Header);
    } break;

    case model::TypeKind::UnionType: {
      printDefinition(cast<model::UnionType>(T), Header);
    } break;

    default:
      revng_abort();
    }
  }
}

/// Print all type definitions for the types in the model
static void printTypeDefinitions(const model::Binary &Model,
                                 ptml::PTMLIndentedOstream &Header,
                                 QualifiedTypeNameMap &AdditionalTypeNames) {
  DependencyGraph Dependencies = buildDependencyGraph(Model.Types);
  const auto &TypeNodes = Dependencies.TypeNodes();

  std::set<const TypeDependencyNode *> Defined;

  for (const auto *Root : Dependencies.nodes()) {
    revng_log(Log, "======== PostOrder " << getNodeLabel(Root));

    for (const auto *Node : llvm::post_order_ext(Root, Defined)) {
      revng_log(Log, "== visiting " << getNodeLabel(Node));
      for (const auto *Child :
           llvm::children<const TypeDependencyNode *>(Node)) {
        revng_log(Log, "= child " << getNodeLabel(Child));
        if (Defined.count(Child))
          revng_log(Log, "      DEFINED");
        else
          revng_log(Log, "      NOT DEFINED");
      }

      const model::Type *NodeT = Node->T;
      const auto DeclKind = Node->K;
      constexpr auto TypeName = TypeNode::Kind::TypeName;
      constexpr auto FullType = TypeNode::Kind::FullType;
      if (DeclKind == FullType) {

        // When emitting a full definition we also want to emit a forward
        // declaration first, if it wasn't already emitted somewhere else.
        if (Defined.insert(TypeNodes.at({ NodeT, TypeName })).second)
          printDeclaration(*NodeT, Header, AdditionalTypeNames, Model);

        if (not declarationIsDefinition(NodeT))
          printDefinition(*NodeT, Header, AdditionalTypeNames, Model);

        // This is always a full type definition
        Defined.insert(TypeNodes.at({ NodeT, FullType }));
      } else {
        printDeclaration(*NodeT, Header, AdditionalTypeNames, Model);
        Defined.insert(TypeNodes.at({ NodeT, TypeNode::Kind::TypeName }));

        // For primitive types and enums the forward declaration we emit is
        // also a full definition, so we need to keep track of this.
        if (isa<model::PrimitiveType>(NodeT) or isa<model::EnumType>(NodeT))
          Defined.insert(TypeNodes.at({ NodeT, TypeNode::Kind::FullType }));

        // For struct and unions the forward declaration is just a forward
        // declaration, without body.

        // TypedefType, RawFunctionType and CABIFunctionType are emitted in C
        // as typedefs, so they don't represent fully defined types, but just
        // names, unless all the types they depend from are also fully
        // defined, but that happens when DeclKind == FullType, not here.
      }
    }
    revng_log(Log, "====== PostOrder DONE");
  }
}

bool dumpModelToHeader(const model::Binary &Model, llvm::raw_ostream &Out) {
  ptml::PTMLIndentedOstream Header(Out, 4);
  {
    auto Scope = Tag(ptml::tags::Div).scope(Header);

    Header << helpers::pragmaOnce();
    Header << helpers::includeAngle("stdint.h");
    Header << helpers::includeAngle("stdbool.h");
    Header << helpers::includeQuote("revngfloat.h");
    Header << "\n";

    Header << directives::IfNotDef << " " << constants::Null << "\n"
           << directives::Define << " " << constants::Null << " ("
           << constants::Zero << ")\n"
           << directives::EndIf << "\n\n";

    QualifiedTypeNameMap AdditionalTypeNames;
    printTypeDefinitions(Model, Header, AdditionalTypeNames);

    for (const model::Function &MF : Model.Functions) {
      const model::Type *FT = MF.Prototype.get();
      auto FName = model::Identifier::fromString(MF.name());

      if (Log.isEnabled()) {
        helpers::BlockComment CommentScope(Header);
        Header << "Analyzing Model function " << FName << "\n";
        serialize(Header, MF);
        Header << "Prototype\n";
        serialize(Header, *FT);
      }

      printFunctionPrototype(*FT, MF, Header, Model, false);
      Header << ";\n";
    }

    for (const model::DynamicFunction &MF : Model.ImportedDynamicFunctions) {
      const model::Type *FT = MF.prototype(Model).get();
      revng_assert(FT != nullptr);
      auto FName = model::Identifier::fromString(MF.name());

      if (Log.isEnabled()) {
        helpers::BlockComment CommentScope(Header);
        Header << "Analyzing dynamic function " << FName << "\n";
        serialize(Header, MF);
        Header << "Prototype\n";
        serialize(Header, *FT);
      }
      printFunctionPrototype(*FT, MF, Header, Model, true);
      Header << ";\n";
    }

    for (const model::Segment &Segment : Model.Segments)
      printSegmentsTypes(Segment, Header);
  }
  return true;
}
