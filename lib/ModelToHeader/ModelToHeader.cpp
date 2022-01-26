//
// Copyright (c) rev.ng Srls. See LICENSE.md for details.
//

#include <type_traits>
#include <utility>

#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/DOTGraphTraits.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/GraphWriter.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/ADT/FilteredGraphTraits.h"
#include "revng/ADT/GenericGraph.h"
#include "revng/Model/Binary.h"
#include "revng/Model/Type.h"
#include "revng/Support/Assert.h"
#include "revng/Support/Debug.h"

#include "revng-c/ModelToHeader/ModelToHeader.h"

using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;
using llvm::Twine;

static Logger<> Log{ "model-to-header" };

static bool declarationIsDefinition(const model::Type *T) {
  return not isa<model::StructType>(T) and not isa<model::UnionType>(T);
}

static llvm::SmallString<16> printNamedCInstance(const model::QualifiedType &QT,
                                                 llvm::StringRef InstanceName) {
  llvm::SmallString<16> Result;

  const model::Type *Unqualified = QT.UnqualifiedType.get();

  if (isa<model::RawFunctionType>(Unqualified)
      or isa<model::CABIFunctionType>(Unqualified))
    Result += "unnamed_function_type_";

  Result.append(Unqualified->name());

  auto QIt = QT.Qualifiers.begin();
  auto QEnd = QT.Qualifiers.end();
  bool PointerFound = false;
  for (; QIt != QEnd and not QIt->isArrayQualifier(); ++QIt) {
    switch (QIt->Kind) {
    case model::QualifierKind::Const:
      Result.append(" const");
      break;
    case model::QualifierKind::Pointer:
      Result.append(" *");
      PointerFound = true;
      break;
    default:
      revng_abort();
    }
  }

  if (not Result.empty() and not InstanceName.empty() and Result.back() != '*')
    Result.append(" ");

  Result.append(InstanceName);

  for (; QIt != QEnd; ++QIt) {
    // TODO revng_assert(QIt->isArrayQualifier()); instead of the following
    revng_assert(not QIt->isPointerQualifier());
    Result.append((Twine("[") + Twine(QIt->Size) + Twine("]")).str());
  }

  return Result;
}

static void
printDeclaration(const model::PrimitiveType &P, llvm::raw_ostream &Header) {
  switch (P.PrimitiveKind) {

  case model::PrimitiveTypeKind::Unsigned: {
    // If it's 16 byte wide we need a typedef, since uint128_t is not defined
    // by the language
    if (P.Size == 16)
      Header << "typedef __uint128_t " << P.name() << ";\n";
    else if (Log.isEnabled())
      Header << "// not necessary, already in stdint.h\n";

  } break;

  case model::PrimitiveTypeKind::Signed: {
    if (P.Size == 16)
      Header << "typedef __int128_t " << P.name() << ";\n";
    else if (Log.isEnabled())
      Header << "// not necessary, already in stdint.h\n";

  } break;

  case model::PrimitiveTypeKind::Void: {
    if (Log.isEnabled())
      Header << "// not necessary, already in stdint.h\n";
  } break;

  case model::PrimitiveTypeKind::Float: {

    switch (P.Size) {

    case 2:
      Header << "// half-precision floating point, not supported yet";
      break;

    case 4:
      Header << "typedef float " << P.name() << ";\n";
      break;

    case 8:
      Header << "typedef double " << P.name() << ";\n";
      break;

    case 16:
      Header << "typedef long double " << P.name() << ";\n";
      break;

    default:
      if (Log.isEnabled())
        Header << "// unsupported floating point with size " << P.Size
               << " bytes\n";
      break;
    }
  } break;

  case model::PrimitiveTypeKind::Number:
  case model::PrimitiveTypeKind::PointerOrNumber:
  case model::PrimitiveTypeKind::Generic: {
    switch (P.Size) {
    case 1:
      Header << "typedef uint8_t " << P.name() << ";\n";
      break;

    case 2:
      Header << "typedef uint16_t " << P.name() << ";\n";
      break;

    case 4:
      Header << "typedef uint32_t " << P.name() << ";\n";
      break;

    case 8:
      Header << "typedef uint64_t " << P.name() << ";\n";
      break;

    case 16:
      Header << "typedef __uint128_t " << P.name() << ";\n";
      break;
    }
  } break;

  default:
    if (Log.isEnabled())
      Header << "// invalid primitive type\n";
  }
}

static void
printDeclaration(const model::EnumType &E, llvm::raw_ostream &Header) {
  // We have to make the enum of the correct size of the underlying type
  const auto *P = cast<model::PrimitiveType>(E.UnderlyingType.get());
  auto ByteSize = P->Size;
  revng_assert(ByteSize <= 8);
  size_t FullMask = std::numeric_limits<size_t>::max();
  size_t MaxBitPatternInEnum = (ByteSize == 8) ?
                                 FullMask :
                                 ((FullMask) xor (FullMask << (8 * ByteSize)));

  Header << "typedef enum __attribute__((packed)) {\n";

  for (const auto &Entry : E.Entries) {
    if (not Entry.CustomName.empty()) {
      Header << "  " << E.name() << "_" << Entry.CustomName << " = 0x";
      Header.write_hex(Entry.Value);
      Header << "U,\n";
    }

  }

  // This ensures the enum is large exactly like the Underlying type
  Header << "  " << E.name() << "_max_held_value = 0x";
  Header.write_hex(MaxBitPatternInEnum);
  Header << "U,\n} " << E.name() << ";\n";
}

static void
printForwardDeclaration(const model::StructType &S, llvm::raw_ostream &Header) {
  Header << "struct __attribute__((packed)) " << S.name() << ";\n";
  Header << "typedef struct __attribute__((packed)) " << S.name() << ' '
         << S.name() << ";\n";
}

static void
printDefinition(const model::StructType &S, llvm::raw_ostream &Header) {
  Header << "struct __attribute__((packed)) " << S.name() << "{\n";

  size_t NextOffset = 0ULL;
  for (const auto &Field : S.Fields) {

    if (NextOffset < Field.Offset)
      Header << "  uint8_t padding_at_offset_" << Twine(NextOffset) << "["
             << Twine(Field.Offset - NextOffset) << "];\n";

    Header << "  " << printNamedCInstance(Field.Type, Field.name()) << ";\n";

    NextOffset = Field.Offset + Field.Type.size().value();
  }

  if (NextOffset < S.Size)
    Header << "  uint8_t padding_at_offset_" << Twine(NextOffset) << "["
           << Twine(S.Size - NextOffset) << "];\n";

  Header << "};\n";
}

static void
printForwardDeclaration(const model::UnionType &U, llvm::raw_ostream &Header) {
  Header << "union __attribute__((packed)) " << U.name() << ";\n";
  Header << "typedef union __attribute__((packed)) " << U.name() << ' '
         << U.name() << ";\n";
}

static void
printDefinition(const model::UnionType &U, llvm::raw_ostream &Header) {
  Header << "union __attribute__((packed)) " << U.name() << "{\n";

  for (const auto &Field : U.Fields)
    Header << "  " << printNamedCInstance(Field.Type, Field.name()) << ";\n";

  Header << "};\n";
}

static void
printDeclaration(const model::TypedefType &TD, llvm::raw_ostream &Header) {
  Header << "typedef " << printNamedCInstance(TD.UnderlyingType, TD.name())
         << ";\n";
}

static llvm::SmallString<16>
getRawFunctionReturnTypeName(const model::RawFunctionType &F) {
  llvm::SmallString<16> Result;

  // We need to make sure that the return type is fully defined.
  // This is always true for scalar values, but if it returns more than one
  // value we need to declare a special struct for it on the fly.
  switch (F.ReturnValues.size()) {

  case 0: {
    Result = "void ";
  } break;

  case 1: {
    Result = printNamedCInstance(F.ReturnValues.begin()->Type, "");
  } break;

  default: {

    Result = (Twine("unnamed_return_type_") + Twine(F.name())).str();

  } break;
  }

  revng_assert(not Result.empty());

  return Result;
}

static void
printDeclaration(const model::RawFunctionType &F, llvm::raw_ostream &Header) {

  auto RetTypeName = getRawFunctionReturnTypeName(F);

  if (F.ReturnValues.size() > 1) {
    if (Log.isEnabled())
      Header << "// definition the of return type needed\n";

    Header << "typedef struct __attribute__((packed)) {\n";

    for (auto &Group : llvm::enumerate(F.ReturnValues)) {
      const model::QualifiedType &RetTy = Group.value().Type;
      revng_assert(isa<model::PrimitiveType>(RetTy.UnqualifiedType.get()));
      std::string FName = (Twine("return_field_") + Twine(Group.index())).str();
      Header << "  " << printNamedCInstance(RetTy, FName) << ";\n";
    }

    Header << "} " << RetTypeName << ";\n ";
  }

  Header << "typedef " << RetTypeName << " unnamed_function_type_" << F.name();

  if (F.Arguments.empty()) {
    Header << "(void);\n";
  } else {
    const llvm::StringRef Open = "(";
    const llvm::StringRef Comma = ", ";
    llvm::StringRef Separator = Open;
    for (const auto &Arg : F.Arguments) {
      Header << Separator << printNamedCInstance(Arg.Type, Arg.name());
      Separator = Comma;
    }
    Header << ");\n";
  }
}

static bool isEventuallyArray(const model::QualifiedType &QT) {

  const model::QualifiedType *NextQt = &QT;

  while (NextQt) {
    if (not NextQt->Qualifiers.empty())
      if (NextQt->Qualifiers.back().isArrayQualifier())
        return true;

    const model::Type *Unqualified = NextQt->UnqualifiedType.get();
    if (const auto *TD = dyn_cast<model::TypedefType>(Unqualified))
      NextQt = &TD->UnderlyingType;
    else
      NextQt = nullptr;
  }

  // We've traversed all layers of typedefs and we have never found an array
  // qualifier, hence this is QT is not eventually an array.
  return false;
}

// static bool isEventuallyFunction(const model::QualifiedType &QT) {
// return false;
// }

static llvm::SmallString<32> getArrayTypeName(const model::QualifiedType &QT) {
  llvm::SmallString<32> Result{ "unnamed_array_wrapper_" };

  for (const auto &Qualifier : llvm::reverse(QT.Qualifiers)) {

    switch (Qualifier.Kind) {

    case model::QualifierKind::Const: {
      Result += "const_";
    } break;

    case model::QualifierKind::Pointer: {
      Result += "ptr_to_";
    } break;

    case model::QualifierKind::Array: {
      auto NElem = Qualifier.Size;
      Result.append((Twine("array_") + Twine(NElem) + Twine("_of_")).str());
    } break;

    default:
      revng_abort();
    }
  }

  Result.append(QT.UnqualifiedType.get()->name());

  return Result;
}

// Some model::QualifiedTypes require to declare new types (e.g. for returning
// an array from a functions you need to wrap it into a struct).
// For those model::QualifiedTypes we need to keep track of which already have
// the associated type, because otherwise the type declarations will be
// duplicated.
// This FrozenQualifiedType is used for that.
class FrozenQualifiedType {
  const model::Type *Unqualified;

  std::vector<model::Qualifier> Qualifiers = {};

public:
  FrozenQualifiedType(const model::QualifiedType &QT) :
    Unqualified{ QT.UnqualifiedType.get() }, Qualifiers{ QT.Qualifiers } {}

  std::strong_ordering
  operator<=>(const FrozenQualifiedType &Other) const = default;
};

using QualifiedTypeNameMap = std::map<FrozenQualifiedType, std::string>;

static llvm::SmallString<16>
getCABIFunctionReturnTypeName(const model::CABIFunctionType &F,
                              llvm::raw_ostream &Header,
                              QualifiedTypeNameMap &AdditionalTypeNames) {
  llvm::SmallString<16> Result;

  const auto &RetTy = F.ReturnType;

  auto AdditionalRetNameIt = AdditionalTypeNames.find(RetTy);
  if (AdditionalRetNameIt != AdditionalTypeNames.end()) {
    Result = AdditionalRetNameIt->second;
  } else if (isEventuallyArray(F.ReturnType)) {
    if (Log.isEnabled())
      Header << "// definition of argument or return type needed\n";

    Result = getArrayTypeName(RetTy);

    if (AdditionalTypeNames.emplace(RetTy, Result).second) {
      Header << "typedef struct __attribute__((packed)) {\n";
      Header << "  " << printNamedCInstance(RetTy, "the_array") << ";\n";
      Header << "} " << Result << ";\n ";
    }
  } else {
    Result = printNamedCInstance(RetTy, "");
  }

  revng_assert(not Result.empty());

  return Result;
}

static llvm::SmallString<16>
getCABIFunctionArgumentDeclaration(const model::Argument &Arg,
                                   llvm::raw_ostream &Header,
                                   QualifiedTypeNameMap &AdditionalTypeNames) {
  llvm::SmallString<16> Result;
  const auto &ArgTy = Arg.Type;
  const auto &ArgName = Arg.name();

  auto AdditionalArgNameIt = AdditionalTypeNames.find(ArgTy);
  if (AdditionalArgNameIt != AdditionalTypeNames.end()) {
    Result = (Twine(AdditionalArgNameIt->second) + Twine(" ") + Twine(ArgName))
               .str();
  } else if (isEventuallyArray(ArgTy)) {
    if (Log.isEnabled())
      Header << "// definition or argument type " << Twine(Arg.Index)
             << " needed\n";

    auto Name = getArrayTypeName(ArgTy);

    if (AdditionalTypeNames.emplace(ArgTy, Name).second) {
      Header << "typedef struct __attribute__((packed)) {\n";
      Header << "  " << printNamedCInstance(ArgTy, "the_array") << ";\n";
      Header << "} " << Name << ";\n ";
    }
    Result = (Twine(Name) + Twine(" ") + Twine(ArgName)).str();
  } else {
    Result = printNamedCInstance(ArgTy, ArgName);
  }

  revng_assert(not Result.empty());
  return Result;
}

llvm::SmallVector<llvm::SmallString<16>, 8>
getCABIFunctionArgumentDeclarations(const model::CABIFunctionType &F,
                                    llvm::raw_ostream &Header,
                                    QualifiedTypeNameMap &AdditionalTypeNames) {
  llvm::SmallVector<llvm::SmallString<16>, 8> ArgDeclarations;
  for (const auto &Arg : F.Arguments) {
    auto ArgDecl = getCABIFunctionArgumentDeclaration(Arg,
                                                      Header,
                                                      AdditionalTypeNames);
    ArgDeclarations.emplace_back(std::move(ArgDecl));
  }
  revng_assert(F.Arguments.size() == ArgDeclarations.size());
  return ArgDeclarations;
}

static void printDeclaration(const model::CABIFunctionType &F,
                             llvm::raw_ostream &Header,
                             QualifiedTypeNameMap &AdditionalTypeNames) {

  auto RetTypeName = getCABIFunctionReturnTypeName(F,
                                                   Header,
                                                   AdditionalTypeNames);

  auto ArgDecls = getCABIFunctionArgumentDeclarations(F,
                                                      Header,
                                                      AdditionalTypeNames);

  Header << "typedef " << RetTypeName << " unnamed_function_type_" << F.name();

  if (ArgDecls.empty()) {
    Header << "(void);\n";
  } else {
    const llvm::StringRef Open = "(";
    const llvm::StringRef Comma = ", ";
    llvm::StringRef Separator = Open;
    for (const auto &ArgDecl : ArgDecls) {
      Header << Separator << ArgDecl;
      Separator = Comma;
    }
    Header << ");\n";
  }
}

static void printDeclaration(const model::Type &T,
                             llvm::raw_ostream &Header,
                             QualifiedTypeNameMap &AdditionalTypeNames) {
  if (Log.isEnabled())
    Header << "// Declaration of " << getNameFromYAMLScalar(T.key()) << '\n';

  revng_log(Log, "Declaring " << getNameFromYAMLScalar(T.key()));

  switch (T.Kind) {

  case model::TypeKind::Invalid: {
    if (Log.isEnabled())
      Header << "// invalid\n";
  } break;

  case model::TypeKind::Primitive: {
    printDeclaration(cast<model::PrimitiveType>(T), Header);
  } break;

  case model::TypeKind::Enum: {
    printDeclaration(cast<model::EnumType>(T), Header);
  } break;

  case model::TypeKind::Struct: {
    printForwardDeclaration(cast<model::StructType>(T), Header);
  } break;

  case model::TypeKind::Union: {
    printForwardDeclaration(cast<model::UnionType>(T), Header);
  } break;

  case model::TypeKind::Typedef: {
    printDeclaration(cast<model::TypedefType>(T), Header);
  } break;

  case model::TypeKind::RawFunctionType: {
    printDeclaration(cast<model::RawFunctionType>(T), Header);
  } break;

  case model::TypeKind::CABIFunctionType: {
    printDeclaration(cast<model::CABIFunctionType>(T),
                     Header,
                     AdditionalTypeNames);
  } break;
  default:
    revng_abort();
  }
}

/// Represents a model::Type in the DependencyGraph
struct TypeNode {

  /// A pointer to the associated model::Type
  const model::Type *T;

  /// For each model::Type we'll have nodes representing the type name or
  /// the full type, depending on this enum.
  enum Kind { TypeName, FullType } K;
};

static llvm::StringRef toString(TypeNode::Kind K) {
  switch (K) {
  case TypeNode::Kind::TypeName:
    return "TypeName";
  case TypeNode::Kind::FullType:
    return "FullType";
  }
  return "Invalid";
}

using TypeDependencyNode = BidirectionalNode<TypeNode>;
using TypeKindPair = std::pair<const model::Type *, TypeNode::Kind>;
using TypeToDependencyNodeMap = std::map<TypeKindPair, TypeDependencyNode *>;

/// Represents the graph of dependencies among types
struct DependencyGraph : public GenericGraph<TypeDependencyNode> {

  void addNode(const model::Type *T) {

    constexpr auto TypeName = TypeNode::Kind::TypeName;
    auto *NameNode = GenericGraph::addNode(TypeNode{ T, TypeName });
    TypeToNode[TypeKindPair{ T, TypeName }] = NameNode;

    constexpr auto FullType = TypeNode::Kind::FullType;
    auto *FullNode = GenericGraph::addNode(TypeNode{ T, FullType });
    TypeToNode[TypeKindPair{ T, FullType }] = FullNode;
  }

  const TypeToDependencyNodeMap &TypeNodes() const { return TypeToNode; }

private:
  TypeToDependencyNodeMap TypeToNode;
};

static std::string getNodeLabel(const TypeDependencyNode *N) {
  return (Twine(getNameFromYAMLScalar(N->T->key())) + Twine("-")
          + Twine(toString(N->K)))
    .str();
}

template<>
struct llvm::DOTGraphTraits<DependencyGraph *>
  : public llvm::DefaultDOTGraphTraits {
  using llvm::DefaultDOTGraphTraits::DefaultDOTGraphTraits;

  std::string
  getNodeLabel(const TypeDependencyNode *N, const DependencyGraph *G) {
    return ::getNodeLabel(N);
  }
};

static TypeDependencyNode *
getDependencyForTypeName(const model::QualifiedType &QT,
                         const TypeToDependencyNodeMap &TypeToNode) {
  const auto *Unqualified = QT.UnqualifiedType.get();

  // If we find at least a pointer qualifier, then we only need the name of
  // the unqualified type, not its full definition.
  bool ArrayFound = false;
  for (const auto &Qualifier : QT.Qualifiers) {
    if (Qualifier.isPointerQualifier())
      return TypeToNode.at({ Unqualified, TypeNode::Kind::TypeName });
    if (Qualifier.isArrayQualifier())
      ArrayFound = true;
  }

  // If we reach this pointe we haven't found not even a single pointer
  // qualifier.

  // If we did find an array qualifier, we need the full type of the
  // unqualified type.
  if (ArrayFound)
    return TypeToNode.at({ Unqualified, TypeNode::Kind::FullType });

  // Otherwise we can get away with just the name of the unqualified type.
  return TypeToNode.at({ Unqualified, TypeNode::Kind::TypeName });
}

static TypeDependencyNode *
getDependencyForFullType(const model::QualifiedType &QT,
                         const TypeToDependencyNodeMap &TypeToNode) {
  const auto *Unqualified = QT.UnqualifiedType.get();

  // If we find at least a pointer qualifier, then we only need the name of
  // the unqualified type, not its full definition.
  bool ArrayFound = false;
  for (const auto &Qualifier : QT.Qualifiers) {
    if (Qualifier.isPointerQualifier())
      return TypeToNode.at({ Unqualified, TypeNode::TypeName });
    if (Qualifier.isArrayQualifier())
      ArrayFound = true;
  }

  // If we reach this pointe we haven't found not even a single pointer
  // qualifier. Given that we need the full definition, we need the full
  // type of of the unqualified type.
  return TypeToNode.at({ Unqualified, TypeNode::FullType });
}

static void registerDependencies(const model::Type *T,
                                 const TypeToDependencyNodeMap &TypeToNode) {

  const auto GetTypeRef =
    [](const auto &Typed) -> const model::QualifiedType & {
    return Typed.Type;
  };

  using Edge = std::pair<TypeDependencyNode *, TypeDependencyNode *>;
  llvm::SmallVector<Edge, 2> Deps;

  switch (T->Kind) {

  case model::TypeKind::Invalid: {
    revng_abort("Primitive or Invalid type should never depend on others");
  } break;

  case model::TypeKind::Primitive: {
    // Nothing to do here. Primitive types names and full definitions can
    // always be defined without dependencies, because they are either not
    // necessary (for primitive types that are already present in stdint.h)
    // or they boil down to a simple typedef of a type in stdint.h. In both
    // cases, the definition provide visibility on both the name and on the
    // full definition.
  } break;

  case model::TypeKind::Enum: {
    // Enum names and full definitions could always be conjured out of thin
    // air. However, given that we have enums with underlying primitive
    // types, for consistency we enforce that enums names and full
    // definitions always depend on full definition of the underlying
    // primitive type. This adds a little unnessary edges, but makes the
    // overall structure of the graph easier to reason about. Moreover, full
    // definitions of primitive types can also always be conjured out of
    // thin air, so we're always sure that this does not generates infinite
    // loops.
    const auto *E = cast<model::EnumType>(T);
    auto *Underlying = cast<model::PrimitiveType>(E->UnderlyingType.get());
    auto *EnumName = TypeToNode.at({ E, TypeNode::Kind::TypeName });
    auto *EnumFull = TypeToNode.at({ E, TypeNode::Kind::FullType });
    auto *UnderFull = TypeToNode.at({ Underlying, TypeNode::Kind::FullType });
    Deps.push_back({ EnumName, UnderFull });
    Deps.push_back({ EnumFull, UnderFull });
    revng_log(Log,
              getNodeLabel(EnumName)
                << " depends on " << getNodeLabel(UnderFull));
    revng_log(Log,
              getNodeLabel(EnumFull)
                << " depends on " << getNodeLabel(UnderFull));
  } break;

  case model::TypeKind::Struct: {
    // Struct names can always be conjured out of thin air thanks to
    // typedefs. So we only need to add dependencies between their full
    // definition and the full definition of their fields.
    auto *Struct = cast<model::StructType>(T);
    auto *StructFull = TypeToNode.at({ Struct, TypeNode::Kind::FullType });
    for (const model::StructField &Field : Struct->Fields) {
      TypeDependencyNode *Dep = getDependencyForFullType(Field.Type,
                                                         TypeToNode);
      Deps.push_back({ StructFull, Dep });
      revng_log(Log,
                getNodeLabel(StructFull)
                  << " depends on " << getNodeLabel(Dep));
    }
  } break;

  case model::TypeKind::Union: {
    // Union names can always be conjured out of thin air thanks to
    // typedefs. So we only need to add dependencies between their full
    // definition and the full definition of their fields.
    auto *Union = cast<model::UnionType>(T);
    auto *UnionFull = TypeToNode.at({ Union, TypeNode::Kind::FullType });
    for (const model::UnionField &Field : Union->Fields) {
      TypeDependencyNode *Dep = getDependencyForFullType(Field.Type,
                                                         TypeToNode);
      Deps.push_back({ UnionFull, Dep });
      revng_log(Log,
                getNodeLabel(UnionFull) << " depends on " << getNodeLabel(Dep));
    }
  } break;

  case model::TypeKind::Typedef: {
    // Typedefs are nasty.
    auto *TD = cast<model::TypedefType>(T);
    const model::QualifiedType &Underlying = TD->UnderlyingType;

    auto *TDName = TypeToNode.at({ TD, TypeNode::Kind::TypeName });
    TypeDependencyNode *NameDep = getDependencyForTypeName(Underlying,
                                                           TypeToNode);
    Deps.push_back({ TDName, NameDep });
    revng_log(Log,
              getNodeLabel(TDName) << " depends on " << getNodeLabel(NameDep));

    auto *TDFull = TypeToNode.at({ TD, TypeNode::Kind::FullType });
    TypeDependencyNode *FullDep = getDependencyForFullType(Underlying,
                                                           TypeToNode);
    Deps.push_back({ TDFull, FullDep });
    revng_log(Log,
              getNodeLabel(TDFull) << " depends on " << getNodeLabel(FullDep));
  } break;

  case model::TypeKind::RawFunctionType: {
    // For function types we can print a valid typedef definition as long as
    // we have visibility on all the names of all the argument types and all
    // return types.
    auto *F = cast<model::RawFunctionType>(T);
    auto *FunctionFull = TypeToNode.at({ F, TypeNode::Kind::FullType });
    auto *FunctionName = TypeToNode.at({ F, TypeNode::Kind::TypeName });

    auto RegArgTypeRefs = llvm::map_range(F->Arguments, GetTypeRef);
    auto RegRetValTypeRefs = llvm::map_range(F->ReturnValues, GetTypeRef);

    for (const auto &RegType :
         llvm::concat<const model::QualifiedType>(RegArgTypeRefs,
                                                  RegRetValTypeRefs)) {

      TypeDependencyNode *FullDep = getDependencyForFullType(RegType,
                                                             TypeToNode);
      Deps.push_back({ FunctionFull, FullDep });
      TypeDependencyNode *NameDep = getDependencyForTypeName(RegType,
                                                             TypeToNode);
      Deps.push_back({ FunctionName, NameDep });
      revng_log(Log,
                getNodeLabel(FunctionFull)
                  << " depends on " << getNodeLabel(FullDep));
      revng_log(Log,
                getNodeLabel(FunctionName)
                  << " depends on " << getNodeLabel(NameDep));
    }
  } break;

  case model::TypeKind::CABIFunctionType: {
    auto *F = cast<model::CABIFunctionType>(T);
    auto *FunctionFull = TypeToNode.at({ F, TypeNode::Kind::FullType });
    auto *FunctionName = TypeToNode.at({ F, TypeNode::Kind::TypeName });

    auto RegArgTypeRefs = llvm::map_range(F->Arguments, GetTypeRef);
    auto RegRetTypeRefs = llvm::ArrayRef(std::as_const(F->ReturnType));

    for (const auto &RegType :
         llvm::concat<const model::QualifiedType>(RegArgTypeRefs,
                                                  RegRetTypeRefs)) {

      TypeDependencyNode *FullDep = getDependencyForFullType(RegType,
                                                             TypeToNode);
      Deps.push_back({ FunctionFull, FullDep });
      TypeDependencyNode *NameDep = getDependencyForTypeName(RegType,
                                                             TypeToNode);
      Deps.push_back({ FunctionName, NameDep });
      revng_log(Log,
                getNodeLabel(FunctionFull)
                  << " depends on " << getNodeLabel(FullDep));
      revng_log(Log,
                getNodeLabel(FunctionName)
                  << " depends on " << getNodeLabel(NameDep));
    }

  } break;

  default:
    revng_abort();
  }

  for (const auto &[From, To] : Deps) {
    revng_log(Log,
              "Adding edge " << getNodeLabel(From) << " --> "
                             << getNodeLabel(To));
    From->addSuccessor(To);
  }
}

static void printDefinition(const model::Type &T,
                            llvm::raw_ostream &Header,
                            QualifiedTypeNameMap &AdditionalTypeNames) {
  if (Log.isEnabled())
    Header << "// Definition of " << getNameFromYAMLScalar(T.key()) << '\n';

  revng_log(Log, "Defining " << getNameFromYAMLScalar(T.key()));

  switch (T.Kind) {

  case model::TypeKind::Invalid: {
    if (Log.isEnabled())
      Header << "// invalid\n";
  } break;

  case model::TypeKind::Primitive: {
    printDeclaration(cast<model::PrimitiveType>(T), Header);
  } break;

  case model::TypeKind::Enum: {
    printDeclaration(cast<model::EnumType>(T), Header);
  } break;

  case model::TypeKind::Struct: {
    printDefinition(cast<model::StructType>(T), Header);
  } break;

  case model::TypeKind::Union: {
    printDefinition(cast<model::UnionType>(T), Header);
  } break;

  case model::TypeKind::Typedef: {
    printDeclaration(cast<model::TypedefType>(T), Header);
  } break;

  case model::TypeKind::RawFunctionType: {
    printDeclaration(cast<model::RawFunctionType>(T), Header);
  } break;

  case model::TypeKind::CABIFunctionType: {
    printDeclaration(cast<model::CABIFunctionType>(T),
                     Header,
                     AdditionalTypeNames);
  } break;

  default:
    revng_abort();
  }
}

static DependencyGraph buildDependencyGraph(const auto &Types) {
  DependencyGraph Dependencies;

  // Create nodes
  for (const UpcastablePointer<model::Type> &MT : Types)
    Dependencies.addNode(MT.get());

  // Compute dependencies and add them to the graph
  for (const UpcastablePointer<model::Type> &MT : Types)
    registerDependencies(MT.get(), Dependencies.TypeNodes());

  // if (Log.isEnabled())
  // llvm::ViewGraph(&DependencyGraph, "type-deps.dot");

  return Dependencies;
}

/// Print all type definitions for the types in the model
static void printTypeDefinitions(const model::Binary &Model,
                                 llvm::raw_ostream &Header,
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
      /*
      // llvm::post_order_ext(ExtendedDepGraphNode(Root), FullyDefined)) {
      // for (auto *Node :
      //     llvm::post_order_ext(FullDefDepGraphNode(Root), FullyDefined))
      {
      //  for (auto *Child : llvm::children<TypedefDepGraphNode>(Node)) {
      //    for (auto *TypedefUnderlying :
      //    llvm::post_order_ext(TypedefDepGraphNode(Child), FullyDefined))
      {
      //      if (FullyDefined.count(TypedefUnderlying))
      //        continue;
      //  printDefinition(TypedefUnderlying->T, Header);
      //  FullyDefined.insert(TypedefUnderlying);
      //  ForwardDeclared.insert(TypedefUnderlying);
      //  }
      //  }

      // for (const auto *ForwardDep :
      // llvm::children(ForwardDeclDepGraphNode(Root))) {
      // FullyDefined.count(ForwardDep)) {
      // }
      // }
      */
      const model::Type *NodeT = Node->T;
      const auto DeclKind = Node->K;
      constexpr auto TypeName = TypeNode::Kind::TypeName;
      constexpr auto FullType = TypeNode::Kind::FullType;
      if (DeclKind == FullType) {

        // When emitting a full definition we also want to emit a forward
        // declaration first, if it wasn't already emitted somewhere else.
        if (Defined.insert(TypeNodes.at({ NodeT, TypeName })).second)
          printDeclaration(*NodeT, Header, AdditionalTypeNames);

        if (not declarationIsDefinition(NodeT))
          printDefinition(*NodeT, Header, AdditionalTypeNames);

        // This is always a full type definition
        Defined.insert(TypeNodes.at({ NodeT, FullType }));
      } else {
        printDeclaration(*NodeT, Header, AdditionalTypeNames);
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

void dumpFunctionDecl(const model::Identifier FunctionName,
                      const model::Type *FT,
                      llvm::raw_ostream &Header,
                      QualifiedTypeNameMap &AdditionalTypeNames) {

  revng_assert(FunctionName.verify());

  if (const auto *RF = dyn_cast<model::RawFunctionType>(FT)) {
    Header << getRawFunctionReturnTypeName(*RF) << " " << FunctionName;

    if (RF->Arguments.empty()) {
      Header << "(void);\n";
    } else {
      const llvm::StringRef Open = "(";
      const llvm::StringRef Comma = ", ";
      llvm::StringRef Separator = Open;
      for (const auto &Arg : RF->Arguments) {
        Header << Separator << printNamedCInstance(Arg.Type, Arg.name());
        Separator = Comma;
      }
      Header << ");\n";
    }

  } else if (const auto *CF = dyn_cast<model::CABIFunctionType>(FT)) {
    Header << getCABIFunctionReturnTypeName(*CF, Header, AdditionalTypeNames)
           << " " << FunctionName;

    auto ArgDecls = getCABIFunctionArgumentDeclarations(*CF,
                                                        Header,
                                                        AdditionalTypeNames);

    if (ArgDecls.empty()) {
      Header << "(void);\n";
    } else {
      const llvm::StringRef Open = "(";
      const llvm::StringRef Comma = ", ";
      llvm::StringRef Separator = Open;
      for (const auto &ArgDecl : ArgDecls) {
        Header << Separator << ArgDecl;
        Separator = Comma;
      }
      Header << ");\n";
    }

  } else {
    revng_abort();
  }
}

bool dumpModelToHeader(const model::Binary &Model, llvm::raw_ostream &Header) {

  revng_assert(Model.verify(true));

  Header << "#include <stdint.h>\n";

  QualifiedTypeNameMap AdditionalTypeNames;
  printTypeDefinitions(Model, Header, AdditionalTypeNames);

  for (const model::Function &MF : Model.Functions) {
    // Ignore fake functions
    if (MF.Type == model::FunctionType::Fake)
      continue;

    const model::Type *FT = MF.Prototype.get();
    auto FName = model::Identifier::fromString(MF.name());
    dumpFunctionDecl(FName, FT, Header, AdditionalTypeNames);
  }

  for (const model::DynamicFunction &MF : Model.ImportedDynamicFunctions) {
    const model::Type *FT = MF.Prototype.get();
    auto FName = model::Identifier::fromString(MF.name());
    dumpFunctionDecl(FName, FT, Header, AdditionalTypeNames);
  }

  // TODO: eventually we should emit types and declarations of global variables
  // representing types and data containted in segments.

  return true;
}
