//
// Copyright (c) rev.ng Srls. See LICENSE.md for details.
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
#include "revng/Support/Assert.h"
#include "revng/Support/Debug.h"
#include "revng/Support/YAMLTraits.h"

#include "revng-c/HeadersGeneration/ModelToHeader.h"
#include "revng-c/Support/ModelHelpers.h"
#include "revng-c/TypeNames/ModelTypeNames.h"

#include "DependencyGraph.h"

using ArtificialTypes::ArrayWrapperFieldName;

using llvm::cast;
using llvm::isa;
using llvm::Twine;

static Logger<> Log{ "model-to-header" };

static bool declarationIsDefinition(const model::Type *T) {
  return not isa<model::StructType>(T) and not isa<model::UnionType>(T);
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
    if (Log.isEnabled())
      Header << "// not necessary, already in revngfloat.h\n";
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
    revng_abort("Invalid primitive type");
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

    Header << "  " << getNamedCInstance(Field.Type, Field.name()) << ";\n";

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
    Header << "  " << getNamedCInstance(Field.Type, Field.name()) << ";\n";

  Header << "};\n";
}

static void
printDeclaration(const model::TypedefType &TD, llvm::raw_ostream &Header) {
  Header << "typedef " << getNamedCInstance(TD.UnderlyingType, TD.name())
         << ";\n";
}

/// Generate the definition of a new struct type that wraps all the
///        return values of \a F. The name of the struct type is provided by the
///        caller.
static void generateReturnValueWrapper(const model::RawFunctionType &F,
                                       llvm::raw_ostream &Header) {
  revng_assert(F.ReturnValues.size() > 1);
  if (Log.isEnabled())
    Header << "// definition the of return type needed\n";

  Header << "typedef struct __attribute__((packed)) {\n";

  for (auto &Group : llvm::enumerate(F.ReturnValues)) {
    const model::QualifiedType &RetTy = Group.value().Type;
    const auto &FieldName = getReturnField(F, Group.index());
    Header << "  " << getNamedCInstance(RetTy, FieldName) << ";\n";
  }

  Header << "} " << getReturnTypeName(F) << ";\n ";
}

/// If the function has more than one return value, generate a wrapper
///        struct that contains them.
static void printRawFunctionWrappers(const model::RawFunctionType *F,
                                     llvm::raw_ostream &Header) {
  if (F->ReturnValues.size() > 1)
    generateReturnValueWrapper(*F, Header);

  for (auto &Arg : F->Arguments)
    revng_assert(Arg.Type.isScalar());
}

/// Print a typedef for a RawFunctionType, that can be used when you have
///        a variable that is a pointer to a function.
static void
printDeclaration(const model::RawFunctionType &F, llvm::raw_ostream &Header) {
  printRawFunctionWrappers(&F, Header);

  Header << "typedef ";
  // In this case, we are defining a type for the function, not the function
  // itself, so the token right before the parenthesis is the name of the type.
  printFunctionPrototype(F, getTypeName(F), Header);
  Header << ";\n";
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

/// Generate the definition of a new struct type that wraps \a ArrayType.
///        This is used to wrap array arguments or array return values of
///        CABIFunctionTypes.
static void generateArrayWrapper(const model::QualifiedType &ArrayType,
                                 llvm::raw_ostream &Header,
                                 QualifiedTypeNameMap &NamesCache) {
  revng_assert(ArrayType.isArray());
  auto WrapperName = getArrayWrapper(ArrayType);

  // Check if the wrapper was already added
  bool IsNew = NamesCache.emplace(ArrayType, WrapperName).second;
  if (not IsNew)
    return;

  Header << "typedef struct __attribute__((packed)) {\n";
  Header << "  " << getNamedCInstance(ArrayType, ArrayWrapperFieldName)
         << ";\n";
  Header << "} " << WrapperName << ";\n ";
}

/// If the return value or any of the arguments is an array, generate
///        a wrapper struct for each of them, if it's not already in the cache.
static void printCABIFunctionWrappers(const model::CABIFunctionType *F,
                                      llvm::raw_ostream &Header,
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
                             llvm::raw_ostream &Header,
                             QualifiedTypeNameMap &NamesCache) {
  printCABIFunctionWrappers(&F, Header, NamesCache);

  Header << "typedef ";
  // In this case, we are defining a type for the function, not the function
  // itself, so the token right before the parenthesis is the name of the type.
  printFunctionPrototype(F, getTypeName(F), Header);
  Header << ";\n";
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

static void printDefinition(const model::Type &T,
                            llvm::raw_ostream &Header,
                            QualifiedTypeNameMap &AdditionalTypeNames) {
  if (Log.isEnabled())
    Header << "// Definition of " << getNameFromYAMLScalar(T.key()) << '\n';

  revng_log(Log, "Defining " << getNameFromYAMLScalar(T.key()));

  if (declarationIsDefinition(&T)) {
    printDeclaration(T, Header, AdditionalTypeNames);
  } else {
    switch (T.Kind) {

    case model::TypeKind::Invalid: {
      if (Log.isEnabled())
        Header << "// invalid\n";
    } break;

    case model::TypeKind::Struct: {
      printDefinition(cast<model::StructType>(T), Header);
    } break;

    case model::TypeKind::Union: {
      printDefinition(cast<model::UnionType>(T), Header);
    } break;

    default:
      revng_abort();
    }
  }
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

bool dumpModelToHeader(const model::Binary &Model, llvm::raw_ostream &Header) {
  Header << "#include <stdint.h>\n";
  Header << "#include <stdbool.h>\n";
  Header << "#include \"revngfloat.h\"\n\n";

  Header << "#ifndef NULL \n"
         << "#define NULL (0) \n"
         << "#endif \n\n";

  QualifiedTypeNameMap AdditionalTypeNames;
  printTypeDefinitions(Model, Header, AdditionalTypeNames);

  for (const model::Function &MF : Model.Functions) {
    // Ignore fake functions
    if (MF.Type == model::FunctionType::Fake)
      continue;

    const model::Type *FT = MF.Prototype.get();
    auto FName = model::Identifier::fromString(MF.name());

    if (Log.isEnabled()) {
      Header << "/* Analyzing Model function " << FName << "\n";
      serialize(Header, MF);
      Header << "Prototype\n";
      serialize(Header, *FT);
      Header << "*/\n";
    }

    printFunctionPrototype(*FT, FName, Header);
    Header << ";\n";
  }

  for (const model::DynamicFunction &MF : Model.ImportedDynamicFunctions) {
    const model::Type *FT = MF.prototype(Model).get();
    revng_assert(FT != nullptr);
    auto FName = model::Identifier::fromString(MF.name());

    if (Log.isEnabled()) {
      Header << "/* Analyzing dynamic function " << FName << "\n";
      serialize(Header, MF);
      Header << "Prototype\n";
      serialize(Header, *FT);
      Header << "*/\n";
    }
    printFunctionPrototype(*FT, FName, Header);
    Header << ";\n";
  }

  // TODO: eventually we should emit types and declarations of global variables
  // representing types and data containted in segments.

  return true;
}
