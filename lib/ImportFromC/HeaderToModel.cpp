//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/TextDiagnostic.h"

#include "revng/Model/Processing.h"
#include "revng/Model/QualifiedType.h"
#include "revng/Model/Register.h"
#include "revng/Model/TypeDefinition.h"
#include "revng/Support/Debug.h"

#include "revng-c/Pipes/Ranks.h"
#include "revng-c/Support/ModelHelpers.h"
#include "revng-c/Support/PTMLC.h"
#include "revng-c/TypeNames/ModelTypeNames.h"

#include "HeaderToModel.h"

using namespace model;
using namespace revng;

static Logger<> Log("header-to-model");
static constexpr std::string_view InputCFile = "revng-input.c";
static constexpr std::string_view PrimitiveTypeHeader = "revng-primitive-"
                                                        "types.h";
static constexpr llvm::StringRef RawABIPrefix = "raw_";

static constexpr const char *ABIAnnotatePrefix = "abi:";
static constexpr size_t
  ABIAnnotatePrefixLength = std::char_traits<char>::length(ABIAnnotatePrefix);

static constexpr const char *RegAnnotatePrefix = "reg:";
static constexpr size_t
  RegAnnotatePrefixLength = std::char_traits<char>::length(RegAnnotatePrefix);

static constexpr const char *EnumAnnotatePrefix = "enum_underlying_type:";
static constexpr size_t
  EnumAnnotatePrefixLength = std::char_traits<char>::length(EnumAnnotatePrefix);

template<typename T>
concept HasCustomName = requires(const T &Element) {
  { Element.CustomName() } -> std::same_as<const model::Identifier &>;
  { Element.name() } -> std::same_as<model::Identifier>;
};

template<HasCustomName T>
static void setCustomName(T &Element, llvm::StringRef NewName) {
  if (Element.name() != NewName)
    Element.CustomName() = NewName;
}

namespace clang {
namespace tooling {

class HeaderToModel : public ASTConsumer {
public:
  HeaderToModel(TupleTree<model::Binary> &Model,
                std::optional<model::TypeDefinition *> &Type,
                std::optional<model::Function> &Function,
                std::optional<ParseCCodeError> &Error,
                enum ImportFromCOption AnalysisOption) :
    Model(Model),
    Type(Type),
    Function(Function),
    Error(Error),
    AnalysisOption(AnalysisOption) {
    // Either one of these two should be null, since the editing features are
    // exclusive.
    revng_assert(not Type or not Function);
  }

  virtual void HandleTranslationUnit(ASTContext &Context) override;

private:
  TupleTree<model::Binary> &Model;
  std::optional<model::TypeDefinition *> &Type;
  std::optional<model::Function> &Function;
  std::optional<ParseCCodeError> &Error;
  enum ImportFromCOption AnalysisOption;
};

class DeclVisitor : public clang::RecursiveASTVisitor<DeclVisitor> {
private:
  TupleTree<model::Binary> &Model;
  ASTContext &Context;
  std::optional<model::TypeDefinition *> &Type;
  std::optional<model::Function> &Function;
  std::optional<revng::ParseCCodeError> &Error;
  enum ImportFromCOption AnalysisOption;

  // These are used for reporting source location of an error, if any.
  unsigned CurrentLineNumber = 0;
  unsigned CurrentColumnNumber = 0;

  // Used to remember return values locations when parsing struct representing
  // the multi-reg return value. Represents register ID and mode::Type.
  using ModelType = std::pair<model::TypeDefinitionPath,
                              std::vector<Qualifier>>;
  using RawLocation = std::pair<model::Register::Values, ModelType>;
  std::optional<llvm::SmallVector<RawLocation, 4>> MultiRegisterReturnValue;

public:
  explicit DeclVisitor(TupleTree<model::Binary> &Model,
                       ASTContext &Context,
                       std::optional<model::TypeDefinition *> &Type,
                       std::optional<model::Function> &Function,
                       std::optional<ParseCCodeError> &Error,
                       enum ImportFromCOption AnalysisOption);

  void run(clang::TranslationUnitDecl *TUD);
  bool TraverseDecl(clang::Decl *D);

  bool VisitFunctionDecl(const clang::FunctionDecl *FD);
  bool VisitRecordDecl(const clang::RecordDecl *RD);
  bool VisitEnumDecl(const EnumDecl *D);
  bool VisitTypedefDecl(const TypedefDecl *D);
  bool VisitFunctionPrototype(const FunctionProtoType *FP,
                              std::optional<llvm::StringRef> TheABI);

private:
  // This checks that the declaration is the one user provided as input.
  bool comesFromInternalFile(const clang::Decl *D);

  // This checks that the declaration comes from revng-primitive-types header
  // file.
  bool comesFromPrimitiveTypesHeader(const clang::RecordDecl *RD);

  // Set up line and column for the declaratrion.
  void setupLineAndColumn(const clang::Decl *D);

  // Handle clang's Struct type.
  bool handleStructType(const clang::RecordDecl *RD);
  // Handle clang's Union type.
  bool handleUnionType(const clang::RecordDecl *RD);

  // Convert clang::type to model::type.
  std::optional<model::TypeDefinitionPath>
  getOrCreatePrimitive(const BuiltinType *UnderlyingBuiltin, QualType Type);

  // Get model type for clang::RecordType (Struct/Unoion).
  std::optional<model::TypeDefinitionPath>
  getTypeForRecordType(const clang::RecordType *RecordType,
                       const QualType &ClangType);

  // Get model type for clang::EnumType.
  std::optional<model::TypeDefinitionPath>
  getTypeForEnumType(const clang::EnumType *EnumType);

  std::optional<model::TypeDefinitionPath>
  getTypeByNameOrID(llvm::StringRef Name, TypeDefinitionKind::Values Kind);

  std::optional<model::QualifiedType>
  getModelTypeForClangType(const QualType &QT);

  std::optional<model::TypeDefinitionPath>
  getEnumUnderlyingType(const std::string &TypeName);
};

DeclVisitor::DeclVisitor(TupleTree<model::Binary> &Model,
                         ASTContext &Context,
                         std::optional<model::TypeDefinition *> &Type,
                         std::optional<model::Function> &Function,
                         std::optional<ParseCCodeError> &Error,
                         enum ImportFromCOption AnalysisOption) :
  Model(Model),
  Context(Context),
  Type(Type),
  Function(Function),
  Error(Error),
  AnalysisOption(AnalysisOption) {
}

// Parse ABI from the annotate attribute content.
static std::optional<std::string> getABI(llvm::StringRef ABIAnnotate) {
  if (not ABIAnnotate.startswith(ABIAnnotatePrefix))
    return std::nullopt;
  return std::string(ABIAnnotate.substr(ABIAnnotatePrefixLength));
}

static model::Architecture::Values getRawABIArchitecture(llvm::StringRef ABI) {
  revng_assert(ABI.starts_with(RawABIPrefix));
  return model::Architecture::fromName(ABI.substr(RawABIPrefix.size()));
}

// Parse location of parameter (`reg:` or `stack`).
static std::optional<std::string> getLoc(llvm::StringRef Annotate) {
  if (Annotate == "stack")
    return Annotate.str();

  if (not Annotate.startswith(RegAnnotatePrefix))
    return std::nullopt;
  return std::string(Annotate.substr(RegAnnotatePrefixLength));
}

// Parse enum's underlying type from the annotate attribute content.
static std::optional<std::string>
parseEnumUnderlyingType(llvm::StringRef Annotate) {
  if (not Annotate.startswith(EnumAnnotatePrefix))
    return std::nullopt;
  return std::string(Annotate.substr(EnumAnnotatePrefixLength));
}

std::optional<model::TypeDefinitionPath>
DeclVisitor::getEnumUnderlyingType(const std::string &TypeName) {
  auto MaybePrimitive = model::PrimitiveDefinition::fromName(TypeName);
  if (not MaybePrimitive) {
    revng_log(Log, "Not a primitive type");
    return std::nullopt;
  }

  return Model->getPrimitiveType(MaybePrimitive->PrimitiveKind(),
                                 MaybePrimitive->Size());
}

std::optional<model::TypeDefinitionPath>
DeclVisitor::getOrCreatePrimitive(const BuiltinType *UnderlyingBuiltin,
                                  QualType Type) {
  revng_assert(UnderlyingBuiltin);

  auto AsElaboratedType = Type->getAs<ElaboratedType>();
  if (not AsElaboratedType) {
    PrintingPolicy Policy(Context.getLangOpts());
    std::string ErrorMessage = "revng: Builtin type `"
                               + UnderlyingBuiltin->getName(Policy).str()
                               + "` not allowed, please use a revng "
                                 "model::PrimitiveDefinition instead";
    Error = { ErrorMessage, CurrentLineNumber, CurrentColumnNumber };

    return std::nullopt;
  }

  while (auto Typedef = AsElaboratedType->getAs<TypedefType>()) {
    auto TheUnderlyingType = Typedef->getDecl()->getUnderlyingType();
    if (not TheUnderlyingType->getAs<ElaboratedType>())
      break;
    AsElaboratedType = TheUnderlyingType->getAs<ElaboratedType>();
  }

  std::string TypeName = AsElaboratedType->getNamedType().getAsString();
  auto MaybePrimitive = model::PrimitiveDefinition::fromName(TypeName);
  if (not MaybePrimitive.has_value()) {
    std::string ErrorMessage = "revng: `"
                               + AsElaboratedType->getNamedType().getAsString()
                               + "`, please use a revng "
                                 "model::PrimitiveDefinition instead";
    Error = { ErrorMessage, CurrentLineNumber, CurrentColumnNumber };

    return std::nullopt;
  }

  model::TypeDefinitionPath Result;
  switch (UnderlyingBuiltin->getKind()) {
  case BuiltinType::UInt128: {
    return Model->getPrimitiveType(model::PrimitiveKind::Unsigned, 16);
  }
  case BuiltinType::Int128: {
    return Model->getPrimitiveType(model::PrimitiveKind::Signed, 16);
  }
  case BuiltinType::ULongLong:
  case BuiltinType::ULong: {
    return Model->getPrimitiveType(model::PrimitiveKind::Unsigned, 8);
  }
  case BuiltinType::LongLong:
  case BuiltinType::Long: {
    return Model->getPrimitiveType(model::PrimitiveKind::Signed, 8);
  }
  case BuiltinType::WChar_U:
  case BuiltinType::UInt: {
    return Model->getPrimitiveType(model::PrimitiveKind::Unsigned, 4);
  }
  case BuiltinType::WChar_S:
  case BuiltinType::Char32:
  case BuiltinType::Int: {
    return Model->getPrimitiveType(model::PrimitiveKind::Signed, 4);
  }
  case BuiltinType::Char16:
  case BuiltinType::Short: {
    return Model->getPrimitiveType(model::PrimitiveKind::Signed, 2);
  }
  case BuiltinType::UShort: {
    return Model->getPrimitiveType(model::PrimitiveKind::Unsigned, 2);
  }
  case BuiltinType::Char_S:
  case BuiltinType::SChar:
  case BuiltinType::Char8:
  case BuiltinType::Bool: {
    return Model->getPrimitiveType(model::PrimitiveKind::Signed, 1);
  }
  case BuiltinType::Char_U:
  case BuiltinType::UChar: {
    return Model->getPrimitiveType(model::PrimitiveKind::Unsigned, 1);
  }
  case BuiltinType::Void: {
    return Model->getPrimitiveType(model::PrimitiveKind::Void, 0);
  }
  case BuiltinType::Float16: {
    return Model->getPrimitiveType(model::PrimitiveKind::Float, 2);
  }

  case BuiltinType::Float: {
    return Model->getPrimitiveType(model::PrimitiveKind::Float, 4);
  }
  case BuiltinType::Double: {
    return Model->getPrimitiveType(model::PrimitiveKind::Float, 8);
  }
  case BuiltinType::Float128:
  case BuiltinType::LongDouble: {
    return Model->getPrimitiveType(model::PrimitiveKind::Float, 16);
  }

  default: {
    revng_log(Log, "Unable to handle a primitive type");
    break;
  }
  }

  return std::nullopt;
}

namespace TDKind = model::TypeDefinitionKind;
std::optional<model::TypeDefinitionPath>
DeclVisitor::getTypeByNameOrID(llvm::StringRef Name,
                               TypeDefinitionKind::Values Kind) {
  const bool IsStruct = Kind == TDKind::StructDefinition;
  const bool IsUnion = Kind == TDKind::UnionDefinition;
  const bool IsEnum = Kind == TDKind::EnumDefinition;
  const bool IsCABI = Kind == TDKind::CABIFunctionDefinition;
  const bool IsRaw = Kind == TDKind::RawFunctionDefinition;

  // Find by name first.
  for (auto &Type : Model->TypeDefinitions()) {
    if (IsStruct and not llvm::isa<model::StructDefinition>(Type.get()))
      continue;

    if (IsUnion and not llvm::isa<model::UnionDefinition>(Type.get()))
      continue;

    if (IsEnum and not llvm::isa<model::EnumDefinition>(Type.get()))
      continue;

    if (IsCABI and not llvm::isa<model::CABIFunctionDefinition>(Type.get()))
      continue;

    if (IsRaw and not llvm::isa<model::RawFunctionDefinition>(Type.get()))
      continue;

    if (Type->CustomName() == Name)
      return Model->getTypeDefinitionPath(Type.get());
  }

  size_t LocationOfID = Name.rfind("_");

  if (LocationOfID != std::string::npos) {
    std::string ID = std::string(Name.substr(LocationOfID + 1));
    uint64_t TypeID;

    std::istringstream TheStream(ID);
    TheStream >> TypeID;

    auto KeyType = model::TypeDefinition::Key{ TypeID, Kind };

    auto TheType = Model->getTypeDefinitionPath(KeyType);
    if (TheType.get())
      return TheType;
  }

  return std::nullopt;
}

std::optional<model::TypeDefinitionPath>
DeclVisitor::getTypeForRecordType(const clang::RecordType *RecordType,
                                  const QualType &ClangType) {
  revng_assert(RecordType);

  // Check if it is a primitive type described with a struct.
  if (comesFromPrimitiveTypesHeader(RecordType->getDecl())) {
    const TypedefType *AsTypedef = ClangType->getAs<TypedefType>();
    if (not AsTypedef) {
      revng_log(Log,
                "There should be a typedef for struct that defines the "
                "primitive type");
      return std::nullopt;
    }
    auto TypeName = AsTypedef->getDecl()->getName();
    auto MaybePrimitive = model::PrimitiveDefinition::fromName(TypeName);
    revng_assert(MaybePrimitive);

    return Model->getPrimitiveType(MaybePrimitive->PrimitiveKind(),
                                   MaybePrimitive->Size());
  }

  auto Name = RecordType->getDecl()->getName();
  if (Name.empty()) {
    revng_log(Log, "Unable to find record type without name");
    return std::nullopt;
  }

  if (RecordType->isStructureType()) {
    auto TheStructType = getTypeByNameOrID(Name, TDKind::StructDefinition);
    if (TheStructType)
      return *TheStructType;
  } else if (RecordType->isUnionType()) {
    auto TheUnionType = getTypeByNameOrID(Name, TDKind::UnionDefinition);
    if (TheUnionType)
      return *TheUnionType;
  }

  revng_log(Log, "Unable to find record type " << Name);
  return std::nullopt;
}

std::optional<model::TypeDefinitionPath>
DeclVisitor::getTypeForEnumType(const clang::EnumType *EnumType) {
  revng_assert(EnumType);
  revng_assert(AnalysisOption != ImportFromCOption::EditFunctionPrototype);

  auto EnumName = EnumType->getDecl()->getName();
  if (EnumName.empty()) {
    revng_log(Log, "Unable to find enum type without name");
    return std::nullopt;
  }

  auto TheEnumType = getTypeByNameOrID(EnumName, TDKind::EnumDefinition);
  if (TheEnumType)
    return *TheEnumType;

  revng_log(Log, "Unable to find enum type " << EnumName);
  return std::nullopt;
}

bool DeclVisitor::comesFromInternalFile(const clang::Decl *D) {
  SourceManager &SM = Context.getSourceManager();
  PresumedLoc Loc = SM.getPresumedLoc(D->getLocation());
  if (!Loc.isValid()) {
    revng_log(Log, "Invalid source location found");
    return false;
  }

  StringRef TheFileName(Loc.getFilename());
  // Process the new type only.
  if (TheFileName.contains(InputCFile))
    return true;

  return false;
}

bool DeclVisitor::comesFromPrimitiveTypesHeader(const clang::RecordDecl *RD) {
  SourceManager &SM = Context.getSourceManager();
  PresumedLoc Loc = SM.getPresumedLoc(RD->getLocation());
  if (!Loc.isValid()) {
    revng_log(Log, "Invalid source location found");
    return false;
  }

  StringRef TheFileName(Loc.getFilename());
  if (TheFileName.contains(PrimitiveTypeHeader))
    return true;

  return false;
}

void DeclVisitor::setupLineAndColumn(const clang::Decl *D) {
  SourceManager &SM = Context.getSourceManager();
  PresumedLoc Loc = SM.getPresumedLoc(D->getLocation());
  if (!Loc.isValid()) {
    revng_log(Log, "Invalid source location found");
    return;
  }

  CurrentLineNumber = Loc.getLine();
  CurrentColumnNumber = Loc.getColumn();
}

std::optional<model::QualifiedType>
DeclVisitor::getModelTypeForClangType(const QualType &QT) {
  std::optional<model::TypeDefinitionPath> TheTypePath;
  std::vector<Qualifier> Qualifiers;

  if (QT.isConstQualified())
    Qualifiers.push_back(Qualifier::createConst());

  const BuiltinType *AsBuiltinType = QT->getAs<BuiltinType>();
  if (AsBuiltinType) {
    TheTypePath = getOrCreatePrimitive(AsBuiltinType, QT);
    if (not TheTypePath)
      return std::nullopt;
  } else if (QT->isPointerType()) {
    auto PointerSize = getPointerSize(Model->Architecture());
    Qualifiers.push_back({ Qualifier::createPointer(PointerSize) });
    QualType BaseType;
    BaseType = QT->getAs<PointerType>()->getPointeeType();
    // Handle pointers-to-pointers-...
    while (BaseType->isPointerType()) {
      Qualifiers.push_back({ Qualifier::createPointer(PointerSize) });
      BaseType = BaseType->getAs<PointerType>()->getPointeeType();
    }
    // NOTE: For Typedefs it will consider the underlying type.
    if (const BuiltinType *PointeeAsBuiltinType = BaseType
                                                    ->getAs<BuiltinType>()) {
      TheTypePath = getOrCreatePrimitive(PointeeAsBuiltinType, BaseType);
      if (not TheTypePath)
        return std::nullopt;
    } else if (const RecordType *AsRecordType = BaseType->getAs<RecordType>()) {
      TheTypePath = getTypeForRecordType(AsRecordType, BaseType);
      if (not TheTypePath)
        return std::nullopt;
    } else if (const EnumType *AsEnum = BaseType->getAs<EnumType>()) {
      TheTypePath = getTypeForEnumType(AsEnum);
      if (not TheTypePath)
        return std::nullopt;
    } else if (const FunctionProtoType *AsFn = BaseType
                                                 ->getAs<FunctionProtoType>()) {
      const TypedefType *AsTypedef = BaseType->getAs<TypedefType>();
      if (not AsTypedef) {
        revng_log(Log, "There should be a typedef for function type");
        return std::nullopt;
      }

      auto FunctionName = AsTypedef->getDecl()->getName();
      auto CABIFunctionTypeKind = TDKind::CABIFunctionDefinition;
      auto TheCABIFunctionType = getTypeByNameOrID(FunctionName,
                                                   CABIFunctionTypeKind);

      auto RawFunctionTypeKind = TDKind::RawFunctionDefinition;
      auto TheRawFunctionType = getTypeByNameOrID(FunctionName,
                                                  RawFunctionTypeKind);

      if (not TheCABIFunctionType and not TheRawFunctionType) {
        revng_log(Log, "Did not find function type in the model");
        return std::nullopt;
      }
      TheTypePath = TheCABIFunctionType ? *TheCABIFunctionType :
                                          *TheRawFunctionType;
    } else {
      revng_log(Log, "Unsupported type used as pointer");
      return std::nullopt;
    }
  } else if (QT->isArrayType()) {
    uint64_t NumberOfElements = 0;
    if (const auto *CAT = dyn_cast<ConstantArrayType>(QT)) {
      NumberOfElements = CAT->getSize().getZExtValue();
    } else {
      // Here we can face clang::VariableArrayType and
      // clang::IncompleteArrayType.
      revng_log(Log, "Unsupported type used as an array");
      return std::nullopt;
    }

    Qualifiers.push_back({ Qualifier::createArray(NumberOfElements) });
    QualType ElementType = Context.getBaseElementType(QT);
    const BuiltinType *ElementAsBuiltin = ElementType->getAs<BuiltinType>();
    if (ElementAsBuiltin) {
      TheTypePath = getOrCreatePrimitive(ElementAsBuiltin, ElementType);
      if (not TheTypePath)
        return std::nullopt;
    } else if (const RecordType *AsRecordType = ElementType
                                                  ->getAs<RecordType>()) {
      TheTypePath = getTypeForRecordType(AsRecordType, ElementType);
      if (not TheTypePath)
        return std::nullopt;
    } else if (const EnumType *AsEnum = ElementType->getAs<EnumType>()) {
      TheTypePath = getTypeForEnumType(AsEnum);
      if (not TheTypePath)
        return std::nullopt;
    } else {
      revng_log(Log, "Unsupported array element type");
      return std::nullopt;
    }
  } else if (const RecordType *AsRecordType = QT->getAs<RecordType>()) {
    TheTypePath = getTypeForRecordType(AsRecordType, QT);
    if (not TheTypePath)
      return std::nullopt;
  } else if (const EnumType *AsEnum = QT->getAs<EnumType>()) {
    TheTypePath = getTypeForEnumType(AsEnum);
    if (not TheTypePath)
      return std::nullopt;
  } else {
    revng_log(Log, "Unsupported QualType");
    return std::nullopt;
  }

  revng_check(TheTypePath);
  QualifiedType Result(*TheTypePath, Qualifiers);

  return Result;
}

bool DeclVisitor::VisitFunctionDecl(const clang::FunctionDecl *FD) {
  if (not comesFromInternalFile(FD))
    return true;

  revng_assert(FD);
  revng_assert(AnalysisOption == ImportFromCOption::EditFunctionPrototype);
  std::optional<std::string> MaybeABI;
  std::vector<AnnotateAttr *> AnnotateAttrs;

  {
    // May be multiple Annotate attributes. One for ABI, one for return value.
    std::for_each(begin(FD->getAttrs()),
                  end(FD->getAttrs()),
                  [&](Attr *Attribute) {
                    if (auto *Annotation = dyn_cast<AnnotateAttr>(Attribute))
                      AnnotateAttrs.push_back(Annotation);
                  });

    for (auto *Annotate : AnnotateAttrs) {
      MaybeABI = getABI(Annotate->getAnnotation());
      if (MaybeABI)
        break;
    }

    if (not MaybeABI or MaybeABI->empty()) {
      revng_log(Log,
                "Functions should have attribute annotate with abi: "
                "specification");
      return false;
    }
  }

  revng_assert(MaybeABI.has_value());
  bool IsRawFunctionType = MaybeABI->starts_with(RawABIPrefix);
  auto NewType = IsRawFunctionType ?
                   makeTypeDefinition<RawFunctionDefinition>() :
                   makeTypeDefinition<CABIFunctionDefinition>();

  if (not IsRawFunctionType) {
    auto TheModelABI = model::ABI::fromName(*MaybeABI);
    if (TheModelABI == model::ABI::Invalid) {
      revng_log(Log, "Invalid ABI provided");
      return false;
    }

    auto FunctionType = cast<CABIFunctionDefinition>(NewType.get());
    FunctionType->ABI() = TheModelABI;
    auto TheRetClangType = FD->getReturnType();
    auto TheRetType = getModelTypeForClangType(TheRetClangType);
    if (not TheRetType) {
      revng_log(Log, "Unsupported type for function return value");
      return false;
    }

    FunctionType->ReturnType() = *TheRetType;

    // Handle params.
    uint32_t Index = 0;
    for (unsigned I = 0, N = FD->getNumParams(); I != N; ++I) {
      auto QT = FD->getParamDecl(I)->getType();
      auto ParamType = getModelTypeForClangType(QT);
      if (not ParamType) {
        revng_log(Log, "Unsupported type for function parameter");
        return false;
      }

      model::Argument &NewArgument = FunctionType->Arguments()[Index];
      setCustomName(NewArgument, FD->getParamDecl(I)->getName());
      NewArgument.Type() = *ParamType;
      ++Index;
    }
  } else {
    auto TheRetClangType = FD->getReturnType();
    auto TheRawFunctionType = cast<RawFunctionDefinition>(NewType.get());

    auto Architecture = getRawABIArchitecture(*MaybeABI);
    if (Architecture == model::Architecture::Invalid) {
      revng_log(Log, "Invalid raw abi architecture");
      return false;
    }
    TheRawFunctionType->Architecture() = Architecture;

    auto ReturnValuesInserter = TheRawFunctionType->ReturnValues()
                                  .batch_insert();

    // This represents multiple register location for return values.
    if (TheRetClangType->isStructureType()) {
      if (not MultiRegisterReturnValue) {
        revng_log(Log, "Return value should be already parsed");
        return false;
      }

      for (auto &ReturnValue : *MultiRegisterReturnValue) {
        NamedTypedRegister ReturnValueReg(ReturnValue.first);
        ReturnValueReg.Type() = { ReturnValue.second.first,
                                  ReturnValue.second.second };
        ReturnValuesInserter.insert(ReturnValueReg);
      }
    } else {
      // Return value as single register.
      std::for_each(begin(FD->getAttrs()),
                    end(FD->getAttrs()),
                    [&](Attr *Attribute) {
                      if (auto *Annotation = dyn_cast<AnnotateAttr>(Attribute))
                        AnnotateAttrs.push_back(Annotation);
                    });

      std::optional<std::string> ReturnValue;
      for (auto *Annotate : AnnotateAttrs) {
        ReturnValue = getLoc(Annotate->getAnnotation());
        if (ReturnValue)
          break;
      }

      if (not ReturnValue or ReturnValue->empty()) {
        revng_log(Log,
                  "Return value should have attribute annotate with reg or "
                  "stack");
        return false;
      }

      // TODO: Handle stack location.
      if (*ReturnValue == "stack") {
        revng_log(Log, "We don't support Return value on stack for now");
        return false;
      }

      auto TheRetType = getModelTypeForClangType(TheRetClangType);
      if (not TheRetType) {
        revng_log(Log, "Unsupported type for function return value");
        return false;
      }

      auto RegisterID = model::Register::fromCSVName(*ReturnValue,
                                                     Model->Architecture());
      if (RegisterID == model::Register::Invalid) {
        revng_log(Log, "Unsupported register location");
        return false;
      }

      NamedTypedRegister ReturnValueReg(RegisterID);
      ReturnValueReg.Type() = *TheRetType;
      ReturnValuesInserter.insert(ReturnValueReg);
    }

    auto ArgumentsInserter = TheRawFunctionType->Arguments().batch_insert();
    for (unsigned I = 0, N = FD->getNumParams(); I != N; ++I) {
      auto ParamDecl = FD->getParamDecl(I);
      if (ParamDecl->hasAttr<AnnotateAttr>()) {
        auto Annotate = std::find_if(begin(ParamDecl->getAttrs()),
                                     end(ParamDecl->getAttrs()),
                                     [&](Attr *Attribute) {
                                       return isa<AnnotateAttr>(Attribute);
                                     });
        auto Loc = getLoc(cast<AnnotateAttr>(*Annotate)->getAnnotation());
        if (not Loc or Loc->empty()) {
          revng_log(Log,
                    "Parameters should have attribute annotate with reg or "
                    "stack");
          return false;
        }

        // TODO: Handle stack location.
        if (*Loc == "stack") {
          revng_log(Log, "We don't support parameters on stack for now");
          return false;
        }

        auto LocationID = model::Register::fromCSVName(*Loc,
                                                       Model->Architecture());
        if (LocationID == model::Register::Invalid) {
          revng_log(Log, "Unsupported register location");
          return false;
        }

        auto QT = ParamDecl->getType();
        auto ParamType = getModelTypeForClangType(QT);
        if (not ParamType) {
          revng_log(Log, "Unsupported type for raw function parameter");
          return false;
        }

        NamedTypedRegister ParamReg(LocationID);
        ParamReg.Type() = *ParamType;
        ArgumentsInserter.insert(ParamReg);
      }
    }
  }

  // Update the name if in the case it got changed.
  auto &ModelFunction = Model->Functions()[Function->Entry()];
  setCustomName(ModelFunction, FD->getName());

  // Clone the other stuff.
  ModelFunction.OriginalName() = Function->OriginalName();
  ModelFunction.ExportedNames() = Function->ExportedNames();

  // TODO: remember/clone StackFrameType as well.

  auto Prototype = Model->recordNewType(std::move(NewType));
  ModelFunction.Prototype() = Prototype;

  return true;
}

bool DeclVisitor::VisitTypedefDecl(const TypedefDecl *D) {
  if (not comesFromInternalFile(D))
    return true;

  revng_assert(AnalysisOption != ImportFromCOption::EditFunctionPrototype);

  QualType TheType = D->getUnderlyingType();
  if (auto Fn = llvm::dyn_cast<FunctionProtoType>(TheType)) {
    // Parse the ABI from annotate attribute attached to the typedef
    // declaration. Please do note that annotations on the parameters are not
    // attached, so we will use default RawFunctionDefinition from the Model if
    // the abi is raw.
    // TODO: Should we change the annotate attached to function types to have
    // info about parameters in the toplevel annotate attribute attached to
    // the typedef itself?
    std::optional<std::string> TheABI;
    if (D->hasAttr<AnnotateAttr>()) {
      auto TheAnnotateAttr = std::find_if(begin(D->getAttrs()),
                                          end(D->getAttrs()),
                                          [&](Attr *Attribute) {
                                            return isa<AnnotateAttr>(Attribute);
                                          });

      TheABI = getABI(cast<AnnotateAttr>(*TheAnnotateAttr)->getAnnotation());
    }

    if (not TheABI or TheABI->empty()) {
      revng_log(Log, "Unable to parse `abi:` from the annotate attribute");
      return false;
    }

    return VisitFunctionPrototype(Fn, TheABI);
  }

  // Regular, non-function, typedef.
  auto ModelTypedefType = getModelTypeForClangType(TheType);
  if (not ModelTypedefType) {
    revng_log(Log, "Unsupported underlying type for typedef");
    return false;
  }
  auto TypeTypedef = model::makeTypeDefinition<model::TypedefDefinition>();
  if (AnalysisOption == ImportFromCOption::EditType)
    TypeTypedef->ID() = (*Type)->ID();

  auto TheTypeTypeDef = cast<model::TypedefDefinition>(TypeTypedef.get());
  TheTypeTypeDef->UnderlyingType() = *ModelTypedefType;
  setCustomName(*TheTypeTypeDef, D->getName());

  if (AnalysisOption == ImportFromCOption::EditType) {
    // Remove old and add new type with the same ID.
    llvm::erase_if(Model->TypeDefinitions(),
                   [&](model::UpcastableTypeDefinition &P) {
                     return P.get()->ID() == (*Type)->ID();
                   });

    Model->TypeDefinitions().insert(std::move(TypeTypedef));
  } else {
    Model->recordNewType(std::move(TypeTypedef));
  }

  return true;
}

bool DeclVisitor::VisitFunctionPrototype(const FunctionProtoType *FP,
                                         std::optional<llvm::StringRef> ABI) {
  revng_assert(AnalysisOption != ImportFromCOption::EditFunctionPrototype);

  if (not ABI) {
    revng_log(Log, "No annotate attribute found with `abi:` information");
    return false;
  }

  bool IsRawFunctionType = ABI->starts_with(RawABIPrefix);
  auto NewType = IsRawFunctionType ?
                   makeTypeDefinition<RawFunctionDefinition>() :
                   makeTypeDefinition<CABIFunctionDefinition>();

  if (AnalysisOption == ImportFromCOption::EditType)
    NewType->ID() = (*Type)->ID();

  if (not IsRawFunctionType) {
    auto FunctionType = cast<CABIFunctionDefinition>(NewType.get());
    auto TheModelABI = model::ABI::fromName(*ABI);
    if (TheModelABI == model::ABI::Invalid) {
      revng_log(Log, "An invalid ABI found as an input");
      return false;
    }

    FunctionType->ABI() = TheModelABI;

    auto TheRetClangType = FP->getReturnType();
    auto TheRetModelType = getModelTypeForClangType(TheRetClangType);
    if (not TheRetModelType) {
      revng_log(Log, "Unsupported type for function return value");
      return false;
    }

    FunctionType->ReturnType() = *TheRetModelType;

    // Handle params.
    uint32_t Index = 0;
    for (auto QT : FP->getParamTypes()) {
      auto ParamType = getModelTypeForClangType(QT);
      if (not ParamType) {
        revng_log(Log, "Unsupported type for function parameter");
        return false;
      }

      model::Argument &NewArgument = FunctionType->Arguments()[Index];
      NewArgument.Type() = *ParamType;
      ++Index;
    }
  } else {
    auto Architecture = getRawABIArchitecture(*ABI);
    if (Architecture == model::Architecture::Invalid) {
      revng_log(Log, "Invalid raw abi architecture");
      return false;
    }

    // TODO: Since we do not have info about parameters annotation, we use
    // default raw function.
    const auto *TheDefaultPrototype = Model->DefaultPrototype().get();
    const auto *DefaultRFT = cast<RawFunctionDefinition>(TheDefaultPrototype);

    auto FunctionType = cast<RawFunctionDefinition>(NewType.get());
    FunctionType->Architecture() = Architecture;
    FunctionType->Arguments() = DefaultRFT->Arguments();
    FunctionType->ReturnValues() = DefaultRFT->ReturnValues();
    FunctionType->PreservedRegisters() = DefaultRFT->PreservedRegisters();
    FunctionType->FinalStackOffset() = DefaultRFT->FinalStackOffset();
  }

  if (AnalysisOption == ImportFromCOption::EditType) {
    // Remove old and add new type with the same ID.
    llvm::erase_if(Model->TypeDefinitions(),
                   [&](model::UpcastableTypeDefinition &P) {
                     return P.get()->ID() == (*Type)->ID();
                   });

    Model->TypeDefinitions().insert(std::move(NewType));
  } else {
    Model->recordNewType(std::move(NewType));
  }

  return true;
}

bool DeclVisitor::handleStructType(const clang::RecordDecl *RD) {
  const RecordDecl *Definition = RD->getDefinition();

  auto NewType = makeTypeDefinition<model::StructDefinition>();
  if (AnalysisOption == ImportFromCOption::EditType)
    NewType->ID() = (*Type)->ID();

  setCustomName(*NewType, RD->getName());
  auto Struct = cast<model::StructDefinition>(NewType.get());
  uint64_t CurrentOffset = 0;

  //
  // Iterate over the struct fields
  //
  llvm::SmallVector<std::pair<model::Register::Values, ModelType>, 4>
    ReturnValues;
  for (const FieldDecl *Field : Definition->fields()) {
    if (Field->isInvalidDecl()) {
      revng_log(Log, "Invalid declaration for a struct field");
      return false;
    }

    std::optional<model::Register::Values> LocationID;
    if (AnalysisOption == ImportFromCOption::EditFunctionPrototype) {
      if (not Field->hasAttr<AnnotateAttr>()) {
        revng_log(Log,
                  "Struct field representing return value should have annotate "
                  "attribute describing location");
        return false;
      }

      auto Annotate = std::find_if(begin(Field->getAttrs()),
                                   end(Field->getAttrs()),
                                   [&](Attr *Attribute) {
                                     return isa<AnnotateAttr>(Attribute);
                                   });

      auto Loc = getLoc(cast<AnnotateAttr>(*Annotate)->getAnnotation());
      if (not Loc or Loc->empty()) {
        revng_log(Log,
                  "Return value should have attribute annotate with reg or "
                  "stack");
        return false;
      }

      // TODO: Handle stack location.
      if (*Loc == "stack") {
        revng_log(Log, "We don't support Return value on stack for now");
        return false;
      }

      LocationID = model::Register::fromCSVName(*Loc, Model->Architecture());
      if (*LocationID == model::Register::Invalid) {
        revng_log(Log, "Unsupported register location");
        return false;
      }
    }

    std::optional<uint64_t> Size = 0;
    const QualType &FieldType = Field->getType();
    auto TheFieldType = getModelTypeForClangType(FieldType);

    if (not TheFieldType) {
      revng_log(Log, "Unsupported type for a struct field");
      return false;
    }

    if (AnalysisOption == ImportFromCOption::EditFunctionPrototype) {
      revng_assert(LocationID);
      ReturnValues.push_back({ *LocationID,
                               { TheFieldType->UnqualifiedType(),
                                 TheFieldType->Qualifiers() } });
    }

    if (FieldType->isPointerType()) {
      auto PointerSize = getPointerSize(Model->Architecture());
      Size = PointerSize;
    } else if (FieldType->isArrayType()) {
      uint64_t NumberOfElements = 0;
      if (const auto *CAT = dyn_cast<ConstantArrayType>(FieldType)) {
        NumberOfElements = CAT->getSize().getZExtValue();
      } else {
        revng_log(Log, "Unsupported array type");
        return false;
      }
      Size = *(TheFieldType->UnqualifiedType().get()->size())
             * NumberOfElements;
    } else {
      Size = *(TheFieldType->UnqualifiedType().get()->size());
    }

    // Do not create fields for padding fields
    if (not Field->getName().starts_with(StructPaddingPrefix)) {
      auto &FieldModelType = Struct->Fields()[CurrentOffset];
      setCustomName(FieldModelType, Field->getName());
      FieldModelType.Type() = *TheFieldType;
    }

    revng_assert(Size);
    CurrentOffset += *Size;
  }

  // TODO: Can this be calculated/fetched automatically?
  Struct->Size() = CurrentOffset;

  switch (AnalysisOption) {
  case ImportFromCOption::EditType:
    // Remove old and add new type with the same ID.
    llvm::erase_if(Model->TypeDefinitions(),
                   [&](model::UpcastableTypeDefinition &P) {
                     return P.get()->ID() == (*Type)->ID();
                   });
    Model->TypeDefinitions().insert(std::move(NewType));
    break;

  case ImportFromCOption::EditFunctionPrototype:
    MultiRegisterReturnValue = ReturnValues;
    break;

  case ImportFromCOption::AddType:
    Model->recordNewType(std::move(NewType));
    break;
  }

  return true;
}

bool DeclVisitor::handleUnionType(const clang::RecordDecl *RD) {
  revng_assert(AnalysisOption != ImportFromCOption::EditFunctionPrototype);

  const RecordDecl *Definition = RD->getDefinition();
  auto NewType = makeTypeDefinition<model::UnionDefinition>();
  if (AnalysisOption == ImportFromCOption::EditType)
    NewType->ID() = (*Type)->ID();

  setCustomName(*NewType, RD->getName().str());
  auto Union = cast<model::UnionDefinition>(NewType.get());

  uint64_t CurrentIndex = 0;
  for (const FieldDecl *Field : Definition->fields()) {
    if (Field->isInvalidDecl()) {
      revng_log(Log, "Invalid declaration for a union field");
      return false;
    }

    std::vector<Qualifier> Qualifiers;
    const QualType &FieldType = Field->getType();

    auto TheFieldType = getModelTypeForClangType(FieldType);

    if (not TheFieldType) {
      revng_log(Log, "Unsupported type for an union field");
      return false;
    }

    auto &FieldModelType = Union->Fields()[CurrentIndex];
    setCustomName(FieldModelType, Field->getName());
    FieldModelType.Type() = *TheFieldType;

    ++CurrentIndex;
  }

  if (AnalysisOption == ImportFromCOption::EditType) {
    // Remove old and add new type with the same ID.
    llvm::erase_if(Model->TypeDefinitions(),
                   [&](model::UpcastableTypeDefinition &P) {
                     return P.get()->ID() == (*Type)->ID();
                   });
    Model->TypeDefinitions().insert(std::move(NewType));
  } else {
    Model->recordNewType(std::move(NewType));
  }

  return true;
}

bool DeclVisitor::VisitRecordDecl(const clang::RecordDecl *RD) {
  if (not comesFromInternalFile(RD))
    return true;

  if (AnalysisOption != ImportFromCOption::EditFunctionPrototype
      and not RD->hasAttr<PackedAttr>()) {
    revng_log(Log, "Unions and Structs should have attribute packed");
    return false;
  }

  QualType TheType = Context.getTypeDeclType(RD);
  if (TheType->isStructureType()) {
    return handleStructType(RD);
  } else if (TheType->isUnionType()) {
    return handleUnionType(RD);
  } else {
    revng_log(Log, "Unhandled record type declaration");
    return false;
  }

  return true;
}

bool DeclVisitor::VisitEnumDecl(const EnumDecl *D) {
  if (not comesFromInternalFile(D))
    return true;

  revng_assert(AnalysisOption != ImportFromCOption::EditFunctionPrototype);

  if (not D->hasAttr<PackedAttr>()) {
    revng_log(Log, "Enums should have attribute packed");
    return false;
  }

  // Parse annotate attribute used for specifying underlying type.
  std::optional<std::string> UnderlyingType;
  if (D->hasAttr<AnnotateAttr>()) {
    auto Annotate = std::find_if(begin(D->getAttrs()),
                                 end(D->getAttrs()),
                                 [&](Attr *Attribute) {
                                   return isa<AnnotateAttr>(Attribute);
                                 });

    llvm::StringRef Annotation = cast<AnnotateAttr>(*Annotate)->getAnnotation();
    UnderlyingType = parseEnumUnderlyingType(Annotation);
    if (not UnderlyingType or UnderlyingType->empty()) {
      revng_log(Log,
                "Unable to parse `enum_underlying_type:` from the annotate "
                "attribute");
      return false;
    }
  }

  revng_assert(UnderlyingType.has_value());
  auto TheUnderlyingModelType = getEnumUnderlyingType(*UnderlyingType);
  if (not TheUnderlyingModelType) {
    revng_log(Log,
              "UnderlyingType of a EnumDefinition can only be Signed or "
              "Unsigned");
    return false;
  }

  auto NewType = makeTypeDefinition<model::EnumDefinition>();
  if (AnalysisOption == ImportFromCOption::EditType)
    NewType->ID() = (*Type)->ID();

  auto *Definition = D->getDefinition();
  auto TypeEnum = cast<model::EnumDefinition>(NewType.get());
  model::QualifiedType TheUnderlyingType(*TheUnderlyingModelType, {});
  TypeEnum->UnderlyingType() = TheUnderlyingType;
  setCustomName(*TypeEnum, Definition->getName());

  for (const auto *Enum : Definition->enumerators()) {
    auto &EnumEntry = TypeEnum->Entries()[Enum->getInitVal().getExtValue()];
    std::string NewName = Enum->getName().str();
    if (TypeEnum->entryName(EnumEntry) != NewName)
      EnumEntry.CustomName() = NewName;
  }

  if (AnalysisOption == ImportFromCOption::EditType) {
    // Remove old and add new type with the same ID.
    llvm::erase_if(Model->TypeDefinitions(),
                   [&](model::UpcastableTypeDefinition &P) {
                     return P.get()->ID() == (*Type)->ID();
                   });

    Model->TypeDefinitions().insert(std::move(NewType));
  } else {
    Model->recordNewType(std::move(NewType));
  }

  return true;
}

void DeclVisitor::run(clang::TranslationUnitDecl *TUD) {
  this->TraverseDecl(TUD);
}

bool DeclVisitor::TraverseDecl(clang::Decl *D) {
  // This can happen due to an error in the code.
  if (!D)
    return true;

  setupLineAndColumn(D);

  if (isa<EnumDecl>(D))
    VisitEnumDecl(cast<EnumDecl>(D));

  clang::RecursiveASTVisitor<DeclVisitor>::TraverseDecl(D);
  return true;
}

void HeaderToModel::HandleTranslationUnit(ASTContext &Context) {
  Model->Architecture() = Model->Architecture();
  Model->DefaultABI() = Model->DefaultABI();

  clang::TranslationUnitDecl *TUD = Context.getTranslationUnitDecl();
  DeclVisitor(Model, Context, Type, Function, Error, AnalysisOption).run(TUD);
}

std::unique_ptr<ASTConsumer> HeaderToModelEditTypeAction::newASTConsumer() {
  std::optional<model::Function> FunctionToBeEdited{ std::nullopt };
  return std::make_unique<HeaderToModel>(Model,
                                         Type,
                                         FunctionToBeEdited,
                                         Error,
                                         AnalysisOption);
}

std::unique_ptr<ASTConsumer> HeaderToModelEditFunctionAction::newASTConsumer() {
  std::optional<model::TypeDefinition *> TypeToBeEdited{ std::nullopt };
  return std::make_unique<HeaderToModel>(Model,
                                         TypeToBeEdited,
                                         Function,
                                         Error,
                                         AnalysisOption);
}

std::unique_ptr<ASTConsumer> HeaderToModelAddTypeAction::newASTConsumer() {
  std::optional<model::TypeDefinition *> TypeToBeEdited{ std::nullopt };
  std::optional<model::Function> FunctionToBeEdited{ std::nullopt };
  return std::make_unique<HeaderToModel>(Model,
                                         TypeToBeEdited,
                                         FunctionToBeEdited,
                                         Error,
                                         AnalysisOption);
}

std::unique_ptr<ASTConsumer>
HeaderToModelAction::CreateASTConsumer(CompilerInstance &, llvm::StringRef) {
  return newASTConsumer();
}

bool HeaderToModelAction::BeginInvocation(clang::CompilerInstance &CI) {
  DiagConsumer = new HeaderToModelDiagnosticConsumer(CI.getDiagnostics());
  CI.getDiagnostics().setClient(DiagConsumer, /*ShouldOwnClient=*/true);
  return true;
}

void HeaderToModelAction::EndSourceFile() {
  if (DiagConsumer->getError()) {
    Error = DiagConsumer->getError();
  }
}

void HeaderToModelDiagnosticConsumer::EndSourceFile() {
  Client->EndSourceFile();
}

using Level = DiagnosticsEngine::Level;
void HeaderToModelDiagnosticConsumer::HandleDiagnostic(Level DiagLevel,
                                                       const Diagnostic &Info) {
  SmallString<100> OutStr;
  Info.FormatDiagnostic(OutStr);

  llvm::raw_svector_ostream DiagMessageStream(OutStr);

  std::string Text;
  std::string ErrorLocation;
  llvm::raw_string_ostream OS(Text);
  auto *DiagOpts = &Info.getDiags()->getDiagnosticOptions();

  uint64_t StartOfLocationInfo = OS.tell();

  TextDiagnostic::printDiagnosticLevel(OS, DiagLevel, DiagOpts->ShowColors);
  const bool IsSupplemental = DiagLevel == DiagnosticsEngine::Note;
  TextDiagnostic::printDiagnosticMessage(OS,
                                         IsSupplemental,
                                         DiagMessageStream.str(),
                                         OS.tell() - StartOfLocationInfo,
                                         DiagOpts->MessageLength,
                                         DiagOpts->ShowColors);

  unsigned Line = 0;
  unsigned Column = 0;
  std::string FileName;
  if (Info.getLocation().isValid()) {
    FullSourceLoc Location(Info.getLocation(), Info.getSourceManager());
    Line = Location.getLineNumber();
    Column = Location.getColumnNumber();
    FileName = Location.getPresumedLoc().getFilename();
  }

  ErrorLocation = FileName + ":" + std::to_string(Line) + ":"
                  + std::to_string(Column) + ": ";
  Text = ErrorLocation + Text;
  // Report all the messages coming from clang.
  if (Error)
    Text = Error->ErrorMessage + Text;

  Error = { Text, Line, Column };

  OS.flush();
}
} // end namespace tooling
} // end namespace clang
