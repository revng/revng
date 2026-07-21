//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "clang/AST/Decl.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/TextDiagnostic.h"

#include "revng/ABI/ModelHelpers.h"
#include "revng/ClangToModel/QualTypeToModel.h"
#include "revng/Model/FunctionAttribute.h"
#include "revng/Model/NameBuilder.h"
#include "revng/Model/Processing.h"
#include "revng/Model/TypeDefinitionByName.h"
#include "revng/PTML/CAttributes.h"
#include "revng/PTML/CBuilder.h"
#include "revng/Ranks/Ranks.h"
#include "revng/Support/Debug.h"

#include "EditCType.h"

using namespace model;
using namespace revng;

static constexpr llvm::StringRef InputCFile = "revng-input.c";
static constexpr llvm::StringRef RawABIPrefix = "raw_";

namespace {

// TODO: listing all the field here is pretty nasty, but it will have to do
//       for now.

void preserveMetadata(const model::Function &Old, model::Function &New) {
  New.Comments() = Old.Comments();
  New.LocalVariables() = Old.LocalVariables();
  New.GotoLabels() = Old.GotoLabels();
  New.CallSitePrototypes() = Old.CallSitePrototypes();
  New.ExportedNames() = Old.ExportedNames();
  New.StackFrame() = Old.StackFrame();

  // TODO: don't forget to extend when new fields are added.
}

void preserveMetadata(const model::TypeDefinition &Old,
                      model::TypeDefinition &New) {
  New.Comment() = Old.Comment();

  // TODO: don't forget to extend when new fields are added.
}

void preserveMetadata(const model::EnumEntry &Old, model::EnumEntry &New) {
  New.Comment() = Old.Comment();

  // TODO: don't forget to extend when new fields are added.
}

template<typename EntityType>
void setNameIfNotAutomatic(model::CNameBuilder &NameBuilder,
                           EntityType &Entity,
                           llvm::StringRef Name) {
  if (not NameBuilder.isAutomaticName(Entity, Name))
    Entity.Name() = Name;
  else
    Entity.Name() = "";
}

template<typename ParentType, typename EntityType>
void setNameIfNotAutomatic(model::CNameBuilder &NameBuilder,
                           const ParentType &Parent,
                           EntityType &Entity,
                           llvm::StringRef Name) {
  if (not NameBuilder.isAutomaticName(Parent, Entity, Name))
    Entity.Name() = Name;
  else
    Entity.Name() = "";
}

} // namespace

namespace clang {
namespace tooling {

class EditCType : public ASTConsumer {
public:
  EditCType(TupleTree<model::Binary> &Model,
            std::optional<model::TypeDefinition::Key> Type,
            MetaAddress FunctionEntry,
            ImportingErrorList &Errors,
            enum EditCTypeOption AnalysisOption) :
    Model(Model),
    Type(Type),
    FunctionEntry(FunctionEntry),
    Errors(Errors),
    AnalysisOption(AnalysisOption) {
    // Either one of these two should be null, since the editing features are
    // exclusive.
    revng_assert(not Type or not FunctionEntry.isValid());
  }

  virtual void HandleTranslationUnit(ASTContext &Context) override;

private:
  TupleTree<model::Binary> &Model;
  std::optional<model::TypeDefinition::Key> Type;
  MetaAddress FunctionEntry;
  ImportingErrorList &Errors;
  enum EditCTypeOption AnalysisOption;
};

class DeclVisitor : public clang::RecursiveASTVisitor<DeclVisitor> {
private:
  TupleTree<model::Binary> &Model;
  ASTContext &Context;
  std::optional<model::TypeDefinition::Key> Type;
  MetaAddress FunctionEntry;
  ImportingErrorList &Errors;
  enum EditCTypeOption AnalysisOption;

  // These are used for reporting source location of an error, if any.
  unsigned CurrentLineNumber = 0;
  unsigned CurrentColumnNumber = 0;

  // Used to remember return values locations when parsing struct representing
  // the multi-reg return value. Represents register ID and model::Type.
  using RawLocation = std::pair<model::Register::Values, model::UpcastableType>;
  std::optional<llvm::SmallVector<RawLocation, 4>> MultiRegisterReturnValue;

  model::CNameBuilder NameBuilder;

public:
  DeclVisitor(TupleTree<model::Binary> &Model,
              ASTContext &Context,
              std::optional<model::TypeDefinition::Key> Type,
              MetaAddress FunctionEntry,
              ImportingErrorList &Errors,
              enum EditCTypeOption AnalysisOption);

  void run(clang::TranslationUnitDecl *TUD);
  bool TraverseDecl(clang::Decl *D);

  bool VisitFunctionDecl(const clang::FunctionDecl *FD);
  bool VisitRecordDecl(const clang::RecordDecl *RD);
  bool VisitEnumDecl(const EnumDecl *D);
  bool VisitTypedefDecl(const TypedefDecl *D);
  bool VisitFunctionPrototype(const FunctionProtoType *FP,
                              llvm::StringRef TheABI);

private:
  // This checks that the declaration is the one user provided as input.
  bool comesFromInternalFile(const clang::Decl *D);

  // Set up line and column for the declaratrion.
  void setupLineAndColumn(const clang::Decl *D);

  // Handle clang's Struct type.
  bool handleStructType(const clang::RecordDecl *RD);
  // Handle clang's Union type.
  bool handleUnionType(const clang::RecordDecl *RD);

  template<ConstexprString Macro, typename Type>
  std::optional<llvm::StringRef>
  parseStringAnnotation(const Type &Declaration, ImportingErrorList &Errors);

  template<ConstexprString Macro, typename Type>
  std::optional<uint64_t>
  parseIntegerAnnotation(const Type &Declaration, ImportingErrorList &Errors);
};

DeclVisitor::DeclVisitor(TupleTree<model::Binary> &Model,
                         ASTContext &Context,
                         std::optional<model::TypeDefinition::Key> Type,
                         MetaAddress FunctionEntry,
                         ImportingErrorList &Errors,
                         enum EditCTypeOption AnalysisOption) :
  Model(Model),
  Context(Context),
  Type(Type),
  FunctionEntry(FunctionEntry),
  Errors(Errors),
  AnalysisOption(AnalysisOption),
  NameBuilder(*Model) {
}

template<ConstexprString Macro, typename Type>
std::optional<llvm::StringRef>
DeclVisitor::parseStringAnnotation(const Type &Declaration,
                                   ImportingErrorList &Errors) {
  static constexpr auto Prefix = ptml::AttributeRegistry::getPrefix<Macro>();

  std::optional<llvm::StringRef> Result;
  if (Declaration.template hasAttr<clang::AnnotateAttr>()) {
    for (auto &Attribute : Declaration.getAttrs()) {
      if (auto *Cast = llvm::dyn_cast<clang::AnnotateAttr>(Attribute)) {
        llvm::StringRef Annotation = Cast->getAnnotation();
        if (not Annotation.startswith(Prefix))
          continue;

        llvm::StringRef Value = Annotation.substr(Prefix.size());
        if (Result.has_value() && Result.value() != Value) {
          std::string ErrorPrefix = "edit-c-type:";

          SourceManager &SM = Context.getSourceManager();
          PresumedLoc Loc = SM.getPresumedLoc(Attribute->getRange().getBegin());
          if (Loc.isValid())
            ErrorPrefix += std::to_string(Loc.getLine()) + ":"
                           + std::to_string(Loc.getColumn()) + ":";

          Errors.emplace_back(ErrorPrefix + " Multiple conflicting values (`"
                              + Result.value().str() + "` and `" + Value.str()
                              + "`) were found for the `" + std::string(Macro)
                              + "` annotation.\n");
          return std::nullopt;
        }

        Result = Value;
      }
    }
  }

  return Result;
}

template<ConstexprString Macro, typename Type>
std::optional<uint64_t>
DeclVisitor::parseIntegerAnnotation(const Type &Declaration,
                                    ImportingErrorList &Errors) {
  std::optional Result = parseStringAnnotation<Macro>(Declaration, Errors);
  if (not Result.has_value())
    return std::nullopt;

  uint64_t IntegerResult;
  if (Result->getAsInteger(0, IntegerResult)) {
    Errors.emplace_back("edit-c-type: Ignoring a non-integer value (`"
                        + Result->str() + "`) of an integer annotation: `"
                        + std::string(Macro) + "`.\n");
    return std::nullopt;
  }

  return IntegerResult;
}

static model::Architecture::Values getRawABIArchitecture(llvm::StringRef ABI) {
  revng_assert(ABI.starts_with(RawABIPrefix));
  return model::Architecture::fromName(ABI.substr(RawABIPrefix.size()));
}

bool DeclVisitor::comesFromInternalFile(const clang::Decl *D) {
  SourceManager &SM = Context.getSourceManager();
  PresumedLoc Loc = SM.getPresumedLoc(D->getLocation());
  if (!Loc.isValid())
    return false;

  StringRef TheFileName(Loc.getFilename());
  // Process the new type only.
  if (TheFileName.contains(InputCFile))
    return true;

  return false;
}

void DeclVisitor::setupLineAndColumn(const clang::Decl *D) {
  SourceManager &SM = Context.getSourceManager();
  PresumedLoc Loc = SM.getPresumedLoc(D->getLocation());
  if (!Loc.isValid())
    return;

  CurrentLineNumber = Loc.getLine();
  CurrentColumnNumber = Loc.getColumn();
}

bool DeclVisitor::VisitFunctionDecl(const clang::FunctionDecl *FD) {
  if (not comesFromInternalFile(FD))
    return true;

  revng_assert(FD);
  revng_assert(AnalysisOption == EditCTypeOption::EditFunctionPrototype);

  std::optional ABI = parseStringAnnotation<"_ABI">(*FD, Errors);
  if (not ABI.has_value() or ABI->empty()) {
    Errors.emplace_back("edit-c-type failed: Functions without an "
                        "`_ABI($name)` or `_ABI(raw_$arch)` annotation are not "
                        "allowed.\n");
    return false;
  }

  bool IsRawFunctionType = ABI->starts_with(RawABIPrefix);
  auto NewType = IsRawFunctionType ?
                   makeTypeDefinition<RawFunctionDefinition>() :
                   makeTypeDefinition<CABIFunctionDefinition>();

  if (not IsRawFunctionType) {
    auto TheModelABI = model::ABI::fromName(*ABI);
    if (TheModelABI == model::ABI::Invalid) {
      Errors.emplace_back("edit-c-type failed: Unknown ABI: `" + ABI->str()
                          + "`.\n");
      return false;
    }

    auto &FunctionType = llvm::cast<CABIFunctionDefinition>(*NewType);
    FunctionType.ABI() = TheModelABI;

    // A void return is left as an empty ReturnType.
    auto TheRetClangType = FD->getReturnType();
    if (not TheRetClangType->isVoidType()) {
      model::UpcastableType RetType = revng::qualTypeToModel(TheRetClangType,
                                                             *Model,
                                                             Context,
                                                             Errors,
                                                             "edit-c-type:");
      if (not RetType) {
        Errors.emplace_back("edit-c-type failed: Unable to parse the type of "
                            "the return value.\n");
        return false;
      }

      FunctionType.ReturnType() = std::move(RetType);
    }

    // Handle params.
    uint32_t Index = 0;
    for (unsigned I = 0, N = FD->getNumParams(); I != N; ++I) {
      auto QT = FD->getParamDecl(I)->getType();
      model::UpcastableType ParamType = revng::qualTypeToModel(QT,
                                                               *Model,
                                                               Context,
                                                               Errors,
                                                               "edit-c-type:");
      if (not ParamType) {
        Errors.emplace_back("edit-c-type failed: Unable to parse the type of "
                            "the argument #"
                            + std::to_string(I) + ".\n");
        return false;
      }

      model::Argument &NewArgument = FunctionType.Arguments()[Index];

      setNameIfNotAutomatic(NameBuilder,
                            FunctionType,
                            NewArgument,
                            FD->getParamDecl(I)->getName());

      // TODO: This discard whatever comments might have been attached to
      //       the original argument.

      NewArgument.Type() = std::move(ParamType);
      ++Index;
    }
  } else {
    auto TheRetClangType = FD->getReturnType();
    auto &TheRawFunctionType = llvm::cast<RawFunctionDefinition>(*NewType);

    auto Architecture = getRawABIArchitecture(*ABI);
    if (Architecture == model::Architecture::Invalid) {
      Errors.emplace_back("edit-c-type failed: Unknown architecture: `"
                          + ABI->substr(RawABIPrefix.size()).str() + "`.\n");
      return false;
    }
    TheRawFunctionType.Architecture() = Architecture;

    auto ReturnValuesInserter = TheRawFunctionType.ReturnValues()
                                  .batch_insert();

    // This represents multiple register location for return values.
    if (TheRetClangType->isStructureType()) {
      if (not MultiRegisterReturnValue) {
        Errors.emplace_back("edit-c-type failed: Unable to parse the type of "
                            "the return value.\n");
        return false;
      }

      for (auto &[Location, Type] : *MultiRegisterReturnValue) {
        model::NamedTypedRegister &NTR = ReturnValuesInserter.emplace(Location);
        NTR.Type() = Type;
      }
    } else {
      std::optional Register = parseStringAnnotation<"_REG">(*FD, Errors);
      if (not Register.has_value()) {
        std::optional Stack = parseStringAnnotation<"_STACK">(*FD, Errors);
        if (Stack.has_value()) {
          Errors.emplace_back("edit-c-type failed: Only register values are "
                              "allowed as a part of a raw function's return "
                              "value. As such, they must not use _STACK "
                              "annotation.\n");
          return false;

        } else {
          Errors.emplace_back("edit-c-type failed: Return values of a raw "
                              "function must have a _REG($name) annotation.\n");
          return false;
        }
      }

      model::UpcastableType RetType = revng::qualTypeToModel(TheRetClangType,
                                                             *Model,
                                                             Context,
                                                             Errors,
                                                             "edit-c-type:");
      if (not RetType) {
        Errors.emplace_back("edit-c-type failed: Unable to parse the type of "
                            "the return value.\n");
        return false;
      }

      auto Location = model::Register::fromRegisterName(*Register,
                                                        Model->Architecture());
      if (Location == model::Register::Invalid) {
        Errors.emplace_back("edit-c-type: While parsing the return value:\n");
        Errors.emplace_back("edit-c-type failed: Unknown register: `"
                            + Register->str() + "`.\n");
        return false;
      }

      auto &ReturnValueReg = ReturnValuesInserter.emplace(Location);
      ReturnValueReg.Type() = std::move(RetType);
    }

    auto ArgumentsInserter = TheRawFunctionType.Arguments().batch_insert();
    for (unsigned I = 0, N = FD->getNumParams(); I != N; ++I) {
      auto ParamDecl = FD->getParamDecl(I);
      auto QT = ParamDecl->getType();
      model::UpcastableType ParamType = revng::qualTypeToModel(QT,
                                                               *Model,
                                                               Context,
                                                               Errors,
                                                               "edit-c-type:");
      if (not ParamType) {
        Errors.emplace_back("edit-c-type failed: Unable to parse the type of "
                            "the argument #"
                            + std::to_string(I) + ".\n");
        return false;
      }

      std::optional Register = parseStringAnnotation<"_REG">(*ParamDecl,
                                                             Errors);
      std::optional Stack = parseStringAnnotation<"_STACK">(*ParamDecl, Errors);
      if (not Register.has_value()) {
        if (not Stack.has_value()) {
          Errors.emplace_back("edit-c-type failed: Argument #"
                              + std::to_string(I)
                              + " is missing it's location annotation.\n");
          Errors.emplace_back("                      Please add either "
                              "`_REG($name)` or `_STACK`.\n");
          return false;
        } else {
          if (I != N - 1) {
            Errors.emplace_back("edit-c-type failed: Only the very last RFT "
                                "argument is allowed to represent stack, which "
                                "also means there can only be one.\n");
            Errors.emplace_back("                      Please either remove "
                                "`_STACK` annotation from the argument #"
                                + std::to_string(I)
                                + " or move it into the stack argument "
                                  "struct.\n");
            return false;
          }

          if (not ParamType->isStruct()) {
            Errors.emplace_back("edit-c-type failed: RFT stack argument must "
                                "be a "
                                "struct. You can use fields of such a struct "
                                "to represent separate arguments.\n");
            return false;
          }

          revng_assert(TheRawFunctionType.StackArgumentsType().isEmpty());

          TheRawFunctionType.StackArgumentsType() = std::move(ParamType);
          if (ParamDecl->getName() != "stack") {
            Errors.emplace_back("edit-c-type: stack argument name (`"
                                + ParamDecl->getName().str()
                                + "`) was ignored, as model stores the struct "
                                  "as is.\n");
          }
        }
      } else {
        if (Stack.has_value()) {
          Errors.emplace_back("edit-c-type failed: A single argument cannot "
                              "use both a register and stack: the model does "
                              "not support that. Please use two separate "
                              "arguments.\n");
          return false;
        }

        using namespace model;
        auto Location = Register::fromRegisterName(*Register,
                                                   Model->Architecture());
        if (Location == Register::Invalid) {
          Errors.emplace_back("edit-c-type: While parsing argument #"
                              + std::to_string(I) + ":\n");
          Errors.emplace_back("edit-c-type failed: Unknown register: `"
                              + Register->str() + "`.\n");
          return false;
        }

        NamedTypedRegister &ParamReg = ArgumentsInserter.emplace(Location);
        ParamReg.Type() = std::move(ParamType);
        if (not ParamDecl->getName().empty())
          setNameIfNotAutomatic(NameBuilder,
                                TheRawFunctionType,
                                ParamReg,
                                ParamDecl->getName());

        // TODO: This discard whatever comments might have been attached to
        //       the original register.
      }
    }
  }

  // Update the name in case it changed.
  auto &ModelFunction = Model->Functions()[FunctionEntry];

  if (FD->isNoReturn())
    ModelFunction.Attributes().emplace(model::FunctionAttribute::NoReturn);

  if (FD->hasAttr<clang::AlwaysInlineAttr>())
    ModelFunction.Attributes().emplace(model::FunctionAttribute::AlwaysInline);

  setNameIfNotAutomatic(NameBuilder, ModelFunction, FD->getName());

  if (auto *OriginalFunction = Model->Functions().tryGet(FunctionEntry))
    preserveMetadata(*OriginalFunction, ModelFunction);

  auto &&[_, Prototype] = Model->recordNewType(std::move(NewType));
  ModelFunction.Prototype() = Prototype;

  return true;
}

bool DeclVisitor::VisitTypedefDecl(const TypedefDecl *D) {
  if (not comesFromInternalFile(D))
    return true;

  // A function's prototype is edited from a function declaration (see
  // VisitFunctionDecl); a typedef cannot carry the function attributes.
  if (AnalysisOption == EditCTypeOption::EditFunctionPrototype) {
    Errors.emplace_back("edit-c-type failed: editing a function prototype "
                        "requires a function declaration, not a typedef.\n");
    return false;
  }

  QualType TheType = D->getUnderlyingType();
  if (auto Fn = llvm::dyn_cast<FunctionProtoType>(TheType)) {
    // Parse the ABI from annotate attribute attached to the typedef
    // declaration. Please do note that annotations on the parameters are not
    // attached, so we will use default RawFunctionDefinition from the Model if
    // the abi is raw.
    // TODO: Should we change the annotate attached to function types to have
    // info about parameters in the toplevel annotate attribute attached to
    // the typedef itself?
    std::optional ABI = parseStringAnnotation<"_ABI">(*D, Errors);
    if (not ABI.has_value()) {
      Errors.emplace_back("edit-c-type failed: a function typedef must "
                          "either have `_ABI($name)` or `_ABI(raw_$arch)` "
                          "annotation attached.\n");
      return false;
    }
    if (ABI->empty()) {
      Errors.emplace_back("edit-c-type failed: _ABI annotation must not be "
                          "empty.\n");
      Errors.emplace_back("                      Please specify an abi name or "
                          "an architecture name.\n");
      return false;
    }

    return VisitFunctionPrototype(Fn, *ABI);
  }

  // Regular, non-function, typedef.
  model::UpcastableType ModelTypedefType = revng::qualTypeToModel(TheType,
                                                                  *Model,
                                                                  Context,
                                                                  Errors,
                                                                  "edit-c-"
                                                                  "type:");
  if (not ModelTypedefType) {
    Errors.emplace_back("edit-c-type failed: Unable to parse the underlying "
                        "type of the typedef.\n");
    return false;
  }
  auto &&[ID, Kind] = *Type;
  auto NewTypedef = model::makeTypeDefinition<model::TypedefDefinition>();
  if (AnalysisOption == EditCTypeOption::EditType)
    NewTypedef->ID() = ID;

  auto TheTypeTypeDef = cast<model::TypedefDefinition>(NewTypedef.get());
  TheTypeTypeDef->UnderlyingType() = std::move(ModelTypedefType);

  setNameIfNotAutomatic(NameBuilder, *TheTypeTypeDef, D->getName());

  if (AnalysisOption == EditCTypeOption::EditType)
    if (auto *OldType = Model->TypeDefinitions().tryGet(*Type))
      preserveMetadata(**OldType, *NewTypedef);

  if (AnalysisOption == EditCTypeOption::EditType) {
    revng_assert(*Type == NewTypedef->key());
    Model->TypeDefinitions().erase(*Type);
    Model->TypeDefinitions().insert(std::move(NewTypedef));
  } else {
    Model->recordNewType(std::move(NewTypedef));
  }

  return true;
}

bool DeclVisitor::VisitFunctionPrototype(const FunctionProtoType *FP,
                                         llvm::StringRef ABI) {
  revng_assert(AnalysisOption != EditCTypeOption::EditFunctionPrototype);
  revng_assert(ABI != "");

  bool IsRawFunctionType = ABI.starts_with(RawABIPrefix);
  auto NewType = IsRawFunctionType ?
                   makeTypeDefinition<RawFunctionDefinition>() :
                   makeTypeDefinition<CABIFunctionDefinition>();

  if (AnalysisOption == EditCTypeOption::EditType) {
    auto &&[ID, Kind] = *Type;
    NewType->ID() = ID;
  }

  if (not IsRawFunctionType) {
    auto &FunctionType = llvm::cast<CABIFunctionDefinition>(*NewType);
    auto TheModelABI = model::ABI::fromName(ABI);
    if (TheModelABI == model::ABI::Invalid) {
      Errors.emplace_back("edit-c-type failed: Unknown ABI: `" + ABI.str()
                          + "`.\n");
      return false;
    }

    FunctionType.ABI() = TheModelABI;

    // A void return is left as an empty ReturnType.
    auto TheRetClangType = FP->getReturnType();
    if (not TheRetClangType->isVoidType()) {
      model::UpcastableType RetType = revng::qualTypeToModel(TheRetClangType,
                                                             *Model,
                                                             Context,
                                                             Errors,
                                                             "edit-c-type:");
      if (not RetType) {
        Errors.emplace_back("edit-c-type failed: Unable to parse the return "
                            "value type.\n");
        return false;
      }

      FunctionType.ReturnType() = std::move(RetType);
    }

    // Handle params.
    uint32_t Index = 0;
    for (auto QT : FP->getParamTypes()) {
      model::UpcastableType ParamType = revng::qualTypeToModel(QT,
                                                               *Model,
                                                               Context,
                                                               Errors,
                                                               "edit-c-type:");
      if (not ParamType) {
        Errors.emplace_back("edit-c-type failed: Unable to parse the type of "
                            "argument #`"
                            + std::to_string(Index) + "`.\n");
        return false;
      }

      model::Argument &NewArgument = FunctionType.Arguments()[Index];
      NewArgument.Type() = std::move(ParamType);
      ++Index;
    }
  } else {
    auto Architecture = getRawABIArchitecture(ABI);
    if (Architecture == model::Architecture::Invalid) {
      Errors.emplace_back("edit-c-type failed: Unknown architecture: `"
                          + ABI.substr(RawABIPrefix.size()).str() + "`.\n");
      return false;
    }

    // TODO: Since we do not have info about parameters annotation, we use
    // default raw function.
    auto Default = cast<RawFunctionDefinition>(*Model->defaultPrototype());

    auto &FunctionType = llvm::cast<RawFunctionDefinition>(*NewType);
    FunctionType.Architecture() = Architecture;
    FunctionType.Arguments() = Default.Arguments();
    FunctionType.ReturnValues() = Default.ReturnValues();
    FunctionType.PreservedRegisters() = Default.PreservedRegisters();
    FunctionType.FinalStackOffset() = Default.FinalStackOffset();
  }

  if (AnalysisOption == EditCTypeOption::EditType) {
    revng_assert(*Type == NewType->key());
    Model->TypeDefinitions().erase(*Type);
    Model->TypeDefinitions().insert(std::move(NewType));
  } else {
    Model->recordNewType(std::move(NewType));
  }

  return true;
}

bool DeclVisitor::handleStructType(const clang::RecordDecl *RD) {
  const RecordDecl *Definition = RD->getDefinition();
  if (Definition == nullptr) {
    Errors.emplace_back("edit-c-type failed: Unable to parse the struct.\n");
    return false;
  }

  auto &&[ID, Kind] = *Type;
  auto NewType = makeTypeDefinition<model::StructDefinition>();
  if (AnalysisOption == EditCTypeOption::EditType)
    NewType->ID() = ID;

  setNameIfNotAutomatic(NameBuilder, *NewType, RD->getName());

  auto *Struct = cast<model::StructDefinition>(NewType.get());
  uint64_t CurrentOffset = 0;

  const model::StructDefinition *OldStruct = nullptr;
  if (AnalysisOption == EditCTypeOption::EditType)
    if (auto *OldType = Model->TypeDefinitions().tryGet(*Type))
      OldStruct = dyn_cast<model::StructDefinition>(&**OldType);

  if (OldStruct != nullptr)
    preserveMetadata(*OldStruct, *Struct);

  //
  // Iterate over the struct fields
  //
  llvm::SmallVector<RawLocation, 4> ReturnValues;
  for (const FieldDecl *Field : Definition->fields()) {
    if (Field->isInvalidDecl()) {
      Errors.emplace_back("edit-c-type failed: The declaration of the struct "
                          "field #`"
                          + std::to_string(Struct->Fields().size())
                          + "` is not valid.\n");
      return false;
    }

    model::Register::Values Location;
    if (AnalysisOption == EditCTypeOption::EditFunctionPrototype) {
      std::optional Stack = parseStringAnnotation<"_STACK">(*Field, Errors);
      if (Stack.has_value()) {
        Errors.emplace_back("edit-c-type failed: Only register values are "
                            "allowed as a part of a raw function's return "
                            "value. As such, they must not use _STACK "
                            "annotation.\n");
        return false;
      }

      std::optional Register = parseStringAnnotation<"_REG">(*Field, Errors);
      if (not Register.has_value()) {
        Errors.emplace_back("edit-c-type failed: Return values of a raw "
                            "function must have a _REG($name) annotation.\n");
        return false;
      }

      Location = model::Register::fromRegisterName(*Register,
                                                   Model->Architecture());
      if (Location == model::Register::Invalid) {
        Errors.emplace_back("edit-c-type: While parsing return value #"
                            + std::to_string(Struct->Fields().size()) + ":\n");
        Errors.emplace_back("edit-c-type failed: Unknown register: `"
                            + Register->str() + "`.\n");
        return false;
      }
    }

    std::optional<uint64_t> Size = 0;
    const QualType &ClangFieldType = Field->getType();
    model::UpcastableType ModelField = revng::qualTypeToModel(ClangFieldType,
                                                              *Model,
                                                              Context,
                                                              Errors,
                                                              "edit-c-type:");

    if (ModelField.isEmpty()) {
      Errors.emplace_back("edit-c-type failed: Unable to parse the type of "
                          "struct field #"
                          + std::to_string(Struct->Fields().size()) + ".\n");
      return false;
    }

    if (AnalysisOption == EditCTypeOption::EditFunctionPrototype)
      ReturnValues.emplace_back(Location, ModelField);

    if (ClangFieldType->isPointerType()) {
      Size = model::Architecture::getPointerSize(Model->Architecture());

    } else if (ClangFieldType->isArrayType()) {
      uint64_t NumberOfElements = 0;
      if (const auto *CAT = dyn_cast<ConstantArrayType>(ClangFieldType)) {
        NumberOfElements = CAT->getSize().getZExtValue();
      } else {
        Errors.emplace_back("edit-c-type failed: Unsupported array type.\n");
        return false;
      }

      const model::Type &Element = *ModelField->toArray().ElementType();
      Size = *Element.size() * NumberOfElements;

    } else {
      Size = *ModelField->size();
    }

    const auto &Config = Model->Configuration().Naming();
    bool IsPadding = Field->getName().starts_with(Config.StructPaddingPrefix());
    auto ExplicitOffset = parseIntegerAnnotation<"_STARTS_AT">(*Field, Errors);
    if (ExplicitOffset.has_value()) {
      if (IsPadding) {
        Errors.emplace_back("edit-c-type: While parsing field #"
                            + std::to_string(Struct->Fields().size()) + ":\n");
        Errors.emplace_back("edit-c-type failed: Padding fields (`uint8_t "
                            "padding_at_$offset[$size]`) must not have "
                            "`_STARTS_AT` annotation attached.\n");
        return false;
      }

      if (not Struct->Fields().empty() and CurrentOffset > *ExplicitOffset) {
        Errors.emplace_back("edit-c-type: While parsing field #"
                            + std::to_string(Struct->Fields().size()) + ":\n");
        Errors.emplace_back("edit-c-type failed: `_STARTS_AT` must not be "
                            "used to make fields overlap.\n");
        return false;
      }

      CurrentOffset = *ExplicitOffset;
    }

    if (not IsPadding) {
      auto &FieldModelType = Struct->Fields()[CurrentOffset];

      setNameIfNotAutomatic(NameBuilder,
                            *Struct,
                            FieldModelType,
                            Field->getName());

      // TODO: This discard whatever comments might have been attached to
      //       the original field.

      FieldModelType.Type() = std::move(ModelField);
    } else {
      // Do not create fields for padding
    }

    revng_assert(Size);
    CurrentOffset += *Size;
  }

  if (std::optional ExplicitSize = parseIntegerAnnotation<"_SIZE">(*Definition,
                                                                   Errors)) {
    // Prefer explicit size if it's available.
    Struct->Size() = *ExplicitSize;

  } else {
    // If not, just use final offset,
    Struct->Size() = CurrentOffset;

    // Unless we're editing a type and have access to the previous size.
    if (OldStruct != nullptr)
      if (auto OldSize = *OldStruct->size(); Struct->Size() < OldSize)
        Struct->Size() = OldSize;
  }

  if (parseStringAnnotation<"_CAN_CONTAIN_CODE">(*RD, Errors))
    Struct->CanContainCode() = true;

  switch (AnalysisOption) {
  case EditCTypeOption::EditType:
    revng_assert(*Type == NewType->key());
    Model->TypeDefinitions().erase(*Type);
    Model->TypeDefinitions().insert(std::move(NewType));
    break;

  case EditCTypeOption::EditFunctionPrototype:
    MultiRegisterReturnValue = std::move(ReturnValues);
    break;

  case EditCTypeOption::AddType:
    Model->recordNewType(std::move(NewType));
    break;
  }

  return true;
}

bool DeclVisitor::handleUnionType(const clang::RecordDecl *RD) {
  revng_assert(AnalysisOption != EditCTypeOption::EditFunctionPrototype);

  const RecordDecl *Definition = RD->getDefinition();
  if (Definition == nullptr) {
    Errors.emplace_back("edit-c-type failed: Unable to parse the union.\n");
    return false;
  }

  auto &&[ID, Kind] = *Type;
  auto NewType = makeTypeDefinition<model::UnionDefinition>();
  if (AnalysisOption == EditCTypeOption::EditType)
    NewType->ID() = ID;

  setNameIfNotAutomatic(NameBuilder, *NewType, RD->getName());

  auto Union = cast<model::UnionDefinition>(NewType.get());

  if (AnalysisOption == EditCTypeOption::EditType)
    if (auto *OldType = Model->TypeDefinitions().tryGet(*Type))
      preserveMetadata(**OldType, *Union);

  uint64_t CurrentIndex = 0;
  for (const FieldDecl *Field : Definition->fields()) {
    if (Field->isInvalidDecl()) {
      Errors.emplace_back("edit-c-type failed: The declaration of the union "
                          "field #`"
                          + std::to_string(Union->Fields().size())
                          + "` is not valid.\n");
      return false;
    }

    const QualType &FieldType = Field->getType();
    model::UpcastableType TheFieldType = revng::qualTypeToModel(FieldType,
                                                                *Model,
                                                                Context,
                                                                Errors,
                                                                "edit-c-type:");

    if (not TheFieldType) {
      Errors.emplace_back("edit-c-type failed: Unable to parse the type of "
                          "union field #"
                          + std::to_string(Union->Fields().size()) + ".\n");
      return false;
    }

    auto &FieldModelType = Union->Fields()[CurrentIndex];

    setNameIfNotAutomatic(NameBuilder,
                          *Union,
                          FieldModelType,
                          Field->getName());

    // TODO: This discard whatever comments might have been attached to
    //       the original field.

    FieldModelType.Type() = std::move(TheFieldType);

    ++CurrentIndex;
  }

  if (AnalysisOption == EditCTypeOption::EditType) {
    revng_assert(*Type == NewType->key());
    Model->TypeDefinitions().erase(*Type);
    Model->TypeDefinitions().insert(std::move(NewType));
  } else {
    Model->recordNewType(std::move(NewType));
  }

  return true;
}

bool DeclVisitor::VisitRecordDecl(const clang::RecordDecl *RD) {
  if (not comesFromInternalFile(RD))
    return true;

  if (AnalysisOption != EditCTypeOption::EditFunctionPrototype
      and not RD->hasAttr<PackedAttr>()) {
    Errors.emplace_back("edit-c-type failed: Unions and Structs must be "
                        "`_PACKED`.\n");
    return false;
  }

  QualType TheType = Context.getTypeDeclType(RD);
  if (TheType->isStructureType()) {
    return handleStructType(RD);
  } else if (TheType->isUnionType()) {
    return handleUnionType(RD);
  } else {
    Errors.emplace_back("edit-c-type failed: As of now, only struct and "
                        "union record types are supported.\n");
    Errors.emplace_back("                      Please rewrite your type as one "
                        "of those two.\n");
    return false;
  }

  return true;
}

bool DeclVisitor::VisitEnumDecl(const EnumDecl *D) {
  if (not comesFromInternalFile(D))
    return true;

  revng_assert(AnalysisOption != EditCTypeOption::EditFunctionPrototype);

  if (not D->hasAttr<PackedAttr>()) {
    Errors.emplace_back("edit-c-type failed: Enums must be `_PACKED`.\n");
    return false;
  }

  // Parse annotate attribute used for specifying underlying type.
  auto UnderlyingTypeName = parseStringAnnotation<"_ENUM_UNDERLYING">(*D,
                                                                      Errors);
  if (not UnderlyingTypeName.has_value()) {
    Errors.emplace_back("edit-c-type failed: Enums without an "
                        "`_ENUM_UNDERLYING($type)` annotation are not "
                        "allowed.\n");
    return false;
  }
  if (UnderlyingTypeName->empty()) {
    Errors.emplace_back("edit-c-type failed: `_ENUM_UNDERLYING` must not be "
                        "empty: please specify a valid type name.\n");
    return false;
  }

  revng_assert(UnderlyingTypeName.has_value());
  auto UnderlyingType = model::PrimitiveType::fromCName(*UnderlyingTypeName);
  if (not UnderlyingType) {
    Errors.emplace_back("edit-c-type failed: unknown primitive type: `"
                        + UnderlyingTypeName->str() + "`.\n");
    return false;

  } else if (not UnderlyingType->isSignedPrimitive()
             and not UnderlyingType->isUnsignedPrimitive()) {
    Errors.emplace_back("edit-c-type failed: Underlying type of an enum can "
                        "only be signed or unsigned.\n");
    Errors.emplace_back("                      `" + UnderlyingTypeName->str()
                        + "` was found instead.\n");

    return false;
  }

  model::EnumDefinition *NewType = nullptr;
  if (AnalysisOption == EditCTypeOption::EditType) {
    revng_assert(Type != std::nullopt);
    model::TypeDefinition &Definition = *Model->TypeDefinitions().at(*Type);
    if (auto *Enum = llvm::dyn_cast<model::EnumDefinition>(&Definition)) {
      NewType = Enum;
    } else {
      // It seems like the kind of the type got changed. Since it affects
      // the key we need to erase the old type before adding the new one.
      Model->TypeDefinitions().erase(*Type);
      NewType = &Model->makeEnumDefinition().first;
    }
  }

  NewType->UnderlyingType() = std::move(UnderlyingType);

  auto *Definition = D->getDefinition();
  setNameIfNotAutomatic(NameBuilder, *NewType, Definition->getName());

  const model::EnumDefinition *OldEnum = nullptr;
  if (AnalysisOption == EditCTypeOption::EditType) {
    if (auto *OldType = Model->TypeDefinitions().tryGet(*Type)) {
      OldEnum = dyn_cast<model::EnumDefinition>(&**OldType);
      preserveMetadata(*OldEnum, *NewType);
    }
  }

  for (const auto *Enum : Definition->enumerators()) {
    auto Value = Enum->getInitVal().getExtValue();
    auto NewIterator = NewType->Entries().insert(Value).first;
    setNameIfNotAutomatic(NameBuilder,
                          *NewType,
                          *NewIterator,
                          Enum->getName().str());

    if (OldEnum != nullptr)
      if (auto *OldEntry = OldEnum->Entries().tryGet(Value))
        preserveMetadata(*OldEntry, *NewIterator);
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

  clang::RecursiveASTVisitor<DeclVisitor>::TraverseDecl(D);
  return true;
}

void EditCType::HandleTranslationUnit(ASTContext &Context) {
  clang::TranslationUnitDecl *TUD = Context.getTranslationUnitDecl();
  DeclVisitor(Model, Context, Type, FunctionEntry, Errors, AnalysisOption)
    .run(TUD);
}

std::unique_ptr<ASTConsumer> EditCTypeEditTypeAction::newASTConsumer() {
  return std::make_unique<EditCType>(Model,
                                     Type,
                                     MetaAddress::invalid(),
                                     Errors,
                                     AnalysisOption);
}

std::unique_ptr<ASTConsumer> EditCTypeEditFunctionAction::newASTConsumer() {
  return std::make_unique<EditCType>(Model,
                                     std::nullopt,
                                     FunctionEntry,
                                     Errors,
                                     AnalysisOption);
}

std::unique_ptr<ASTConsumer> EditCTypeAddTypeAction::newASTConsumer() {
  return std::make_unique<EditCType>(Model,
                                     std::nullopt,
                                     MetaAddress::invalid(),
                                     Errors,
                                     AnalysisOption);
}

std::unique_ptr<ASTConsumer>
EditCTypeAction::CreateASTConsumer(CompilerInstance &, llvm::StringRef) {
  return newASTConsumer();
}

bool EditCTypeAction::BeginInvocation(clang::CompilerInstance &CI) {
  DiagConsumer = new EditCTypeDiagnosticConsumer(CI.getDiagnostics());
  CI.getDiagnostics().setClient(DiagConsumer, false);
  return true;
}

void EditCTypeAction::EndSourceFile() {
  if (DiagConsumer) {
    for (auto &Error : DiagConsumer->extractErrors())
      Errors.emplace_back(std::move(Error));
  }
}

void EditCTypeDiagnosticConsumer::EndSourceFile() {
  Client->EndSourceFile();
}

using Level = DiagnosticsEngine::Level;
void EditCTypeDiagnosticConsumer::HandleDiagnostic(Level DiagLevel,
                                                   const Diagnostic &Info) {
  SmallString<100> OutStr;
  Info.FormatDiagnostic(OutStr);

  llvm::raw_svector_ostream DiagMessageStream(OutStr);

  std::string Text;
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
  OS.flush();

  unsigned Line = 0;
  unsigned Column = 0;
  std::string FileName;
  if (Info.getLocation().isValid()) {
    FullSourceLoc Location(Info.getLocation(), Info.getSourceManager());
    Line = Location.getLineNumber();
    Column = Location.getColumnNumber();
  }

  // Report all the messages coming from clang.
  Errors.emplace_back("clang:" + std::to_string(Line) + ":"
                      + std::to_string(Column) + ": " + std::move(Text));
}

} // end namespace tooling
} // end namespace clang
