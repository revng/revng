/// \brief Use to edit Types by omitting rewriting of Model directly

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <fstream>
#include <string>
#include <tuple>
#include <vector>

#include "llvm/Support/Error.h"
#include "llvm/Support/ToolOutputFile.h"

#include "clang/Driver/Driver.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "clang/StaticAnalyzer/Frontend/FrontendActions.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"

#include "revng/ABI/Definition.h"
#include "revng/Clift/Clift.h"
#include "revng/CliftEmitC/CEmitter.h"
#include "revng/CliftEmitC/Configuration.h"
#include "revng/CliftEmitC/Headers.h"
#include "revng/CliftEmitC/TypeDefinitionEmitter.h"
#include "revng/CliftImportModel/ImportModel.h"
#include "revng/ImportFromC/ImportFromCAnalysis.h"
#include "revng/Model/Binary.h"
#include "revng/Model/VerifyHelper.h"
#include "revng/PTML/CTokenEmitter.h"
#include "revng/Pipeline/Context.h"
#include "revng/Pipeline/Kind.h"
#include "revng/Pipeline/Option.h"
#include "revng/Pipeline/RegisterAnalysis.h"
#include "revng/Pipes/ModelGlobal.h"
#include "revng/Support/PathList.h"
#include "revng/Support/TemporaryFile.h"
#include "revng/Support/YAMLTraits.h"
#include "revng/TupleTree/TupleTreeDiff.h"

#include "HeaderToModel.h"
#include "ImportFromCAnalysis.h"

using namespace llvm;
using namespace clang;
using namespace clang::tooling;

static constexpr std::string_view InputCFile = "revng-input.c";

static std::vector<std::string>
getOptionsFromCFGFile(llvm::StringRef FilePath) {
  std::vector<std::string> Result;

  auto MaybeBuffer = llvm::MemoryBuffer::getFile(FilePath);
  revng_assert(MaybeBuffer);

  llvm::SmallVector<llvm::StringRef, 0> Lines;
  MaybeBuffer->get()->getBuffer().split(Lines, '\n');
  for (llvm::StringRef &Line : Lines) {
    if (Line.size() > 0 and Line[0] == '-')
      Result.push_back(Line.str());
  }

  return Result;
}

static std::optional<std::string> findHeaderFile(const std::string &File) {
  auto MaybeHeaderPath = revng::ResourceFinder.findFile(File);
  if (not MaybeHeaderPath)
    return std::nullopt;
  auto Index = (*MaybeHeaderPath).rfind('/');
  if (Index == std::string::npos)
    return std::nullopt;

  return (*MaybeHeaderPath).substr(0, Index);
}

static bool isSeparateDeclarationAllowed(const model::TypeDefinition &T) {
  return llvm::isa<model::StructDefinition>(&T)
         or llvm::isa<model::UnionDefinition>(&T);
}

static std::pair<std::unique_ptr<mlir::MLIRContext>,
                 mlir::OwningOpRef<mlir::ModuleOp>>
makeHeaderModule(const model::Binary &Model) {
  std::unique_ptr<mlir::MLIRContext> Context = clift::makeContext();
  mlir::OwningOpRef<mlir::ModuleOp> Module = clift::makeModule(*Context);

  clift::importAllModelTypes(Model, Module.get());
  clift::importAllModelFunctionDeclarations(Model, Module.get());
  clift::importAllModelSegmentDeclarations(Model, Module.get());

  clift::importDescriptiveInfo(Model, Module.get());

  return std::make_pair(std::move(Context), std::move(Module));
}

static Logger Log("header-to-model-errors");

struct ImportFromCAnalysis {
  static constexpr auto Name = "import-from-c";

  constexpr static std::tuple Options = { pipeline::Option("location-to-edit",
                                                           ""),
                                          pipeline::Option("ccode", "") };

  std::vector<std::vector<pipeline::Kind *>> AcceptedKinds = {};

  llvm::Error run(pipeline::ExecutionContext &EC,
                  std::string LocationToEdit,
                  std::string CCode) {
    auto &Model = revng::getWritableModelFromContext(EC);
    return run(Model, LocationToEdit, CCode);
  }

  static llvm::Error run(TupleTree<model::Binary> &Model,
                         llvm::StringRef LocationToEdit,
                         llvm::StringRef CCode) {
    enum ImportFromCOption TheOption;

    // This will be used iff {Edit|Add}TypeFeature is used.
    model::TypeDefinition *TypeToEdit = nullptr;

    // This will be used iff EditFunctionPrototypeFeature is used.
    model::Function *FunctionToEdit = nullptr;

    namespace RRanks = revng::ranks;
    if (LocationToEdit.empty()) {
      // This is the default option of the analysis.
      TheOption = ImportFromCOption::AddType;
    } else {
      if (auto L = pipeline::locationFromString(revng::ranks::Function,
                                                LocationToEdit)) {
        auto &&[Key] = L->at(revng::ranks::Function);
        auto Iterator = Model->Functions().find(Key);
        if (Iterator == Model->Functions().end()) {
          return revng::createError("Couldn't find the function "
                                    + LocationToEdit);
        }

        FunctionToEdit = &*Iterator;
        TheOption = ImportFromCOption::EditFunctionPrototype;
      } else if (auto L = pipeline::locationFromString(RRanks::TypeDefinition,
                                                       LocationToEdit)) {
        auto &&[Key, Kind] = L->at(revng::ranks::TypeDefinition);
        auto Iterator = Model->TypeDefinitions().find({ Key, Kind });
        if (Iterator == Model->TypeDefinitions().end()) {
          return revng::createError("Couldn't find the type " + LocationToEdit);
        }

        TypeToEdit = Iterator->get();
        TheOption = ImportFromCOption::EditType;
      } else {
        return revng::createError("Invalid location");
      }
    }

    auto MaybeFilterModelPath = TemporaryFile::make("filtered-model-header-"
                                                    "ptml",
                                                    "h");
    if (not MaybeFilterModelPath)
      return MaybeFilterModelPath.takeError();

    TemporaryFile &FilterModelPath = MaybeFilterModelPath.get();
    std::error_code ErrorCode;
    llvm::raw_fd_ostream Out(FilterModelPath.path(), ErrorCode);
    if (ErrorCode) {
      return llvm::createStringError(ErrorCode,
                                     "Couldn't open file for "
                                     "filtered-model-header-ptml.h: "
                                       + ErrorCode.message());
    }

    auto [Context, HeaderModule] = makeHeaderModule(*Model);

    TypeEmitterConfiguration Configuration = {
      .TypeToOmit = {},
      .EmitMaximumEnumValue = true,
      .ExplicitPadding = false,
    };

    // Extra variable is here to extend the lifetime of the string.
    std::string EditedTypeHandle;

    if (TheOption == ImportFromCOption::EditType) {
      EditedTypeHandle = pipeline::locationString(revng::ranks::TypeDefinition,
                                                  TypeToEdit->key());
      Configuration.TypeToOmit = EditedTypeHandle;
    }

    {
      ptml::CTokenEmitter Tokens(Out, ptml::Tagging::Disabled);
      emitCommonIncludes(Tokens, abi::getDataModel(*Model));

      if (TheOption == ImportFromCOption::EditType
          and isSeparateDeclarationAllowed(*TypeToEdit)) {

        mlir::Location Loc = mlir::UnknownLoc::get(Context.get());
        auto EmitError = [&]() -> mlir::InFlightDiagnostic {
          return Context->getDiagEngine().emit(Loc,
                                               mlir::DiagnosticSeverity::Error);
        };
        auto Current = clift::importType(EmitError, Context.get(), *TypeToEdit);

        // TODO: we only need information about one type, don't import them all!
        clift::importDescriptiveInfo(*Model, *HeaderModule);

        TypeDefinitionEmitter TDE(Tokens,
                                  abi::getDataModel(*Model),
                                  Configuration);
        TDE.emitForwardDeclaration(Current);
        Tokens.emitNewline();
      }

      emitTypes(Tokens, *HeaderModule, Configuration);
    }

    Out.close();

    std::string FilteredHeader = std::string("#include \"")
                                 + FilterModelPath.path().str()
                                 + std::string("\"");
    TupleTree<model::Binary> OutModel(Model);

    ImportingErrorList Errors;
    std::unique_ptr<HeaderToModelAction> Action;

    if (TheOption == ImportFromCOption::EditType) {
      revng_assert(TypeToEdit != nullptr);
      Action = std::make_unique<HeaderToModelEditTypeAction>(OutModel,
                                                             Errors,
                                                             TypeToEdit->key());
    } else if (TheOption == ImportFromCOption::EditFunctionPrototype) {
      revng_assert(FunctionToEdit != nullptr);
      using EditFunctionPrototype = HeaderToModelEditFunctionAction;
      Action = std::make_unique<EditFunctionPrototype>(OutModel,
                                                       Errors,
                                                       FunctionToEdit->Entry());
    } else {
      Action = std::make_unique<HeaderToModelAddTypeAction>(OutModel, Errors);
    }

    // Find compile flags to be applied to clang.
    StringRef CompileFlagsPath = "share/revng/compile-flags.cfg";
    auto MaybeCompileCFGPath = revng::ResourceFinder.findFile(CompileFlagsPath);
    if (not MaybeCompileCFGPath) {
      return revng::createError("Couldn't find compile-flags.cfg");
    }

    // Since the `--config` is just a clang Driver option, we need to parse it
    // manually.
    auto FromCFGFile = getOptionsFromCFGFile(*MaybeCompileCFGPath);
    std::vector<std::string> Compilation(FromCFGFile);
    Compilation.push_back("-xc");

    SmallString<16> CompilerHeadersPath;
    {
      StringRef LLVMLibrary = getLibrariesFullPath().at("libLLVMSupport");
      using namespace llvm::sys::path;
      SmallString<16> ClangPath;
      append(ClangPath, parent_path(parent_path(LLVMLibrary)));
      append(ClangPath, Twine("bin"));
      append(ClangPath, Twine("clang"));
      CompilerHeadersPath = clang::driver::Driver::GetResourcesPath(ClangPath);
      append(CompilerHeadersPath, Twine("include"));
    }
    Compilation.push_back("-I" + CompilerHeadersPath.str().str());

    // Find primitive-types.h and attributes.h.
    const char *PrimitivesHeader = "share/revng/include/"
                                   "primitive-types.h";
    auto MaybePrimitiveHeaderPath = findHeaderFile(PrimitivesHeader);
    if (not MaybePrimitiveHeaderPath) {
      return revng::createError("Couldn't find primitive-types.h");
    }
    Compilation.push_back("-I" + *MaybePrimitiveHeaderPath);

    FilteredHeader += "\n";
    FilteredHeader += CCode;

    if (not clang::tooling::runToolOnCodeWithArgs(std::move(Action),
                                                  FilteredHeader,
                                                  Compilation,
                                                  InputCFile)) {
      return revng::createError("Unable to run clang");
    }

    // Check if an error was reported by clang or revng during parsing of C
    // code.
    if (not Errors.empty()) {
      std::string Result;
      for (auto &Error : Errors)
        Result += std::move(Error);

      revng_log(Log, Result.c_str());
      return revng::createError(Result);
    }

    model::VerifyHelper VH;
    if (not OutModel->verify(VH)) {
      return revng::createError("New model does not verify: " + VH.getReason());
    }

    // Replace the original Model with the OutModel that contains the changes.
    Model = OutModel;

    return llvm::Error::success();
  }
};

pipeline::RegisterAnalysis<ImportFromCAnalysis> ImportFromCReg;

struct ImportFromCConfiguration {
  std::string LocationToEdit;
  std::string CCode;
};

template<>
struct llvm::yaml::MappingTraits<ImportFromCConfiguration> {
  static void mapping(IO &IO, ImportFromCConfiguration &Fields) {
    IO.mapRequired("LocationToEdit", Fields.LocationToEdit);
    IO.mapRequired("CCode", Fields.CCode);
  }
};

namespace revng::pypeline::analyses {

llvm::Error ImportFromC::run(Model &Model,
                             const Request &Incoming,
                             llvm::StringRef StrConfiguration) {
  auto
    MaybeConfiguration = fromString<ImportFromCConfiguration>(StrConfiguration);
  if (not MaybeConfiguration)
    return MaybeConfiguration.takeError();

  ImportFromCConfiguration &Configuration = MaybeConfiguration.get();
  return ImportFromCAnalysis::run(Model.get(),
                                  Configuration.LocationToEdit,
                                  Configuration.CCode);
}

} // namespace revng::pypeline::analyses
