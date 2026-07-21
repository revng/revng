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

#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "clang/StaticAnalyzer/Frontend/FrontendActions.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"

#include "revng/ClangToModel/CompileFlags.h"
#include "revng/Clift/Clift.h"
#include "revng/CliftEmitC/CEmitter.h"
#include "revng/CliftEmitC/Configuration.h"
#include "revng/CliftEmitC/Headers.h"
#include "revng/CliftEmitC/TypeDefinitionEmitter.h"
#include "revng/CliftImportModel/ImportModel.h"
#include "revng/EditCType/EditCTypeAnalysis.h"
#include "revng/Model/ABI/Definition.h"
#include "revng/Model/Binary.h"
#include "revng/Model/EnumDefinition.h"
#include "revng/Model/VerifyHelper.h"
#include "revng/PTML/CTokenEmitter.h"
#include "revng/Ranks/Ranks.h"
#include "revng/Support/PathList.h"
#include "revng/Support/ResourceFinder.h"
#include "revng/Support/TemporaryFile.h"
#include "revng/Support/YAMLTraits.h"
#include "revng/TupleTree/TupleTreeDiff.h"

#include "EditCType.h"
#include "EditCTypeAnalysis.h"

using namespace llvm;
using namespace clang;
using namespace clang::tooling;

static bool isSeparateDeclarationAllowed(const model::TypeDefinition &T) {
  return llvm::isa<model::StructDefinition>(&T)
         or llvm::isa<model::UnionDefinition>(&T)
         or llvm::isa<model::EnumDefinition>(&T);
}

static Logger Log("edit-c-type-clang-input");

namespace rr = revng::ranks;

struct EditCTypeAnalysis {
private:
  struct EditCTypeState {
    // Reference to the original Model.
    TupleTree<model::Binary> &Model;

    // A copy of the original Model.
    TupleTree<model::Binary> Result;

    // The list of errors reported either by clang or us.
    ImportingErrorList Errors = {};

    // Context behind the action being performed.
    std::unique_ptr<EditCTypeAction> Action = nullptr;

    // This pointer is only set when a *type* is being *edited*.
    // It will remain as `nullptr` in all other cases.
    model::TypeDefinition *TypeToEdit = nullptr;

  public:
    EditCTypeState(TupleTree<model::Binary> &Model) :
      Model(Model), Result(Model) {}

    EditCTypeState(EditCTypeState &&) = delete;
    EditCTypeState &operator=(EditCTypeState &&) = delete;
    EditCTypeState(const EditCTypeState &) = delete;
    EditCTypeState &operator=(const EditCTypeState &) = delete;
  };

  static llvm::Error prepareActionHelper(EditCTypeState &Out,
                                         llvm::StringRef LocationToEdit) {
    if (LocationToEdit.empty()) {
      Out.Action = std::make_unique<EditCTypeAddTypeAction>(Out.Result,
                                                            Out.Errors);

    } else if (auto L = pipeline::locationFromString(rr::Function,
                                                     LocationToEdit)) {
      auto &&[Key] = L->at(rr::Function);
      auto Iterator = Out.Model->Functions().find(Key);
      if (Iterator == Out.Model->Functions().end())
        return revng::createError("Couldn't find the function "
                                  + LocationToEdit.str());

      using EditFunction = EditCTypeEditFunctionAction;
      Out.Action = std::make_unique<EditFunction>(Out.Result,
                                                  Out.Errors,
                                                  Iterator->Entry());

    } else if (auto L = pipeline::locationFromString(rr::TypeDefinition,
                                                     LocationToEdit)) {
      auto &&[Key, Kind] = L->at(rr::TypeDefinition);
      auto Iterator = Out.Model->TypeDefinitions().find({ Key, Kind });
      if (Iterator == Out.Model->TypeDefinitions().end())
        return revng::createError("Couldn't find the type "
                                  + LocationToEdit.str());

      Out.TypeToEdit = Iterator->get();

      using EditType = EditCTypeEditTypeAction;
      Out.Action = std::make_unique<EditType>(Out.Result,
                                              Out.Errors,
                                              Out.TypeToEdit->key());

    } else {
      return revng::createError("Invalid location");
    }

    return llvm::Error::success();
  }

  static std::pair<TemporaryFile, std::unique_ptr<llvm::raw_fd_ostream>>
  setupTemporaryFile() {
    auto File = TemporaryFile::make("filtered-model-header-ptml", "h");
    revng_check(File);

    std::error_code ErrorCode;
    auto Out = std::make_unique<llvm::raw_fd_ostream>(File->path(), ErrorCode);
    revng_assert(not ErrorCode);

    return { std::move(*File), std::move(Out) };
  }

  static void emitFilteredHeader(llvm::raw_fd_ostream &Out,
                                 const model::Binary &Model,
                                 model::TypeDefinition *TypeToEdit) {
    auto [Context, HeaderModule] = clift::makeHeaderModule(Model,
                                                           /*IncludeGlobals=*/
                                                           false);

    TypeEmitterConfiguration Configuration = {
      .TypeToOmit = {},
      .EmitMaximumEnumValue = true,
      .ExplicitPadding = false,
    };

    // This variable is necessary to ensure the string (when it's used) survives
    // until the end of the function.
    std::string EditedTypeHandle = {};
    if (TypeToEdit != nullptr)
      EditedTypeHandle = pipeline::locationString(rr::TypeDefinition,
                                                  TypeToEdit->key());
    Configuration.TypeToOmit = EditedTypeHandle;

    ptml::CTokenEmitter Tokens(Out, ptml::Tagging::Disabled);
    emitCommonIncludes(Tokens, Model.targetDataModel());

    if (TypeToEdit != nullptr and isSeparateDeclarationAllowed(*TypeToEdit)) {
      TypeDefinitionEmitter TDE(Tokens, Model.targetDataModel(), Configuration);

      auto Current = clift::importType(Context.get(), *TypeToEdit);
      TDE.emitForwardDeclaration(Current);
      Tokens.emitNewline();
    }

    emitTypes(Tokens, *HeaderModule, Configuration);
  }

  static llvm::Error parseCompiledC(EditCTypeState &State,
                                    llvm::StringRef HeaderPath,
                                    llvm::StringRef CCode) {
    std::string InputC = std::string("#include \"") + HeaderPath.str()
                         + std::string("\"\n") + CCode.str();
    revng_log(Log, "Real input:\n" << InputC << "\n");

    static constexpr std::string_view InputFileName = "revng-input.c";
    if (not clang::tooling::runToolOnCodeWithArgs(std::move(State.Action),
                                                  InputC,
                                                  revng::getClangCompileFlags(),
                                                  InputFileName)) {
      return revng::createError("Unable to run clang");
    }

    // Check if an error was reported by clang or revng during parsing of C
    // code.
    //
    // TODO: adjusting line numbers to account for the lines we append would
    //       make UX considerably better.
    if (not State.Errors.empty()) {
      std::string ConcatenatedErrorMessage;
      for (auto &Error : State.Errors)
        ConcatenatedErrorMessage += std::move(Error);

      // TODO: the best thing for the UI would be having clang emit SARIF,
      //       we should add an option to emit it instead (it's still worth
      //       keeping the basic output for CLI users).
      return revng::createError(ConcatenatedErrorMessage);
    }

    model::VerifyHelper VH;
    if (not State.Result->verify(VH)) {
      return revng::createError("New model does not verify: " + VH.getReason());
    }

    // Replace the original Model with the Result that contains the changes.
    std::swap(State.Model, State.Result);

    return llvm::Error::success();
  }

public:
  static llvm::Error run(TupleTree<model::Binary> &Model,
                         llvm::StringRef LocationToEdit,
                         llvm::StringRef CCode) {
    EditCTypeState State(Model);
    if (auto Error = prepareActionHelper(State, LocationToEdit))
      return Error;

    auto [FilteredTypeHeader, Out] = setupTemporaryFile();
    emitFilteredHeader(*Out, *Model, State.TypeToEdit);
    Out->close();

    if (Log.isEnabled()) {
      std::ifstream Stream(FilteredTypeHeader.path().str());
      Log << "Filtered header:\n" << Stream.rdbuf() << "\n" << DoLog;
    }

    return parseCompiledC(State, FilteredTypeHeader.path().str(), CCode);
  }
};

struct EditCTypeConfiguration {
  std::string LocationToEdit;
  std::string CCode;
};

template<>
struct llvm::yaml::MappingTraits<EditCTypeConfiguration> {
  static void mapping(IO &IO, EditCTypeConfiguration &Fields) {
    IO.mapRequired("LocationToEdit", Fields.LocationToEdit);
    IO.mapRequired("CCode", Fields.CCode);
  }
};

namespace revng::pypeline::analyses {

llvm::Error EditCType::run(Model &Model,
                           const Request &Incoming,
                           llvm::StringRef StrConfiguration) {
  auto
    MaybeConfiguration = fromString<EditCTypeConfiguration>(StrConfiguration);
  if (not MaybeConfiguration)
    return MaybeConfiguration.takeError();

  EditCTypeConfiguration &Configuration = MaybeConfiguration.get();
  return EditCTypeAnalysis::run(Model.get(),
                                Configuration.LocationToEdit,
                                Configuration.CCode);
}

} // namespace revng::pypeline::analyses
