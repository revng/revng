/// \file ImportFromCAnalysis.cpp
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

#include "revng/Model/Binary.h"
#include "revng/Pipeline/Context.h"
#include "revng/Pipeline/Kind.h"
#include "revng/Pipeline/Option.h"
#include "revng/Pipeline/RegisterAnalysis.h"
#include "revng/Pipes/ModelGlobal.h"
#include "revng/Support/PathList.h"
#include "revng/Support/TemporaryFile.h"
#include "revng/Support/YAMLTraits.h"
#include "revng/TupleTree/TupleTreeDiff.h"

#include "revng-c/Backend/DecompiledCCodeIndentation.h"
#include "revng-c/HeadersGeneration/ModelToHeader.h"

#include "HeaderToModel.h"
#include "ImportFromCAnalysis.h"
#include "ImportFromCHelpers.h"

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

static Logger<> Log("header-to-model-errors");

struct ImportFromCAnalysis {
  static constexpr auto Name = "import-from-c";

  constexpr static std::tuple Options = { pipeline::Option("location-to-edit",
                                                           ""),
                                          pipeline::Option("ccode", "") };

  std::vector<std::vector<pipeline::Kind *>> AcceptedKinds = {};

  llvm::Error run(pipeline::ExecutionContext &EC,
                  std::string LocationToEdit,
                  std::string CCode) {
    enum ImportFromCOption TheOption;
    auto &Model = revng::getWritableModelFromContext(EC);

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
        auto [Key] = L->at(revng::ranks::Function);
        auto Iterator = Model->Functions().find(Key);
        if (Iterator == Model->Functions().end()) {
          return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                         "Couldn't find the function "
                                           + LocationToEdit);
        }

        FunctionToEdit = &*Iterator;
        TheOption = ImportFromCOption::EditFunctionPrototype;
      } else if (auto L = pipeline::locationFromString(RRanks::TypeDefinition,
                                                       LocationToEdit)) {
        auto [Key, Kind] = L->at(revng::ranks::TypeDefinition);
        auto Iterator = Model->TypeDefinitions().find({ Key, Kind });
        if (Iterator == Model->TypeDefinitions().end()) {
          return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                         "Couldn't find the type "
                                           + LocationToEdit);
        }

        TypeToEdit = Iterator->get();
        TheOption = ImportFromCOption::EditType;
      } else {
        return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                       "Invalid location");
      }
    }

    auto MaybeFilterModelPath = TemporaryFile::make("filtered-model-header-"
                                                    "ptml",
                                                    "h");
    if (!MaybeFilterModelPath) {
      std::error_code EC = MaybeFilterModelPath.getError();
      return llvm::createStringError(EC,
                                     "Couldn't create temporary file: "
                                       + EC.message());
    }

    TemporaryFile &FilterModelPath = MaybeFilterModelPath.get();
    std::error_code ErrorCode;
    llvm::raw_fd_ostream Header(FilterModelPath.path(), ErrorCode);
    if (ErrorCode) {
      return llvm::createStringError(ErrorCode,
                                     "Couldn't open file for "
                                     "filtered-model-header-ptml.h: "
                                       + ErrorCode.message());
    }

    ModelToHeaderOptions Options = {
      .GeneratePlainC = true,
      .DisableTypeInlining = true,
    };

    if (TheOption == ImportFromCOption::EditType) {
      // For all the types other than functions and typedefs, generate forward
      // declarations.
      if (not declarationIsDefinition(*TypeToEdit)) {
        llvm::raw_string_ostream Stream(Options.PostIncludes);
        ptml::PTMLCBuilder B(true);
        ptml::PTMLIndentedOstream ThePTMLStream(Stream,
                                                DecompiledCCodeIndentation,
                                                true);
        Stream << B.getLineComment("The type we are editing");
        // The definition of this type will be at the end of the file.
        printForwardDeclaration(*TypeToEdit, ThePTMLStream, B);
        Stream << '\n';
      }

      // Find all types whose definition depends on the type we are editing.
      Options.TypesToOmit = populateDependencies(*TypeToEdit, Model);
    } else if (TheOption == ImportFromCOption::EditFunctionPrototype) {
      Options.FunctionsToOmit.insert(FunctionToEdit->Entry());
    } else {
      revng_assert(TheOption == ImportFromCOption::AddType);
      // We have nothing to ignore
    }

    dumpModelToHeader(*Model, Header, Options);

    Header.close();

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
    StringRef CompileFlagsPath = "share/revng-c/compile-flags.cfg";
    auto MaybeCompileCFGPath = revng::ResourceFinder.findFile(CompileFlagsPath);
    if (not MaybeCompileCFGPath) {
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "Couldn't find compile-flags.cfg");
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
    const char *PrimitivesHeader = "share/revng-c/include/"
                                   "primitive-types.h";
    auto MaybePrimitiveHeaderPath = findHeaderFile(PrimitivesHeader);
    if (not MaybePrimitiveHeaderPath) {
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "Couldn't find primitive-types.h");
    }
    Compilation.push_back("-I" + *MaybePrimitiveHeaderPath);

    FilteredHeader += "\n";
    FilteredHeader += CCode;

    if (not clang::tooling::runToolOnCodeWithArgs(std::move(Action),
                                                  FilteredHeader,
                                                  Compilation,
                                                  InputCFile)) {
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "Unable to run clang");
    }

    // Check if an error was reported by clang or revng during parsing of C
    // code.
    if (not Errors.empty()) {
      std::string Result;
      for (auto &Error : Errors)
        Result += std::move(Error);

      revng_log(Log, Result.c_str());
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     std::move(Result));
    }

    model::VerifyHelper VH(false);
    if (not OutModel->verify(VH)) {
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "New model does not verify: "
                                       + VH.getReason());
    }

    // Replace the original Model with the OutModel that contains the changes.
    Model = OutModel;

    return llvm::Error::success();
  }
};

pipeline::RegisterAnalysis<ImportFromCAnalysis> ImportFromCReg;
