/// \file ImportModelFromCAnalysis.cpp
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

#include "revng/Model/Type.h"
#include "revng/Pipeline/Context.h"
#include "revng/Pipeline/Kind.h"
#include "revng/Pipeline/Option.h"
#include "revng/Pipeline/RegisterAnalysis.h"
#include "revng/Pipes/ModelGlobal.h"
#include "revng/Support/YAMLTraits.h"
#include "revng/TupleTree/TupleTreeDiff.h"

#include "revng-c/HeadersGeneration/ModelToHeader.h"

#include "HeaderToModel.h"
#include "ImportModelFromCAnalysis.h"
#include "ImportModelFromCHelpers.h"

using namespace llvm;
using namespace clang;
using namespace clang::tooling;

static constexpr std::string_view RevngInputCFile = "revng-input.c";

static std::vector<std::string>
getOptionsfromCFGFile(llvm::StringRef FilePath) {
  std::vector<std::string> Result;

  std::fstream TheFile;
  TheFile.open(FilePath.str());
  if (TheFile.is_open()) {
    std::string TheOption;
    while (getline(TheFile, TheOption)) {
      if (TheOption[0] == '-')
        Result.push_back(TheOption);
    }
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

struct ImportModelFromCAnalysis {
  static constexpr auto Name = "ImportModelFromC";

  constexpr static std::tuple Options = { pipeline::Option("location-to-edit",
                                                           ""),
                                          pipeline::Option("ccode", "") };

  std::vector<std::vector<pipeline::Kind *>> AcceptedKinds = {};

  llvm::Error
  run(pipeline::Context &Ctx, std::string LocationToEdit, std::string CCode) {
    enum ImportModelFromCOption TheOption;
    auto &Model = revng::getWritableModelFromContext(Ctx);

    // This will be used iff {Edit|Add}TypeFeature is used.
    std::optional<model::Type *> TypeToEdit;

    // This will be used iff EditFunctionPrototypeFeature is used.
    std::optional<model::Function> FunctionToBeEdited;

    if (LocationToEdit.empty()) {
      // This is the default option of the analysis.
      TheOption = ImportModelFromCOption::AddType;
    } else {
      if (auto L = pipeline::locationFromString(revng::ranks::Function,
                                                LocationToEdit)) {
        auto Key = std::get<0>(L->at(revng::ranks::Function));
        auto Iterator = Model->Functions().find(Key);
        if (Iterator == Model->Functions().end()) {
          return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                         "Couldn't find the function "
                                           + LocationToEdit);
        }

        FunctionToBeEdited = *Iterator;
        TheOption = ImportModelFromCOption::EditFunctionPrototype;
      } else if (auto L = pipeline::locationFromString(revng::ranks::Type,
                                                       LocationToEdit)) {
        auto Key = std::get<0>(L->at(revng::ranks::Type));
        auto TypeKind = std::get<1>(L->at(revng::ranks::Type));

        auto Iterator = Model->Types().find({ Key, TypeKind });
        if (Iterator == Model->Types().end()) {
          return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                         "Couldn't find the type "
                                           + LocationToEdit);
        }

        TypeToEdit = Iterator->get();
        TheOption = ImportModelFromCOption::EditType;
      } else {
        return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                       "Please provide a location that is "
                                       "supported by our Model");
      }
    }

    // Create a temporary directory for the filtered model's header file.
    llvm::SmallString<128> TemporaryDir;
    llvm::sys::fs::createUniqueDirectory("revng-filtered-model", TemporaryDir);

    constexpr std::string_view FilteredModelHeaderAsPTML = "filtered-model-"
                                                           "header-ptml.h";
    llvm::SmallString<160> FilterModelPath;
    llvm::sys::path::append(FilterModelPath,
                            TemporaryDir.str().str(),
                            FilteredModelHeaderAsPTML);

    std::error_code EC;
    llvm::raw_fd_ostream Header(FilterModelPath.str().str(), EC);
    if (EC) {
      return llvm::createStringError(EC,
                                     "Couldn't create the file for "
                                     "filtered-model-header-ptml.h: "
                                       + EC.message());
    }

    if (TheOption == ImportModelFromCOption::EditType) {
      // For all the types other than functions and typedefs, generate forward
      // declarations.
      if (not isa<model::RawFunctionType>(*TypeToEdit)
          and not isa<model::CABIFunctionType>(*TypeToEdit)
          and not isa<model::TypedefType>(*TypeToEdit)) {
        ptml::PTMLCBuilder ThePTMLCBuilder(true);
        ptml::PTMLIndentedOstream ThePTMLStream(Header, 4, true);
        Header << ThePTMLCBuilder.getLineComment("The type we are editing.");
        // The definition of this type will be at the end of the file.
        printForwardDeclaration(**TypeToEdit, ThePTMLStream, ThePTMLCBuilder);
        Header << '\n';
      }

      // Find all types whose definition depends on the type we are editing.
      auto TypesThatDependOnTypeWeEdit = populateDependencies(*TypeToEdit,
                                                              Model);
      dumpModelToHeader(*Model,
                        Header,
                        TypesThatDependOnTypeWeEdit,
                        MetaAddress::invalid(),
                        true);
    } else if (TheOption == ImportModelFromCOption::EditFunctionPrototype) {
      auto FunctionAddress = FunctionToBeEdited->Entry();
      dumpModelToHeader(*Model, Header, {}, FunctionAddress, true);
    } else {
      revng_assert(TheOption == ImportModelFromCOption::AddType);
      // We have nothing to ignore.
      dumpModelToHeader(*Model, Header, {}, MetaAddress::invalid(), true);
    }

    Header.close();

    std::string FilteredHeader = std::string("#include \"")
                                 + FilterModelPath.str().str()
                                 + std::string("\"");
    TupleTree<model::Binary> OutModel(Model);

    std::optional<revng::ParseCCodeError> Error;
    std::unique_ptr<HeaderToModelAction> Action;

    if (TheOption == ImportModelFromCOption::EditType) {
      Action = std::make_unique<HeaderToModelEditTypeAction>(OutModel,
                                                             Error,
                                                             TypeToEdit);
    } else if (TheOption == ImportModelFromCOption::EditFunctionPrototype) {
      using EditFunctionPrototype = HeaderToModelEditFunctionAction;
      Action = std::make_unique<EditFunctionPrototype>(OutModel,
                                                       Error,
                                                       FunctionToBeEdited);
    } else {
      Action = std::make_unique<HeaderToModelAddTypeAction>(OutModel, Error);
    }

    // Find compile flags to be applied to clang.
    auto MaybeCompileCFGPath = revng::ResourceFinder.findFile("share/revng-c/"
                                                              "compile-flags."
                                                              "cfg");
    if (not MaybeCompileCFGPath) {
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "Couldn't find compile-flags.cfg");
    }

    // Since the `--config` is just a clang Driver option, we need to parse it
    // manually.
    auto FromCFGFile = getOptionsfromCFGFile(*MaybeCompileCFGPath);
    std::vector<std::string> Compilation(FromCFGFile);
    Compilation.push_back("-xc");

    // Find stdbool.h.
    auto MaybeStdBoolHeaderPath = findHeaderFile("lib64/llvm/"
                                                 "llvm/lib/"
                                                 "clang/16/"
                                                 "include/"
                                                 "stdbool.h");
    if (not MaybeStdBoolHeaderPath) {
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "Couldn't find stdbool.h");
    }
    Compilation.push_back("-I" + *MaybeStdBoolHeaderPath);

    // Find stdint.h.
    auto MaybeStdIntHeaderPath = findHeaderFile("lib64/llvm/"
                                                "llvm/lib/"
                                                "clang/16/"
                                                "include/"
                                                "stdint.h");
    if (not MaybeStdIntHeaderPath) {
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "Couldn't find stdint.h");
    }
    Compilation.push_back("-I" + *MaybeStdIntHeaderPath);

    // Find revng-primitive-types.h and revng-attributes.h.
    auto MaybeRevngHeaderPath = findHeaderFile("share/revng-c/"
                                               "include/"
                                               "revng-"
                                               "primitive-"
                                               "types.h");
    if (not MaybeRevngHeaderPath) {
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "Couldn't find revng-primitive-types.h");
    }
    Compilation.push_back("-I" + *MaybeRevngHeaderPath);

    FilteredHeader += "\n";
    FilteredHeader += CCode;

    if (not clang::tooling::runToolOnCodeWithArgs(std::move(Action),
                                                  FilteredHeader,
                                                  Compilation,
                                                  RevngInputCFile)) {
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "Unable to run clang");
    }

    // Check if an error was reported by clang or revng during parsing of C
    // code.
    if (Error) {
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     (*Error).ErrorMessage);
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

pipeline::RegisterAnalysis<ImportModelFromCAnalysis> ImportModelFromCReg;
