//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <string>
#include <type_traits>

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Progress.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/ADT/UpcastablePointer.h"
#include "revng/Model/Binary.h"
#include "revng/Model/BinaryIdentifier.h"
#include "revng/Model/DynamicFunction.h"
#include "revng/Model/Importer/TypeCopier.h"
#include "revng/Model/Pass/DeduplicateCollidingNames.h"
#include "revng/Model/Pass/DeduplicateEquivalentTypes.h"
#include "revng/Model/Pass/FlattenPrimitiveTypedefs.h"
#include "revng/Model/Pass/PurgeUnnamedAndUnreachableTypes.h"
#include "revng/Model/Type.h"
#include "revng/Support/InitRevng.h"
#include "revng/Support/MetaAddress/YAMLTraits.h"
#include "revng/Support/YAMLTraits.h"
#include "revng/TupleTree/TupleTreeDiff.h"

using namespace llvm;

static cl::OptionCategory ThisToolCategory("Tool options", "");

static cl::opt<std::string> DestinationPath(cl::Positional,
                                            cl::cat(ThisToolCategory),
                                            cl::desc("<destination model>"),
                                            cl::value_desc("model"),
                                            cl::init("-"));

static cl::list<std::string> SourcePaths(cl::Positional,
                                         cl::cat(ThisToolCategory),
                                         cl::desc("<source model>"));

static cl::opt<std::string> OutputPath("o",
                                       cl::init("-"),
                                       cl::desc("Output "
                                                "path"),
                                       cl::value_desc("path"),
                                       cl::cat(ThisToolCategory));

static Logger Log("model-copy");
static ExitOnError AbortOnError;

template<typename T>
UpcastablePointer<model::Type> &getType(T &);

template<>
UpcastablePointer<model::Type> &
getType<model::DynamicFunction>(model::DynamicFunction &Object) {
  return Object.Prototype();
};

template<>
UpcastablePointer<model::Type> &
getType<model::Function>(model::Function &Object) {
  return Object.Prototype();
};

template<>
UpcastablePointer<model::Type> &
getType<model::Segment>(model::Segment &Object) {
  return Object.Type();
};

template<typename T>
constexpr bool HasType = true;

template<>
constexpr bool HasType<model::BinaryIdentifier> = false;

static void copyModelInto(StringRef SourcePath,
                          TupleTree<model::Binary> &Destination) {

  auto MaybeSourceModel = TupleTree<model::Binary>::fromFileOrSTDIN(SourcePath);
  if (not MaybeSourceModel)
    AbortOnError(MaybeSourceModel.takeError());

  TupleTree<model::Binary> &Source = *MaybeSourceModel;

  TypeCopier Copier(Source, Destination);

  auto &SourceList = Source->ImportedDynamicFunctions();
  auto &DestinationList = Destination->ImportedDynamicFunctions();

  auto GetDefinition =
    [](UpcastablePointer<model::Type> &Type) -> model::TypeDefinition * {
    if (Type.isEmpty())
      return nullptr;
    return Type->tryGetAsDefinition();
  };

  auto Merge = [&GetDefinition, &Copier](auto &SourceList,
                                         auto &DestinationList) {
    Task T(SourceList.size(), "Processing objects");
    for (auto &SourceObject : SourceList) {
      T.advance(getNameFromYAMLScalar(SourceObject.key()), true);

      revng_log(Log,
                "Considering " << getNameFromYAMLScalar(SourceObject.key()));
      LoggerIndent Indent(Log);

      bool AlreadyExists = false;
      if constexpr (std::tuple_size_v<decltype(SourceObject.key())> == 1) {
        AlreadyExists = DestinationList
                          .contains(std::get<0>(SourceObject.key()));
      } else {
        AlreadyExists = DestinationList.contains(SourceObject.key());
      }

      if (AlreadyExists) {
        revng_log(Log, "Already present in destination, ignoring");
        continue;
      }

      auto NewObject = SourceObject;

      if constexpr (HasType<std::remove_cvref_t<decltype(SourceObject)>>) {
        if (auto *TypeDefinition = GetDefinition(getType(SourceObject)))
          getType(NewObject) = Copier.copyTypeInto(*TypeDefinition);
      }

      revng_log(Log, "Copying");
      DestinationList.insert(std::move(NewObject));
    }
  };

  Task T(5, "Importing model parts");

  T.advance("ImportedDynamicFunctions", true);
  Merge(Source->ImportedDynamicFunctions(),
        Destination->ImportedDynamicFunctions());

  T.advance("Functions", true);
  Merge(Source->Functions(), Destination->Functions());

  T.advance("Segments", true);
  Merge(Source->Segments(), Destination->Segments());

  T.advance("Binaries", true);
  Merge(Source->Binaries(), Destination->Binaries());

  T.advance("Finalizing", true);
  Copier.finalize();
}

int main(int Argc, char *Argv[]) {
  revng::InitRevng X(Argc, Argv, "", { &ThisToolCategory });

  std::error_code EC;
  ToolOutputFile Output(OutputPath, EC, sys::fs::OF_None);
  AbortOnError(errorCodeToError(EC));

  Task T1(7, "Merging models");

  T1.advance("Load destination model", true);
  using TTB = TupleTree<model::Binary>;
  auto MaybeDestinationModel = TTB::fromFileOrSTDIN(DestinationPath);
  if (not MaybeDestinationModel)
    AbortOnError(MaybeDestinationModel.takeError());

  T1.advance("Import destination models", true);
  Task T2(SourcePaths.size(), "Processing source models");
  for (auto &SourcePath : SourcePaths) {
    T2.advance(SourcePath, true);
    copyModelInto(SourcePath, *MaybeDestinationModel);
  }
  T2.complete();

  T1.advance("Flattening primitive typedefs", true);
  revng_log(Log, "flattenPrimitiveTypedefs");
  model::flattenPrimitiveTypedefs(*MaybeDestinationModel);

  T1.advance("Deduplicating equivalent types", true);
  revng_log(Log, "deduplicateEquivalentTypes");
  model::deduplicateEquivalentTypes(*MaybeDestinationModel);

  T1.advance("Deduplicating colliding names", true);
  revng_log(Log, "deduplicateCollidingNames");
  model::deduplicateCollidingNames(*MaybeDestinationModel);

  T1.advance("Purging invalid types", true);
  revng_log(Log, "purgeInvalidTypes");
  model::purgeInvalidTypes(*MaybeDestinationModel);

  T1.advance("Emitting final model", true);
  MaybeDestinationModel->serialize(Output.os());
  Output.keep();

  return EXIT_SUCCESS;
}
