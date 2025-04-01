/// \file WellKnownModels.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/Binary.h"
#include "revng/Model/Importer/TypeCopier.h"
#include "revng/Model/Pass/DeduplicateCollidingNames.h"
#include "revng/Pipeline/RegisterAnalysis.h"
#include "revng/Pipes/ModelGlobal.h"
#include "revng/Support/ResourceFinder.h"

namespace revng::pipes {

class WellKnownFunctionKey {
public:
  model::Architecture::Values Architecture;
  model::ABI::Values ABI;
  std::string Name;

private:
  auto tie() const { return std::tie(Architecture, ABI, Name); }

public:
  bool operator<(const WellKnownFunctionKey &Other) const {
    return tie() < Other.tie();
  }
};

class WellKnownModel {
public:
  TupleTree<model::Binary> FromModel;
  TypeCopier Copier;

public:
  WellKnownModel(TupleTree<model::Binary> &&FromModel,
                 TupleTree<model::Binary> &DestinationModel) :
    FromModel(FromModel), Copier(this->FromModel, DestinationModel) {}
};

class ImportWellKnownModelsAnalysis {
public:
  static constexpr auto Name = "import-well-known-models";

public:
  std::vector<std::vector<pipeline::Kind *>> AcceptedKinds;

public:
  llvm::Error run(pipeline::ExecutionContext &Context) {
    std::vector<std::unique_ptr<WellKnownModel>> WellKnownModels;
    std::map<WellKnownFunctionKey,
             std::pair<WellKnownModel *, model::Function *>>
      WellKnownFunctions;
    TupleTree<model::Binary> &Model = getWritableModelFromContext(Context);

    // Load all well-known models
    for (const std::string &Path :
         revng::ResourceFinder.list("share/revng/well-known-models", ".yml")) {
      auto MaybeModel = TupleTree<model::Binary>::fromFile(Path);
      revng_assert(MaybeModel);
      using namespace std;
      auto NewWKM = make_unique<WellKnownModel>(std::move(*MaybeModel), Model);
      WellKnownModels.push_back(std::move(NewWKM));
    }

    // Create an index of well-known functions
    for (auto &WellKnownModel : WellKnownModels) {
      auto &Model = WellKnownModel->FromModel;
      // Collect exported functions
      for (model::Function &F : Model->Functions()) {
        for (const std::string &ExportedName : F.ExportedNames()) {
          WellKnownFunctions[{
            Model->Architecture(), Model->DefaultABI(), ExportedName }] = {
            WellKnownModel.get(), &F
          };
        }
      }
    }

    for (model::DynamicFunction &F : Model->ImportedDynamicFunctions()) {
      // Compose the key
      WellKnownFunctionKey Key = { Model->Architecture(),
                                   Model->DefaultABI(),
                                   F.Name() };

      // See if it's a well-known function
      auto It = WellKnownFunctions.find(Key);
      if (It != WellKnownFunctions.end()) {
        model::Function *WellKnownFunction = It->second.second;

        // Copy attributes
        F.Attributes() = WellKnownFunction->Attributes();

        // Copy prototype
        if (const auto *NewPrototype = WellKnownFunction->prototype()) {
          TypeCopier &Copier = It->second.first->Copier;
          F.Prototype() = Copier.copyTypeInto(*NewPrototype);
        }
      }
    }

    for (auto &WLF : WellKnownModels)
      WLF->Copier.finalize();

    model::deduplicateCollidingNames(Model);

    return llvm::Error::success();
  }
};

} // namespace revng::pipes

static pipeline::RegisterAnalysis<revng::pipes::ImportWellKnownModelsAnalysis>
  E;
