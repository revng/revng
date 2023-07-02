/// \file WellKnownModels.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/Binary.h"
#include "revng/Model/Importer/TypeCopier.h"
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
  TupleTree<model::Binary> Model;
  TypeCopier Copier;

public:
  WellKnownModel(TupleTree<model::Binary> &&Model) :
    Model(Model), Copier(this->Model) {}
};

class ImportWellKnownModelsAnalysis {
public:
  static constexpr auto Name = "ImportWellKnownModels";

public:
  std::vector<std::vector<pipeline::Kind *>> AcceptedKinds;

public:
  llvm::Error run(pipeline::Context &Context) {
    std::vector<std::unique_ptr<WellKnownModel>> WellKnownModels;
    std::map<WellKnownFunctionKey,
             std::pair<WellKnownModel *, model::Function *>>
      WellKnownFunctions;

    // Load all well-known models
    for (const std::string &Path :
         revng::ResourceFinder.list("share/revng/well-known-models", ".yml")) {
      auto MaybeModel = TupleTree<model::Binary>::fromFile(Path);
      revng_assert(MaybeModel);
      using namespace std;
      auto NewWKM = make_unique<WellKnownModel>(std::move(*MaybeModel));
      WellKnownModels.push_back(std::move(NewWKM));
    }

    // Create an index of well-known functions
    for (auto &WellKnownModel : WellKnownModels) {
      auto &Model = WellKnownModel->Model;
      // Collect exported functions
      for (model::Function &F : Model->Functions()) {
        for (std::string ExportedName : F.ExportedNames()) {
          WellKnownFunctions[{
            Model->Architecture(), Model->DefaultABI(), ExportedName }] = {
            WellKnownModel.get(), &F
          };
        }
      }
    }

    TupleTree<model::Binary> &Model = getWritableModelFromContext(Context);

    for (model::DynamicFunction &F : Model->ImportedDynamicFunctions()) {
      // Compose the key
      WellKnownFunctionKey Key = { Model->Architecture(),
                                   Model->DefaultABI(),
                                   F.OriginalName() };

      // See if it's a well-known function
      auto It = WellKnownFunctions.find(Key);
      if (It != WellKnownFunctions.end()) {
        model::Function *WellKnownFunction = It->second.second;

        // Copy attributes
        F.Attributes() = WellKnownFunction->Attributes();

        // Copy prototype
        auto NewPrototype = WellKnownFunction->Prototype();
        if (NewPrototype.isValid()) {
          auto TypeKey = NewPrototype.get()->key();
          TypeCopier &Copier = It->second.first->Copier;
          Copier.copyTypeInto(NewPrototype, Model);
          F.Prototype() = Model->getTypePath(TypeKey);
        }
      }
    }

    return llvm::Error::success();
  }
};

} // namespace revng::pipes

static pipeline::RegisterAnalysis<revng::pipes::ImportWellKnownModelsAnalysis>
  E;
