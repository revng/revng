//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Support/Progress.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "revng/Clift/Helpers.h"
#include "revng/CliftTransforms/Passes.h"
#include "revng/Pipebox/MLIRPipe.h"

static bool Initialized = ([]() {
  mlir::registerTransformsPasses();
  clift::registerCliftPasses();
  return true;
})();

namespace rpp = revng::pypeline::pipes;
using Configuration = rpp::PureMLIRPassesPipe::Configuration;

struct PassConfigurationStruct {
  std::string Name;
  std::string Options;
};

// There needs to be two types so that there is one for `PolymorphicTraits` and
// another one for `MappingTraits`.
struct PassConfiguration : PassConfigurationStruct {};

struct rpp::PureMLIRPassesPipe::Configuration {
  std::vector<PassConfiguration> Passes;
  static Configuration parse(llvm::StringRef Input);
};

template<>
struct llvm::yaml::PolymorphicTraits<PassConfiguration> {
  static NodeKind getKind(const PassConfiguration &Element) {
    return NodeKind::Map;
  }

  static std::string &getAsScalar(PassConfiguration &Element) {
    return Element.Name;
  }

  static PassConfigurationStruct &getAsMap(PassConfiguration &Element) {
    return Element;
  }

  static PassConfiguration &getAsSequence(PassConfiguration &Element) {
    revng_abort();
  }
};

template<>
struct llvm::yaml::MappingTraits<PassConfigurationStruct> {
  static void mapping(IO &IO, PassConfigurationStruct &Fields) {
    IO.mapRequired("name", Fields.Name);
    IO.mapRequired("options", Fields.Options);
  }
};

LLVM_YAML_IS_SEQUENCE_VECTOR(PassConfiguration);

template<>
struct llvm::yaml::MappingTraits<Configuration> {
  static void mapping(IO &IO, Configuration &Fields) {
    IO.mapRequired("passes", Fields.Passes);
  }
};

Configuration Configuration::parse(llvm::StringRef Input) {
  return llvm::cantFail(fromString<Configuration>(Input));
}

namespace revng::pypeline::pipes {

PureMLIRPassesPipe::PureMLIRPassesPipe(llvm::StringRef StaticConfiguration) :
  StaticConfiguration(StaticConfiguration) {
  Configuration Configuration = Configuration::parse(StaticConfiguration);
  for (auto &PassData : Configuration.Passes) {
    llvm::StringRef PassName = PassData.Name;
    const mlir::PassInfo *PassInfo = mlir::Pass::lookupPassInfo(PassName);
    revng_assert(PassInfo != nullptr,
                 ("mlir-pipe: Requested pass " + PassName.str()
                  + " was not found")
                   .c_str());
    PassInfos.push_back({ PassInfo, std::move(PassData.Options) });
  }

  auto NameView = Configuration.Passes
                  | std::views::transform([](PassConfiguration &PC)
                                            -> llvm::StringRef {
                      return PC.Name;
                    });
  TaskName = "PureMLIRPasses(" + llvm::join(NameView, ",") + ")";
}

PipeOutput PureMLIRPassesPipe::run(const Model &TheModel,
                                   const revng::pypeline::Request &Incoming,
                                   const revng::pypeline::Request &Outgoing,
                                   llvm::StringRef Configuration,
                                   CliftFunctionContainer &Container) {
  using namespace clift;
  llvm::Task T(Outgoing[0].size(), TaskName);
  mlir::PassManager PM(&Container.getContext(), FunctionOp::getOperationName());
  auto ErrorHandler = [&](const llvm::Twine &Msg) {
    emitError(mlir::UnknownLoc::get(PM.getContext())) << Msg;
    return mlir::failure();
  };
  for (const PassInfo &PassInfo : PassInfos) {
    mlir::LogicalResult R = PassInfo.PassInfo->addToPipeline(PM,
                                                             PassInfo.Options,
                                                             ErrorHandler);
    revng_check(R.succeeded());
  }

  for (const ObjectID *Object : Outgoing[0]) {
    revng_assert(Object->kind() == Kinds::Function);
    const MetaAddress &Key = std::get<MetaAddress>(Object->key());
    T.advance(Key.toString(), true);
    mlir::ModuleOp Module = Container.getModule(*Object);
    FunctionOp Function = getUniqueIsolatedFunction(Module, Key);
    mlir::LogicalResult R = PM.run(Function);
    revng_check(R.succeeded());
  }

  return { {}, {} };
}

} // namespace revng::pypeline::pipes
