//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Support/Progress.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "revng/Clift/Clift.h"
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

[[noreturn]] static void configurationError(const llvm::Twine &Message) {
  revng_abort(("mlir-pipe: " + Message).str().c_str());
}

static constexpr llvm::StringRef
  ModuleName = mlir::ModuleOp::getOperationName();
static constexpr llvm::StringRef
  FunctionName = clift::FunctionOp::getOperationName();

namespace revng::pypeline::pipes {

PureMLIRPassesPipe::PureMLIRPassesPipe(llvm::StringRef StaticConfiguration) :
  StaticConfiguration(StaticConfiguration) {
  Configuration Configuration = Configuration::parse(StaticConfiguration);
  for (auto &PassData : Configuration.Passes) {
    llvm::StringRef PassName = PassData.Name;
    const mlir::PassInfo *PassInfo = mlir::Pass::lookupPassInfo(PassName);
    if (PassInfo == nullptr)
      configurationError("requested pass " + PassName + " was not found");

    // Passes that can run on any operation run on each function.
    llvm::StringRef OpName = PassInfo->getOpName().value_or(FunctionName);
    if (OpName != ModuleName and OpName != FunctionName)
      configurationError("pass " + PassName + " runs on " + OpName
                         + ", but only " + ModuleName + " and " + FunctionName
                         + " are supported");

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
  llvm::Task T(Outgoing[0].size(), TaskName);
  mlir::PassManager PM(Container.getContext(), ModuleName);
  auto ErrorHandler = [&](const llvm::Twine &Msg) {
    emitError(mlir::UnknownLoc::get(PM.getContext())) << Msg;
    return mlir::failure();
  };

  // Module passes go in the top-level pass manager. Each run of consecutive
  // function passes goes in its own nested pass manager. The constructor has
  // already rejected passes restricted to anything else, so every pass that is
  // not a module pass is a function pass.
  mlir::OpPassManager *FunctionPM = nullptr;
  for (const PassInfo &PassInfo : PassInfos) {
    mlir::OpPassManager *Target = &PM;
    if (PassInfo.PassInfo->getOpName() == ModuleName) {
      FunctionPM = nullptr;
    } else {
      if (FunctionPM == nullptr)
        FunctionPM = &PM.nest<clift::FunctionOp>();
      Target = FunctionPM;
    }

    mlir::LogicalResult R = PassInfo.PassInfo->addToPipeline(*Target,
                                                             PassInfo.Options,
                                                             ErrorHandler);
    revng_check(R.succeeded());
  }

  for (const ObjectID *Object : Outgoing[0]) {
    revng_assert(Object->kind() == Kinds::Function);
    const MetaAddress &Key = std::get<MetaAddress>(Object->key());
    T.advance(Key.toString(), true);
    mlir::LogicalResult R = PM.run(Container.getModule(*Object));
    revng_check(R.succeeded());
  }

  return { {}, {} };
}

} // namespace revng::pypeline::pipes
