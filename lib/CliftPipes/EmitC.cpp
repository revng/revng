//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <unordered_map>

#include "revng/Clift/Helpers.h"
#include "revng/CliftEmitC/CBackend.h"
#include "revng/CliftEmitC/CSemantics.h"
#include "revng/CliftEmitC/Configuration.h"
#include "revng/CliftImportModel/Verify.h"
#include "revng/CliftPipes/CliftContainer.h"
#include "revng/CliftPipes/Configuration.h"
#include "revng/CliftPipes/EmitC.h"
#include "revng/Pipeline/RegisterPipe.h"
#include "revng/Pipes/Containers.h"
#include "revng/Pipes/Kinds.h"

using namespace revng;

namespace {

class CBackendPipe {
public:
  static constexpr auto Name = "emit-c";

  std::array<pipeline::ContractGroup, 1> getContract() const {
    using namespace pipeline;
    using namespace kinds;

    return { ContractGroup({ Contract(CliftFunction,
                                      0,
                                      Decompiled,
                                      1,
                                      InputPreservation::Preserve) }) };
  }

  void run(pipeline::ExecutionContext &EC,
           const pipes::CliftFunctionContainer &CliftFunctionContainer,
           pipes::DecompileStringMap &DecompiledFunctionsContainer) {
    mlir::ModuleOp Module = CliftFunctionContainer.getModule();

    const auto &Model = *revng::getModelFromContext(EC);
    revng_assert(verifyCSemantics(Module).succeeded());

    std::unordered_map<MetaAddress, clift::FunctionOp> Functions;
    Module->walk([&](clift::FunctionOp F) {
      MetaAddress MA = getMetaAddress(F);
      if (MA.isValid()) {
        auto [Iterator, Inserted] = Functions.try_emplace(MA, F);
        revng_assert(Inserted);
      }
    });

    for (const model::Function &Function :
         getFunctionsAndCommit(EC, DecompiledFunctionsContainer.name())) {
      auto It = Functions.find(Function.Entry());
      revng_check(It != Functions.end()
                  and "Requested Clift function not found");

      std::string Code;
      {
        llvm::raw_string_ostream OS(Code);
        ptml::CTokenEmitter Emitter(OS, ptml::Tagging::Enabled);
        decompile(It->second, Emitter);
      }

      DecompiledFunctionsContainer.insert_or_assign(Function.Entry(),
                                                    std::move(Code));
    }
  }
};

static pipeline::RegisterPipe<CBackendPipe> X;

} // namespace

namespace revng::pypeline::piperuns {

EmitC::EmitC(const Model &Model,
             llvm::StringRef Configuration,
             llvm::StringRef DynamicConfig,
             CliftFunctionContainer &Input,
             PTMLCFunctionBytesContainer &Output) :
  Input(Input),
  Output(Output),
  Configuration(parseCEmissionPipeConfiguration(Configuration)) {
}

void EmitC::runOnFunction(const model::Function &Function) {
  using namespace clift;

  ObjectID Object(Function.Entry());

  mlir::ModuleOp Module = Input.getModule(Object);

  revng_assert(verifyCSemantics(Module).succeeded());
  FunctionOp MLIRFunction = getUniqueIsolatedFunction(Module, Function.Entry());

  // TODO: once we emit any type definitions, in the decompiled code, we should
  //       carry a `TypeEmitterConfiguration` set from `Options` from here
  //       all the way to wherever the TypeDefinitionEmitter is constructed.
  TypeEmitterConfiguration TEConfiguration = {
    .TypeToOmit = {},
    .EmitMaximumEnumValue = false,
    .ExplicitPadding = true,
  };

  switch (Configuration.Mode) {
  case EmissionMode::Editable:
    TEConfiguration.EmitMaximumEnumValue = true;
    TEConfiguration.ExplicitPadding = false;
    break;

  case EmissionMode::Recompilable:
    TEConfiguration.EmitMaximumEnumValue = false;
    TEConfiguration.ExplicitPadding = true;
    break;

  default:
    revng_abort("Unsupported emission style.");
  };

  auto OS = Output.getOStream(Object);
  ptml::CTokenEmitter Emitter(*OS,
                              Configuration.DisableMarkup ?
                                ptml::Tagging::Disabled :
                                ptml::Tagging::Enabled);
  decompile(MLIRFunction, Emitter);
}

} // namespace revng::pypeline::piperuns
