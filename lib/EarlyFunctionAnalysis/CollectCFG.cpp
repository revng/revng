/// \file CollectCFG.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/BasicAnalyses/GeneratedCodeBasicInfo.h"
#include "revng/EarlyFunctionAnalysis/CFGAnalyzer.h"
#include "revng/EarlyFunctionAnalysis/CFGStringMap.h"
#include "revng/EarlyFunctionAnalysis/FunctionMetadata.h"
#include "revng/EarlyFunctionAnalysis/FunctionSummaryOracle.h"
#include "revng/Model/Binary.h"
#include "revng/Pipeline/Contract.h"
#include "revng/Pipeline/RegisterContainerFactory.h"
#include "revng/Pipeline/RegisterPipe.h"
#include "revng/Pipes/Kinds.h"
#include "revng/Pipes/ModelGlobal.h"
#include "revng/Pipes/StringMap.h"
#include "revng/Support/CommonOptions.h"
#include "revng/Support/YAMLTraits.h"

using namespace llvm;

namespace revng::pipes {

class CollectCFGPipe {
public:
  static constexpr const auto Name = "collect-cfg";

public:
  std::array<pipeline::ContractGroup, 1> getContract() const {
    using namespace pipeline;
    using namespace ::revng::kinds;
    return { pipeline::ContractGroup(kinds::Root,
                                     0,
                                     kinds::CFG,
                                     1,
                                     pipeline::InputPreservation::Preserve) };
  }

public:
  void run(pipeline::ExecutionContext &Context,
           pipeline::LLVMContainer &Module,
           CFGMap &CFGs) {
    const auto &Binary = getModelFromContext(Context);
    llvm::Module &M = Module.getModule();
    using FSOracle = efa::FunctionSummaryOracle;

    // Collect GCBI
    GeneratedCodeBasicInfo GCBI(*Binary);
    GCBI.run(M);

    FSOracle Oracle = FSOracle::importBasicPrototypeData(M, GCBI, *Binary);
    efa::CFGAnalyzer Analyzer(M, GCBI, Binary, Oracle);

    pipeline::TargetsList
      RequestedTargets = Context.getCurrentRequestedTargets()[CFGs.name()];

    for (const pipeline::Target &Target : RequestedTargets) {
      if (&Target.getKind() != &revng::kinds::CFG)
        continue;
      Context.getContext().pushReadFields();

      auto EntryAddress = MetaAddress::fromString(Target
                                                    .getPathComponents()[0]);

      // Recover the control-flow graph of the function
      efa::FunctionMetadata New;
      New.Entry() = EntryAddress;
      New.ControlFlowGraph() = std::move(Analyzer.analyze(EntryAddress).CFG);

      if (DebugNames) {
        auto Function = Binary->Functions().at(EntryAddress);
        New.OriginalName() = Function.OriginalName();
      }

      revng_assert(New.ControlFlowGraph().contains(BasicBlockID(New.Entry())));

      // Run final steps on the CFG
      New.simplify(*Binary);

      revng_assert(New.ControlFlowGraph().contains(BasicBlockID(New.Entry())));

      // TODO: we'd need a function-wise TupleTreeContainer
      CFGs[EntryAddress] = serializeToString(New);

      // Commit the produced target
      Context.commit(Target, CFGs.name());

      Context.getContext().popReadFields();
    }
  }

  void print(const pipeline::Context &Ctx,
             llvm::raw_ostream &OS,
             llvm::ArrayRef<std::string> ContainerNames) const {}

  llvm::Error checkPrecondition(const pipeline::Context &Ctx) const {
    return llvm::Error::success();
  }
};

static pipeline::RegisterPipe<CollectCFGPipe> X;

} // namespace revng::pipes

static pipeline::RegisterDefaultConstructibleContainer<revng::pipes::CFGMap> X2;
