/// \file CollectCFG.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Support/Error.h"

#include "revng/BasicAnalyses/GeneratedCodeBasicInfo.h"
#include "revng/EarlyFunctionAnalysis/CFGAnalyzer.h"
#include "revng/EarlyFunctionAnalysis/CFGStringMap.h"
#include "revng/EarlyFunctionAnalysis/ControlFlowGraph.h"
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
  static constexpr auto Name = "collect-cfg";

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
           pipeline::LLVMContainer &ModuleContainer,
           CFGMap &CFGs) {
    const auto &Binary = getModelFromContext(Context);
    llvm::Module &M = ModuleContainer.getModule();
    using FSOracle = efa::FunctionSummaryOracle;

    // Collect GCBI
    GeneratedCodeBasicInfo GCBI(*Binary);
    GCBI.run(M);

    FSOracle Oracle = FSOracle::importBasicPrototypeData(M, GCBI, *Binary);
    efa::CFGAnalyzer Analyzer(M, GCBI, Binary, Oracle);

    for (const model::Function &Function :
         getFunctionsAndCommit(Context, CFGs.name())) {
      MetaAddress EntryAddress = Function.Entry();

      // Recover the control-flow graph of the function
      efa::ControlFlowGraph New;
      New.Entry() = EntryAddress;
      New.Blocks() = std::move(Analyzer.analyze(EntryAddress).CFG);

      if (DebugNames) {
        auto Function = Binary->Functions().at(EntryAddress);
        New.OriginalName() = Function.OriginalName();
      }

      revng_assert(New.Blocks().contains(BasicBlockID(New.Entry())));

      // Run final steps on the CFG
      New.simplify(*Binary);

      revng_assert(New.Blocks().contains(BasicBlockID(New.Entry())));

      // TODO: we'd need a function-wise TupleTreeContainer
      CFGs[EntryAddress] = serializeToString(New);
    }
  }

  llvm::Error checkPrecondition(const pipeline::Context &Ctx) const {
    return llvm::Error::success();
  }
};

static pipeline::RegisterPipe<CollectCFGPipe> X;

} // namespace revng::pipes

static pipeline::RegisterDefaultConstructibleContainer<revng::pipes::CFGMap> X2;
