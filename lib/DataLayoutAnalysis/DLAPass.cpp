//
// Copyright (c) rev.ng Labs Srl. See LICENSE.md for details.
//

#include "revng/Model/LoadModelPass.h"
#include "revng/Pipeline/Context.h"
#include "revng/Pipeline/LLVMContainer.h"
#include "revng/Pipeline/RegisterAnalysis.h"
#include "revng/Pipes/Kinds.h"
#include "revng/Pipes/ModelGlobal.h"

#include "revng-c/DataLayoutAnalysis/DLALayouts.h"
#include "revng-c/DataLayoutAnalysis/DLAPass.h"
#include "revng-c/Pipes/Kinds.h"

#include "Backend/DLAMakeModelTypes.h"
#include "Frontend/DLATypeSystemBuilder.h"
#include "Middleend/DLAStep.h"

char DLAPass::ID = 0;

static Logger<> BuilderLog("dla-builder-log");

using Register = llvm::RegisterPass<DLAPass>;
static ::Register X("dla", "Data Layout Analysis Pass", false, false);

void DLAPass::getAnalysisUsage(llvm::AnalysisUsage &AU) const {
  AU.addRequired<LoadModelWrapperPass>();
  AU.addRequired<llvm::LoopInfoWrapperPass>();
  AU.addRequired<llvm::ScalarEvolutionWrapperPass>();

  AU.setPreservesAll();
}

bool DLAPass::runOnModule(llvm::Module &M) {
  auto &ModelWrapper = getAnalysis<LoadModelWrapperPass>().get();

  // Front-end: Create the LayoutTypeSystem graph from an LLVM module
  dla::LayoutTypeSystem TS;
  dla::DLATypeSystemLLVMBuilder Builder{ TS };
  const model::Binary &Model = *ModelWrapper.getReadOnlyModel();
  Builder.buildFromLLVMModule(M, this, Model);

  if (BuilderLog.isEnabled())
    Builder.dumpValuesMapping("DLA-values-initial.csv");

  // Middle-end Steps: manipulate nodes and edges of the DLATypeSystem graph
  dla::StepManager SM;
  size_t PtrSize = getPointerSize(Model.Architecture);
  revng_check(SM.addStep<dla::RemoveInvalidPointers>(PtrSize));
  revng_check(SM.addStep<dla::CollapseEqualitySCC>());
  revng_check(SM.addStep<dla::CollapseInstanceAtOffset0SCC>());
  revng_check(SM.addStep<dla::SimplifyInstanceAtOffset0>());
  revng_check(SM.addStep<dla::PruneLayoutNodesWithoutLayout>());
  revng_check(SM.addStep<dla::ComputeUpperMemberAccesses>());
  revng_check(SM.addStep<dla::CollapseCompatibleArrays>());
  revng_check(SM.addStep<dla::RemoveInvalidStrideEdges>());
  revng_check(SM.addStep<dla::PruneLayoutNodesWithoutLayout>());
  revng_check(SM.addStep<dla::ComputeUpperMemberAccesses>());
  revng_check(SM.addStep<dla::ComputeNonInterferingComponents>());
  revng_check(SM.addStep<dla::DeduplicateUnionFields>());
  revng_check(SM.addStep<dla::ComputeNonInterferingComponents>());
  SM.run(TS);

  // Compress the equivalence classes obtained after graph manipulation
  dla::VectEqClasses &EqClasses = TS.getEqClasses();
  EqClasses.compress();
  dla::LayoutTypePtrVect Values = Builder.getValues();

  if (BuilderLog.isEnabled())
    Builder.dumpValuesMapping("DLA-values-after-ME.csv");

  // Generate model types
  auto &WritableModel = ModelWrapper.getWriteableModel();
  auto ValueToTypeMap = dla::makeModelTypes(TS, Values, WritableModel);
  bool Changed = dla::updateFuncSignatures(M, WritableModel, ValueToTypeMap);

  return Changed;
}

class DLAAnalysis {
public:
  static constexpr auto Name = "dla";

  std::vector<std::vector<pipeline::Kind *>> AcceptedKinds = {
    { &revng::pipes::StackAccessesSegregated }
  };

  void run(pipeline::Context &Ctx, pipeline::LLVMContainer &Module) {
    using namespace revng::pipes;

    llvm::legacy::PassManager Manager;
    auto Global = llvm::cantFail(Ctx.getGlobal<ModelGlobal>(ModelGlobalName));
    Manager.add(new LoadModelWrapperPass(ModelWrapper(Global->get())));
    Manager.add(new DLAPass());
    Manager.run(Module.getModule());
  }
};

pipeline::RegisterAnalysis<DLAAnalysis> DLCPipelineReg;
