//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/DataLayoutAnalysis/DLA.h"
#include "revng/DataLayoutAnalysis/DLALayouts.h"
#include "revng/DataLayoutAnalysis/DLAPass.h"
#include "revng/Model/LoadModelPass.h"
#include "revng/Pipeline/Context.h"
#include "revng/Pipeline/LLVMContainer.h"
#include "revng/Pipeline/RegisterAnalysis.h"
#include "revng/Pipes/Kinds.h"
#include "revng/Pipes/ModelGlobal.h"

#include "Backend/DLAMakeModelTypes.h"
#include "Frontend/DLATypeSystemBuilder.h"
#include "Middleend/DLAStep.h"

char DLAPass::ID = 0;

static Logger BuilderLog("dla-builder-log");

using Register = llvm::RegisterPass<DLAPass>;
static ::Register X("dla", "Data Layout Analysis Pass", false, false);

void DLAPass::getAnalysisUsage(llvm::AnalysisUsage &AU) const {
  if (ConstructorModel == nullptr)
    AU.addRequired<LoadModelWrapperPass>();
  AU.addRequired<llvm::ScalarEvolutionWrapperPass>();

  AU.setPreservesAll();
}

bool DLAPass::runOnModule(llvm::Module &M) {
  TupleTree<model::Binary> *WritableModel = nullptr;
  if (ConstructorModel != nullptr) {
    WritableModel = ConstructorModel;
  } else {
    WritableModel = &getAnalysis<LoadModelWrapperPass>()
                       .get()
                       .getWriteableModel();
  }

  llvm::Task T(3, "DLAPass::runOnModule");
  T.advance("DLA Frontend");

  // Front-end: Create the LayoutTypeSystem graph from an LLVM module
  dla::LayoutTypeSystem TS;
  const model::Binary &Model = *WritableModel->get();
  dla::DLATypeSystemLLVMBuilder Builder{ TS, Model };
  Builder.buildFromLLVMModule(M, this);

  if (BuilderLog.isEnabled())
    Builder.dumpValuesMapping("DLA-values-initial.csv");

  // Middle-end Steps: manipulate nodes and edges of the DLATypeSystem graph
  T.advance("DLA Middleend");
  dla::StepManager SM;
  size_t PtrSize = getPointerSize(Model.Architecture());

  //
  // Graph normalization phase
  //
  revng_check(SM.addStep<dla::RemoveInvalidPointers>(PtrSize));
  revng_check(SM.addStep<dla::CollapseEqualitySCC>());
  revng_check(SM.addStep<dla::CollapseInstanceAtOffset0SCC>());
  revng_check(SM.addStep<dla::SimplifyInstanceAtOffset0>());
  revng_check(SM.addStep<dla::PruneLayoutNodesWithoutLayout>());
  revng_check(SM.addStep<dla::ComputeUpperMemberAccesses>());
  revng_check(SM.addStep<dla::RemoveInvalidStrideEdges>());
  revng_check(SM.addStep<dla::PruneLayoutNodesWithoutLayout>());
  revng_check(SM.addStep<dla::ComputeUpperMemberAccesses>());
  revng_check(SM.addStep<dla::DecomposeStridedEdges>());

  //
  // Graph optimization phase
  //
  revng_check(SM.addStep<dla::CollapseSingleChild>());
  revng_check(SM.addStep<dla::DeduplicateFields>());
  revng_check(SM.addStep<dla::MergePointeesOfPointerUnion>(PtrSize));
  revng_check(SM.addStep<dla::MergePointerNodes>());
  revng_check(SM.addStep<dla::CollapseInstanceAtOffset0SCC>());
  revng_check(SM.addStep<dla::SimplifyInstanceAtOffset0>());
  revng_check(SM.addStep<dla::PruneLayoutNodesWithoutLayout>());
  revng_check(SM.addStep<dla::ComputeUpperMemberAccesses>());
  revng_check(SM.addStep<dla::RemoveInvalidStrideEdges>());
  revng_check(SM.addStep<dla::PruneLayoutNodesWithoutLayout>());
  revng_check(SM.addStep<dla::ComputeUpperMemberAccesses>());

  revng_check(SM.addStep<dla::MergePointerNodes>());
  // CollapseSingleChild and DeduplicateFields run before
  // CompactCompatibleArrays and ArrangeAccessesHierarchically, to allow them to
  // produce better results
  revng_check(SM.addStep<dla::CollapseSingleChild>());
  revng_check(SM.addStep<dla::DeduplicateFields>());
  revng_check(SM.addStep<dla::ArrangeAccessesHierarchically>());
  revng_check(SM.addStep<dla::CompactCompatibleArrays>());
  revng_check(SM.addStep<dla::PushDownPointers>());
  // ArrangeAccessesHierarchically can move pointer edges around in some cases,
  // so we want to run MergePointerNodes again afterwards.
  revng_check(SM.addStep<dla::MergePointerNodes>());
  // CollapseSingleChild and DeduplicateFields run again after
  // CompactCompatibleArrays and ArrangeAccessesHierarchically, to allow them to
  // improve the results even further.
  revng_check(SM.addStep<dla::ResolveLeafUnions>());
  revng_check(SM.addStep<dla::CollapseSingleChild>());
  revng_check(SM.addStep<dla::DeduplicateFields>());
  revng_check(SM.addStep<dla::ComputeNonInterferingComponents>());
  SM.run(TS);

  // Compress the equivalence classes obtained after graph manipulation
  dla::VectEqClasses &EqClasses = TS.getEqClasses();
  EqClasses.compress();
  dla::LayoutTypePtrVect Values = Builder.getValues();

  if (BuilderLog.isEnabled())
    Builder.dumpValuesMapping("DLA-values-after-ME.csv");

  T.advance("DLA Backend");

  // Generate model types
  auto ValueToTypeMap = dla::makeModelTypes(TS, Values, *WritableModel);
  bool Changed = false;

  Changed |= dla::updateFuncSignatures(M, *WritableModel, ValueToTypeMap);
  Changed |= dla::updateSegmentsTypes(M, *WritableModel, ValueToTypeMap);
  revng_assert((*WritableModel)->verify(true));

  return Changed;
}

class DLAAnalysis {
public:
  static constexpr auto Name = "analyze-data-layout";

  std::vector<std::vector<pipeline::Kind *>> AcceptedKinds = {
    { &revng::kinds::StackAccessesSegregated }
  };

  void run(pipeline::ExecutionContext &EC, pipeline::LLVMContainer &Module) {
    using namespace revng;

    llvm::legacy::PassManager Manager;
    auto &Global = getWritableModelFromContext(EC);
    Manager.add(new LoadModelWrapperPass(ModelWrapper(Global)));
    Manager.add(new DLAPass());
    Manager.run(Module.getModule());
  }
};

pipeline::RegisterAnalysis<DLAAnalysis> DLCPipelineReg;

namespace revng::pypeline::analyses {

llvm::Error AnalyzeDataLayout::run(Model &Model,
                                   const Request &Incoming,
                                   llvm::StringRef Configuration,
                                   LLVMFunctionContainer &ModuleContainer) {
  auto RootModule = std::make_unique<llvm::Module>("root",
                                                   ModuleContainer
                                                     .getContext());
  for (const ObjectID *Object : Incoming[0])
    linkFunctionModules(ModuleContainer.takeModule(*Object), RootModule);

  llvm::legacy::PassManager Manager;
  Manager.add(new DLAPass(Model.get()));
  Manager.run(*RootModule);

  return llvm::Error::success();
}

} // namespace revng::pypeline::analyses
