//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <ranges>
#include <set>

#include "revng/DataLayoutAnalysis/DLA.h"
#include "revng/DataLayoutAnalysis/DLALayouts.h"

#include "Backend/DLAMakeModelTypes.h"
#include "Frontend/DLATypeSystemBuilder.h"
#include "Middleend/DLAStep.h"

static Logger BuilderLog("dla-builder-log");

namespace revng::pypeline::analyses {

llvm::Error AnalyzeDataLayout::run(Model &Model,
                                   const Request &Incoming,
                                   llvm::StringRef Configuration,
                                   LLVMFunctionContainer &ModuleContainer) {
  // Run DLA directly over the requested per-function modules, without linking
  // them into a temporary root module. The modules are visited lazily through
  // a transform view and stay valid in the container afterwards.
  auto Modules = Incoming[0]
                 | std::views::transform([&](const ObjectID *Object)
                                           -> llvm::Module * {
                     return &ModuleContainer.getModule(*Object);
                   });

  llvm::Task T(3, "runDataLayoutAnalysis");
  T.advance("DLA Frontend");

  // Front-end: Create the LayoutTypeSystem graph from the LLVM modules
  dla::LayoutTypeSystem TS;
  dla::DLATypeSystemLLVMBuilder Builder{ TS, *Model.get() };
  Builder.buildFromLLVMModules(Modules);

  if (BuilderLog.isEnabled())
    Builder.dumpValuesMapping("DLA-values-initial.csv");

  // Middle-end Steps: manipulate nodes and edges of the DLATypeSystem graph
  T.advance("DLA Middleend");
  dla::StepManager SM;
  size_t PtrSize = getPointerSize(Model.get()->Architecture());

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
  auto ValueToTypeMap = dla::makeModelTypes(TS, Values, Model.get());

  std::set<MetaAddress> UpdatedSegments;
  for (llvm::Module *M : Modules) {
    dla::updateFuncSignatures(*M, Model.get(), ValueToTypeMap);
    dla::updateSegmentsTypes(*M, Model.get(), ValueToTypeMap, UpdatedSegments);
  }
  revng_assert(Model.get()->verify(true));

  return llvm::Error::success();
}

} // namespace revng::pypeline::analyses
