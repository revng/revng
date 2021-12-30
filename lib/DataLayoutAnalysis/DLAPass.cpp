//
// Copyright (c) rev.ng Srls. See LICENSE.md for details.
//

#include "revng/Model/LoadModelPass.h"

#include "revng-c/DataLayoutAnalysis/DLALayouts.h"
#include "revng-c/DataLayoutAnalysis/DLAPass.h"

#include "Backend/DLAMakeLayouts.h"
#include "Backend/DLAMakeModelTypes.h"
#include "Frontend/DLATypeSystemBuilder.h"
#include "Middleend/DLAStep.h"

namespace {

llvm::cl::opt<std::string> DLADir("dla-flatc-dir",
                                  llvm::cl::desc("Path to serialize flatc"),
                                  llvm::cl::value_desc("Path"),
                                  llvm::cl::cat(MainCategory),
                                  llvm::cl::NumOccurrencesFlag::Optional);
} // end of unnamed namespace

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
  Builder.buildFromLLVMModule(M, this, ModelWrapper.getReadOnlyModel());

  if (BuilderLog.isEnabled())
    Builder.dumpValuesMapping("DLA-values-initial.csv");

  // Middle-end Steps: manipulate nodes and edges of the DLATypeSystem graph
  dla::StepManager SM;
  revng_check(SM.addStep<dla::CollapseIdentityAndInheritanceCC>());
  revng_check(SM.addStep<dla::PropagateInheritanceToAccessors>());
  revng_check(SM.addStep<dla::RemoveTransitiveInheritanceEdges>());
  revng_check(SM.addStep<dla::MakeInheritanceTree>());
  revng_check(SM.addStep<dla::PruneLayoutNodesWithoutLayout>());
  revng_check(SM.addStep<dla::RemoveConflictingEdges>());
  revng_check(SM.addStep<dla::CollapseSingleChild>());
  revng_check(SM.addStep<dla::ComputeUpperMemberAccesses>());
  revng_check(SM.addStep<dla::CollapseCompatibleArrays>());
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
  dla::updateFuncSignatures(M, WritableModel, ValueToTypeMap);

  // Generate Layouts
  dla::LayoutPtrVector OrderedLayouts = makeLayouts(TS, this->Layouts);
  this->ValueLayoutsMap = makeLayoutMap(Values, OrderedLayouts, EqClasses);

  return true;
}
