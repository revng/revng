//
// Copyright (c) rev.ng Srls. See LICENSE.md for details.
//

#include "revng/Model/LoadModelPass.h"

#include "revng-c/Decompiler/DLALayouts.h"
#include "revng-c/Decompiler/DLAPass.h"

#include "DLAStep.h"
#include "DLATypeSystem.h"
#include "DLATypeSystemBuilder.h"

char DLAPass::ID = 0;

static Logger<> BuilderLog("dla-builder-log");

using Register = llvm::RegisterPass<DLAPass>;
static Register X("dla", "Data Layout Analysis Pass", false, false);

void DLAPass::getAnalysisUsage(llvm::AnalysisUsage &AU) const {
  AU.addRequired<LoadModelWrapperPass>();
  AU.addRequired<llvm::LoopInfoWrapperPass>();
  AU.addRequired<llvm::ScalarEvolutionWrapperPass>();
  AU.setPreservesAll();
}

bool DLAPass::runOnModule(llvm::Module &M) {
  dla::LayoutTypeSystem TS;

  // Front-end: Create the LayoutTypeSystem graph from an LLVM module
  dla::DLATypeSystemLLVMBuilder Builder{ TS };
  Builder.buildFromLLVMModule(M, this);

  if (BuilderLog.isEnabled())
    Builder.dumpValuesMapping("DLA-values-initial.csv");

  // Middle-end Steps: manipulate nodes and edges of the DLATypeSystem graph
  dla::StepManager SM;
  revng_check(SM.addStep<dla::CollapseIdentityAndInheritanceCC>());
  revng_check(SM.addStep<dla::PropagateInheritanceToAccessors>());
  revng_check(SM.addStep<dla::RemoveTransitiveInheritanceEdges>());
  revng_check(SM.addStep<dla::MakeInheritanceTree>());
  revng_check(SM.addStep<dla::PruneLayoutNodesWithoutLayout>());
  revng_check(SM.addStep<dla::ComputeUpperMemberAccesses>());
  revng_check(SM.addStep<dla::CollapseCompatibleArrays>());
  revng_check(SM.addStep<dla::ComputeNonInterferingComponents>());

  SM.run(TS);

  if (BuilderLog.isEnabled())
    Builder.dumpValuesMapping("DLA-values-after-ME.csv");

  // Compress the equivalence classes obtained after graph manipulation
  dla::VectEqClasses &EqClasses = TS.getEqClasses();
  EqClasses.compress();

  // Create Layouts from the final nodes of the graph
  dla::LayoutPtrVector OrderedLayouts = makeLayouts(TS, this->Layouts);

  // Map Layouts back to their corresponding LayoutTypePtr
  dla::LayoutTypePtrVect Values = Builder.getValues();
  this->ValueLayoutsMap = makeLayoutMap(Values, OrderedLayouts, EqClasses);

  return true;
}
