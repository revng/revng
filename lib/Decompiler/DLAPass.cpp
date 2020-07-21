//
// Copyright (c) rev.ng Srls. See LICENSE.md for details.
//

#include "revng-c/Decompiler/DLAPass.h"

#include "DLAStep.h"
#include "DLATypeSystem.h"

char DLAPass::ID = 0;

using Register = llvm::RegisterPass<DLAPass>;
static Register X("dla", "Data Layout Analysis Pass", false, false);

bool DLAPass::runOnModule(llvm::Module &M) {
  dla::StepManager SM;

  // Front-end Steps, that create initial nodes and edges
  revng_check(SM.addStep<dla::CreateInterproceduralTypes>());
  revng_check(SM.addStep<dla::CreateIntraproceduralTypes>(this));
  // Middle-end Steps, that manipulate nodes and edges
  revng_check(SM.addStep<dla::CollapseIdentityAndInheritanceCC>());
  revng_check(SM.addStep<dla::PropagateInheritanceToAccessors>());
  revng_check(SM.addStep<dla::RemoveTransitiveInheritanceEdges>());
  revng_check(SM.addStep<dla::MakeInheritanceTree>());
  revng_check(SM.addStep<dla::PruneLayoutNodesWithoutLayout>());
  revng_check(SM.addStep<dla::ComputeUpperMemberAccesses>());
  revng_check(SM.addStep<dla::CollapseCompatibleArrays>());
  // Back-end Steps, that build Layouts from LayoutTypeSystem nodes
  revng_check(SM.addStep<dla::ComputeNonInterferingComponents>());
  revng_check(SM.addStep<dla::MakeLayouts>(Layouts, ValueLayouts));

  dla::LayoutTypeSystem TS(M);

  SM.run(TS);

  return true;
}
