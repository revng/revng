//
// Copyright (c) rev.ng Srls. See LICENSE.md for details.
//

#include "revng/Support/Assert.h"
#include "revng/Support/Debug.h"

#include "DLAStep.h"

namespace dla {

const char CollapseEqualitySCC::ID = 0;
const char CollapseInstanceAtOffset0SCC::ID = 0;
const char CollapseInheritanceSCC::ID = 0;
const char CollapseCompatibleArrays::ID = 0;
const char CollapseSingleChild::ID = 0;
const char ComputeNonInterferingComponents::ID = 0;
const char ComputeUpperMemberAccesses::ID = 0;
const char DeduplicateUnionFields::ID = 0;
const char MakeInheritanceTree::ID = 0;
const char PropagateInheritanceToAccessors::ID = 0;
const char PruneLayoutNodesWithoutLayout::ID = 0;
const char RemoveConflictingEdges::ID = 0;
const char RemoveInvalidStrideEdges::ID = 0;
const char RemoveTransitiveInheritanceEdges::ID = 0;

static Logger<> DLAStepManagerLog("dla-step-manager");

[[nodiscard]] bool StepManager::addStep(std::unique_ptr<Step> S) {
  const void *StepID = S->getStepID();

  if (Step::IDSetConstRef D = S->getDependencies(); not D.empty()) {

    if (D.count(StepID)) {
      revng_log(DLAStepManagerLog,
                "Step " << StepID << " has a dependency from itself");
      return false;
    }

    const auto IsNotValid = [this](const void *DepID) {
      return InvalidatedSteps.count(DepID) or not InsertedSteps.count(DepID);
    };

    auto DepIt = std::find_if(D.begin(), D.end(), IsNotValid);
    if (DepIt != D.end()) {
      revng_log(DLAStepManagerLog,
                "Step " << StepID << " depends Step " << *DepIt
                        << " which is invalidated or was never inserted");
      return false;
    }
  }

  if (Step::IDSetConstRef I = S->getInvalidated(); not I.empty()) {

    if (I.count(StepID)) {
      revng_log(DLAStepManagerLog, "Step " << StepID << " invalidates itself");
      return false;
    }

    const auto Invalidate = [this](const void *InvalidatedID) {
      if (InsertedSteps.count(InvalidatedID)) {
        InsertedSteps.erase(InvalidatedID);
        InvalidatedSteps.insert(InvalidatedID);
      }
    };
    std::for_each(I.begin(), I.end(), Invalidate);
  }

  InsertedSteps.insert(StepID);
  InvalidatedSteps.erase(StepID);
  Schedule.push_back(std::move(S));
  return true;
}

void StepManager::run(LayoutTypeSystem &TS) {
  if (not hasValidSchedule())
    revng_abort("Cannot run a on LayoutTypeSystem: invalid schedule");
  for (auto &S : Schedule)
    S->runOnTypeSystem(TS);
}

} // end namespace dla
