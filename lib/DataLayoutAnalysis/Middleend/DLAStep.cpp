//
// Copyright (c) rev.ng Labs Srl. See LICENSE.md for details.
//

#include "llvm/Support/Progress.h"

#include "revng/Support/Assert.h"
#include "revng/Support/Debug.h"

#include "DLAStep.h"

namespace dla {

const char ArrangeAccessesHierarchically::ID = 0;
const char CollapseEqualitySCC::ID = 0;
const char CollapseInstanceAtOffset0SCC::ID = 0;
const char CollapseSingleChild::ID = 0;
const char CompactCompatibleArrays::ID = 0;
const char ComputeNonInterferingComponents::ID = 0;
const char ComputeUpperMemberAccesses::ID = 0;
const char DecomposeStridedEdges::ID = 0;
const char DeduplicateFields::ID = 0;
const char MergePointerNodes::ID = 0;
const char MergePointeesOfPointerUnion::ID = 0;
const char PruneLayoutNodesWithoutLayout::ID = 0;
const char PushDownPointers::ID = 0;
const char RemoveInvalidStrideEdges::ID = 0;
const char RemoveInvalidPointers::ID = 0;
const char ResolveLeafUnions::ID = 0;
const char SimplifyInstanceAtOffset0::ID = 0;

static std::string getStepNameFromID(const void *ID) {
  if (ID == ArrangeAccessesHierarchically::getID())
    return "ArrangeAccessesHierarchically";
  else if (ID == CollapseEqualitySCC::getID())
    return "CollapseEqualitySCC";
  else if (ID == CollapseInstanceAtOffset0SCC::getID())
    return "CollapseInstanceAtOffset0SCC";
  else if (ID == CollapseSingleChild::getID())
    return "CollapseSingleChild";
  else if (ID == CompactCompatibleArrays::getID())
    return "CompactCompatibleArrays";
  else if (ID == ComputeNonInterferingComponents::getID())
    return "ComputeNonInterferingComponents";
  else if (ID == ComputeUpperMemberAccesses::getID())
    return "ComputeUpperMemberAccesses";
  else if (ID == DecomposeStridedEdges::getID())
    return "DecomposeStridedEdges";
  else if (ID == DeduplicateFields::getID())
    return "DeduplicateFields";
  else if (ID == MergePointeesOfPointerUnion::getID())
    return "MergePointeesOfPointerUnion";
  else if (ID == MergePointerNodes::getID())
    return "MergePointerNodes";
  else if (ID == PruneLayoutNodesWithoutLayout::getID())
    return "PruneLayoutNodesWithoutLayout";
  else if (ID == PushDownPointers::getID())
    return "PushDownPointers";
  else if (ID == RemoveInvalidStrideEdges::getID())
    return "RemoveInvalidStrideEdges";
  else if (ID == RemoveInvalidPointers::getID())
    return "RemoveInvalidPointers";
  else if (ID == ResolveLeafUnions::getID())
    return "ResolveLeafUnions";
  else if (ID == SimplifyInstanceAtOffset0::getID())
    return "SimplifyInstanceAtOffset0";
  else
    revng_abort("Unexpected ID for DLAStep");
}

static Logger<> DLAStepManagerLog("dla-step-manager");
static Logger<> DLADumpDot("dla-step-dump-dot");

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
  int x = 0;
  if (DLADumpDot.isEnabled())
    TS.dumpDotOnFile("type-system-0.dot", true);

  llvm::Task T{ Schedule.size(), "StepManager::run" };
  for (auto &S : Schedule) {
    T.advance(getStepNameFromID(S->getStepID()));
    S->runOnTypeSystem(TS);
    ++x;
    if (DLADumpDot.isEnabled()) {
      revng_log(DLADumpDot,
                "Step " << getStepNameFromID(S->getStepID())
                        << " Index: " << x);
      std::string DotName = "type-system-" + std::to_string(x) + ".dot";
      TS.dumpDotOnFile(DotName.c_str(), true);
    }
  }
}

} // end namespace dla
