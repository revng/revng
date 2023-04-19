/// \file DLAStepManager.cpp
/// Tests for dla::StepManager

//
// Copyright (c) rev.ng Labs Srl. See LICENSE.md for details.
//

#define BOOST_TEST_MODULE DLAStepManager
bool init_unit_test();
#include "boost/test/unit_test.hpp"

#include "llvm/ADT/SmallPtrSet.h"

#include "revng/Support/Assert.h"
#include "revng/UnitTestHelpers/UnitTestHelpers.h"

#include "lib/DataLayoutAnalysis/Middleend/DLAStep.h"

namespace dla {

class SelfDependentStep : public Step {
  static const char ID;

public:
  static const constexpr void *getID() { return &ID; }

  SelfDependentStep() : Step(ID, { SelfDependentStep::getID() }, {}) {}

  virtual ~SelfDependentStep() override = default;

  virtual bool runOnTypeSystem(LayoutTypeSystem &TS) override { return false; }
};

const char SelfDependentStep::ID = 0;

class SelfInvalidatingStep : public Step {
  static const char ID;

public:
  static const constexpr void *getID() { return &ID; }

  SelfInvalidatingStep() : Step(ID, {}, { SelfInvalidatingStep::getID() }) {}

  virtual ~SelfInvalidatingStep() override = default;

  virtual bool runOnTypeSystem(LayoutTypeSystem &TS) override { return false; }
};

const char SelfInvalidatingStep::ID = 0;

class StepWithNoDeps : public Step {
  static const char ID;

public:
  static const constexpr void *getID() { return &ID; }

  StepWithNoDeps() : Step(ID) {}

  virtual ~StepWithNoDeps() override = default;

  virtual bool runOnTypeSystem(LayoutTypeSystem &TS) override { return false; }
};

const char StepWithNoDeps::ID = 0;

class SingleDependencyStep : public Step {
  static const char ID;

public:
  static const constexpr void *getID() { return &ID; }

  SingleDependencyStep() : Step(ID, { StepWithNoDeps::getID() }, {}) {}

  virtual ~SingleDependencyStep() override = default;

  virtual bool runOnTypeSystem(LayoutTypeSystem &TS) override { return false; }
};

const char SingleDependencyStep::ID = 0;

class StepInvalidateNoDeps : public Step {
  static const char ID;

public:
  static const constexpr void *getID() { return &ID; }

  StepInvalidateNoDeps() : Step(ID, {}, { StepWithNoDeps::getID() }) {}

  virtual ~StepInvalidateNoDeps() override = default;

  virtual bool runOnTypeSystem(LayoutTypeSystem &TS) override { return false; }
};

const char StepInvalidateNoDeps::ID = 0;

} // end namespace dla

using namespace dla;

BOOST_AUTO_TEST_CASE(InitiallyEmpty) {
  StepManager SM;

  BOOST_TEST(SM.getNumSteps() == 0);
  BOOST_TEST(SM.hasValidSchedule());

  SM.reset();

  BOOST_TEST(SM.getNumSteps() == 0);
  BOOST_TEST(SM.hasValidSchedule());
}

BOOST_AUTO_TEST_CASE(InsertSelfDependency) {
  StepManager SM;

  BOOST_TEST(not SM.addStep(std::unique_ptr<Step>(new SelfDependentStep())));
  BOOST_TEST(SM.getNumSteps() == 0);
  // The schedule is still valid because the bad step is not inserted
  BOOST_TEST(SM.hasValidSchedule());
}

BOOST_AUTO_TEST_CASE(InsertSelfInvalidation) {
  StepManager SM;

  BOOST_TEST(not SM.addStep(std::unique_ptr<Step>(new SelfInvalidatingStep())));
  BOOST_TEST(SM.getNumSteps() == 0);
  // The schedule is still valid because the bad step is not inserted
  BOOST_TEST(SM.hasValidSchedule());
}

BOOST_AUTO_TEST_CASE(InsertStepWithNoDeps) {
  StepManager SM;

  BOOST_TEST(SM.addStep(std::unique_ptr<Step>(new StepWithNoDeps())));

  BOOST_TEST(SM.getNumSteps() == 1);
  BOOST_TEST(SM.hasValidSchedule());

  SM.reset();

  BOOST_TEST(SM.getNumSteps() == 0);
  BOOST_TEST(SM.hasValidSchedule());
}

BOOST_AUTO_TEST_CASE(InsertSameStepTwice) {
  StepManager SM;
  BOOST_TEST(SM.addStep(std::make_unique<StepWithNoDeps>()));

  BOOST_TEST(SM.getNumSteps() == 1);
  BOOST_TEST(SM.hasValidSchedule());

  BOOST_TEST(SM.addStep(std::make_unique<StepWithNoDeps>()));

  BOOST_TEST(SM.getNumSteps() == 2);
  BOOST_TEST(SM.hasValidSchedule());

  SM.reset();

  BOOST_TEST(SM.getNumSteps() == 0);
  BOOST_TEST(SM.hasValidSchedule());
}

BOOST_AUTO_TEST_CASE(InsertWithValidDependency) {
  StepManager SM;

  BOOST_TEST(SM.addStep(std::make_unique<StepWithNoDeps>()));

  BOOST_TEST(SM.getNumSteps() == 1);
  BOOST_TEST(SM.hasValidSchedule());

  BOOST_TEST(SM.addStep(std::make_unique<SingleDependencyStep>()));

  BOOST_TEST(SM.getNumSteps() == 2);
  BOOST_TEST(SM.hasValidSchedule());

  SM.reset();

  BOOST_TEST(SM.getNumSteps() == 0);
  BOOST_TEST(SM.hasValidSchedule());
}

BOOST_AUTO_TEST_CASE(InsertWithMissingDependency) {
  StepManager SM;

  BOOST_TEST(not SM.addStep(std::make_unique<SingleDependencyStep>()));
  BOOST_TEST(SM.getNumSteps() == 0);
  // This must be true because the step is not inserted
  BOOST_TEST(SM.hasValidSchedule());
}

BOOST_AUTO_TEST_CASE(SimpleInvalidate) {
  StepManager SM;
  // Initially the sched is empty
  BOOST_TEST(SM.sched_begin() == SM.sched_end());

  // Inserting a step with no dependencies should always succeed
  BOOST_TEST(SM.addStep(std::unique_ptr<Step>(new StepWithNoDeps())));
  BOOST_TEST(SM.getNumSteps() == 1);
  BOOST_TEST(SM.hasValidSchedule());

  // Inserting a step with a valid dependency that is already scheduled should
  // succeed and leave SM with a valid schedule
  BOOST_TEST(SM.addStep(std::unique_ptr<Step>(new SingleDependencyStep())));
  BOOST_TEST(SM.getNumSteps() == 2);
  BOOST_TEST(SM.hasValidSchedule());

  // Insert a step with no dependencies and that invalidates StepWithNoDeps.
  // This should succeed, while invalidating StepWithNoDeps.
  // If later no other step is added that depends on StepWithNoDeps, the
  // schedule should still be valid.
  BOOST_TEST(SM.addStep(std::unique_ptr<Step>(new StepInvalidateNoDeps())));
  BOOST_TEST(SM.getNumSteps() == 3);
  BOOST_TEST(SM.hasValidSchedule());

  // If we now add a step that depends on the invalidated step StepWithNoDeps,
  // the insertion should fail.
  BOOST_TEST(not SM.addStep(std::unique_ptr<Step>(new SingleDependencyStep())));
  BOOST_TEST(SM.getNumSteps() == 3);
  // Again the schedule remains in a valid state, because the bad step is not
  // added.
  BOOST_TEST(SM.hasValidSchedule());

  // We then add a new instance of the invalidated StepWithNoDeps, that is
  // successfully appended to the end of the schedule.
  BOOST_TEST(SM.addStep(std::unique_ptr<Step>(new StepWithNoDeps())));
  BOOST_TEST(SM.getNumSteps() == 4);
  BOOST_TEST(SM.hasValidSchedule());

  // Now the second instance of SingleDependencyStep can be added
  // successfully because the second instance of StepWithNoDeps is scheduled
  // before it and no other step invalidates it.
  BOOST_TEST(SM.addStep(std::unique_ptr<Step>(new SingleDependencyStep())));
  BOOST_TEST(SM.getNumSteps() == 5);
  BOOST_TEST(SM.hasValidSchedule());
}
