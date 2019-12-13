#pragma once

//
// Copyright (c) rev.ng Srls. See LICENSE.md for details.
//

#include <algorithm>
#include <memory>
#include <type_traits>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

namespace dla {

// Forward declaration for dla::LayoutTypeSystem, that will be defined later.
// For this preliminary implementation, dla::Step does not really need the
// implementation of the dla::LayoutTypeSystem class, because for now we are
// only concerned about the schedule of the Steps, and the Step implementations
// are really just empty.
class LayoutTypeSystem;

class Step {
public:
  using IDSet = llvm::SmallPtrSet<const void *, 2>;
  using IDSetRef = llvm::SmallPtrSetImpl<const void *> &;
  using IDSetConstRef = const llvm::SmallPtrSetImpl<const void *> &;

protected:
  const void *StepID;

  IDSet Dependencies;
  IDSet Invalidated;

  Step(const char &C,
       std::initializer_list<const void *> D,
       std::initializer_list<const void *> I) :
    StepID(&C), Dependencies(D), Invalidated(I) {}

  Step(const char &C) : Step(C, {}, {}) {}

public:
  Step() = delete;
  virtual ~Step() = default;

  /// Runs the Step on TS, returns true if it has applied changes to TS.
  virtual bool runOnTypeSystem(LayoutTypeSystem &TS) = 0;

  IDSetConstRef getDependencies() const { return Dependencies; }
  IDSetConstRef getInvalidated() const { return Invalidated; }

  const void *getStepID() const { return StepID; };
};

/// dla::Step that creates types for Function's return types and fromal args
class CreateInterproceduralTypes : public Step {
  static const char ID;

public:
  static const constexpr void *getID() { return &ID; }

  CreateInterproceduralTypes() : Step(ID){};

  virtual ~CreateInterproceduralTypes() override = default;

  virtual bool runOnTypeSystem(LayoutTypeSystem &TS) override { return true; }
};

/// dla::Step that creates types for LLVM Values inside Functions and edges
/// between them.
class CreateIntraproceduralTypes : public Step {
  static const char ID;

public:
  static const constexpr void *getID() { return &ID; }

  CreateIntraproceduralTypes() : Step(ID){};

  virtual ~CreateIntraproceduralTypes() override = default;

  virtual bool runOnTypeSystem(LayoutTypeSystem &TS) override { return true; }
};

/// dla::Step that collapses loops in the type system with equality or
/// inheritange edges
//
// After the execution of this step, the LayoutTypeSystem graph should contain
// only inheritance and instance-of edges, and should be a DAG.
class CollapseIdentityAndInheritanceCC : public Step {
  static const char ID;

public:
  static const constexpr void *getID() { return &ID; }

  CollapseIdentityAndInheritanceCC() : Step(ID){};

  virtual ~CollapseIdentityAndInheritanceCC() override = default;

  virtual bool runOnTypeSystem(LayoutTypeSystem &TS) override { return true; }
};

/// dla::Step that removes transitive inheritance edges
//
// Here we use the notation notation A --> B to mean an inheritance edge.
// After the execution of this step, the LayoutTypeSystem graph should have the
// following property: For each types A, B, C (all different from each other),
// if A --> B and B --> C, then we should not have A --> C.
class RemoveTransitiveInheritanceEdges : public Step {
  static const char ID;

public:
  static const constexpr void *getID() { return &ID; }

  RemoveTransitiveInheritanceEdges() : Step(ID){};

  virtual ~RemoveTransitiveInheritanceEdges() override = default;

  virtual bool runOnTypeSystem(LayoutTypeSystem &TS) override { return true; }
};

/// dla::Step that computes and propagates informations on accesses and type
/// sizes.
class ComputeUpperMemberAccesses : public Step {
  static const char ID;

public:
  static const constexpr void *getID() { return &ID; }

  ComputeUpperMemberAccesses() :
    Step(ID,
         // Dependencies
         { CollapseIdentityAndInheritanceCC::getID() },
         // Invalidated
         {}) {}

  virtual ~ComputeUpperMemberAccesses() override = default;

  virtual bool runOnTypeSystem(LayoutTypeSystem &TS) override { return true; }
};

/// dla::Step that tries to aggregate compatible arrays into a single array
class CollapseCompatibleArrays : public Step {
  static const char ID;

public:
  static const constexpr void *getID() { return &ID; }

  CollapseCompatibleArrays() :
    Step(ID,
         // Dependencies
         { ComputeUpperMemberAccesses::getID() },
         // Invalidated
         {}) {}

  virtual ~CollapseCompatibleArrays() override = default;

  virtual bool runOnTypeSystem(LayoutTypeSystem &TS) override { return true; }
};

/// dla::Step that propagates inheritance relationships through accessor methods
class PropagateInheritanceToAccessors : public Step {
  static const char ID;

public:
  static const constexpr void *getID() { return &ID; }

  PropagateInheritanceToAccessors() :
    Step(ID,
         // Dependencies
         {},
         // Invalidated
         { RemoveTransitiveInheritanceEdges::getID() }) {}

  virtual ~PropagateInheritanceToAccessors() override = default;

  virtual bool runOnTypeSystem(LayoutTypeSystem &TS) override { return true; }
};

/// dla::Step that decompose the LayoutTypeSystem into components, each of which
/// cannot overlap with others
class ComputeNonInterferingComponents : public Step {
  static const char ID;

public:
  static const constexpr void *getID() { return &ID; }

  ComputeNonInterferingComponents() :
    Step(ID,
         // Dependencies
         { ComputeUpperMemberAccesses::getID() },
         // Invalidated
         {}) {}

  virtual ~ComputeNonInterferingComponents() override = default;

  virtual bool runOnTypeSystem(LayoutTypeSystem &TS) override { return true; }
};

/// Final dla::Step, which flattens out the types into memory layouts
class MakeLayouts : public Step {
  static const char ID;

public:
  static const constexpr void *getID() { return &ID; }

  MakeLayouts() :
    Step(ID,
         // Dependencies
         { CollapseIdentityAndInheritanceCC::getID(),
           RemoveTransitiveInheritanceEdges::getID() },
         // Invalidated
         {}) {}

  virtual ~MakeLayouts() override = default;

  virtual bool runOnTypeSystem(LayoutTypeSystem &TS) override { return true; }
};

template<typename IterT>
bool intersect(IterT I1, IterT E1, IterT I2, IterT E2) {
  while ((I1 != E1) and (I2 != E2)) {
    if (*I1 < *I2)
      ++I1;
    else if (*I2 < *I1)
      ++I2;
    else
      return true;
  }
  return false;
}

template<typename RangeT>
bool intersect(const RangeT &R1, const RangeT &R2) {
  return intersect(R1.begin(), R1.end(), R2.begin(), R2.end());
}

class StepManager {

public:
  enum StepState {
    NewlyAdded,
    Invalidated,
    Done,
  };

public:
  llvm::SmallVector<std::unique_ptr<Step>, 16> Schedule;
  llvm::SmallPtrSet<const void *, 16> InsertedSteps;
  llvm::SmallPtrSet<const void *, 16> InvalidatedSteps;

  using sched_const_iterator = decltype(Schedule)::const_iterator;
  using sched_const_range = llvm::iterator_range<sched_const_iterator>;

public:
  StepManager() : Schedule(), InsertedSteps(), InvalidatedSteps() {}

  /// Adds a Step to the StepManager, moving ownership into it.
  [[nodiscard]] bool addStep(std::unique_ptr<Step> S);

  /// Runs the added steps
  void run(LayoutTypeSystem &TS);

  /// Drops all the scheduled steps
  void reset() {
    Schedule.clear();
    InsertedSteps.clear();
    InvalidatedSteps.clear();
  }

  bool hasValidSchedule() const {
    return not intersect(InsertedSteps, InvalidatedSteps);
  }

  /// Get the number of scheduled steps
  auto getNumSteps() const { return Schedule.size(); }

  // Methods for const iteration on Schedule
  sched_const_iterator sched_begin() const { return Schedule.begin(); }
  sched_const_iterator sched_end() const { return Schedule.end(); }
  sched_const_range sched() const {
    return llvm::make_range(sched_begin(), sched_end());
  }
};

} // end namespace dla
