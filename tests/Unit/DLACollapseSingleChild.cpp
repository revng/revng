//
// Copyright rev.ng Srls. See LICENSE.md for details.
//

#define BOOST_TEST_MODULE DLACollapseSingleChild
bool init_unit_test();

#include "boost/test/unit_test.hpp"

#include "revng-c/DataLayoutAnalysis/DLATypeSystem.h"
#include "revng-c/Utils/Utils.h"

#include "lib/DataLayoutAnalysis/Middleend/DLAStep.h"

using LTSN = dla::LayoutTypeSystemNode;

using namespace llvm;
using namespace dla;

static void runCollapseSingleChild(LayoutTypeSystem &TS) {
  // Run CollapseSingleChild
  dla::StepManager SM;
  revng_check(SM.addStep<dla::CollapseSingleChild>());
  SM.run(TS);

  // Compress the equivalence classes
  dla::VectEqClasses &Eq = TS.getEqClasses();
  Eq.compress();
}

static LTSN *addChildAtOffset(LayoutTypeSystem &TS,
                              LTSN *Parent,
                              unsigned ChildOffset,
                              unsigned ChildSize) {
  LTSN *Child = TS.createArtificialLayoutType();
  Child->Size = ChildSize;
  OffsetExpression OE{};
  OE.Offset = ChildOffset;
  TS.addInstanceLink(Parent, Child, std::move(OE));

  return Child;
}

// Test cases

///\brief Test nominal case with one parent and one instance child at offset 0
BOOST_AUTO_TEST_CASE(TestShouldCollapse) {
  dla::LayoutTypeSystem TS;

  // Build TS
  LTSN *Parent = TS.createArtificialLayoutType();
  Parent->Size = 0U;
  unsigned ParentID = Parent->ID;

  LTSN *Child = addChildAtOffset(TS, Parent, /*offset=*/0U, /*size=*/8U);
  unsigned ChildID = Child->ID;

  // Run step
  runCollapseSingleChild(TS);

  // Check graph
  revng_check(TS.getNumLayouts() == 1);
  LTSN *SurvivedNode = (*TS.getLayoutsRange().begin());
  revng_check(SurvivedNode->ID == ParentID);
  revng_check(SurvivedNode->Size == 8U);

  // Check Eq Classes
  dla::VectEqClasses &Eq = TS.getEqClasses();
  revng_check(Eq.getNumElements() == 2);
  revng_check(Eq.getNumClasses() == 1);
  revng_check(not Eq.isRemoved(ParentID) and not Eq.isRemoved(ChildID));
  revng_check(Eq.haveSameEqClass(ParentID, ChildID));
  revng_check(Eq.getEqClassID(ParentID) == Eq.getEqClassID(ChildID));
}

///\brief Test the case where there is only one child but with offset > 0
BOOST_AUTO_TEST_CASE(TestSingleChildWithOffset) {
  dla::LayoutTypeSystem TS;

  // Build TS
  LTSN *Parent = TS.createArtificialLayoutType();
  Parent->Size = 0U;
  unsigned ParentID = Parent->ID;

  LTSN *Child = addChildAtOffset(TS, Parent, /*offset=*/8U, /*size=*/8U);
  unsigned ChildID = Child->ID;

  // Run step
  runCollapseSingleChild(TS);

  // Check graph
  revng_check(TS.getNumLayouts() == 2);

  // Check Eq Classes
  dla::VectEqClasses &Eq = TS.getEqClasses();
  revng_check(Eq.getNumElements() == 2);
  revng_check(Eq.getNumClasses() == 2);
  revng_check(not Eq.isRemoved(ParentID) and not Eq.isRemoved(ChildID));
  revng_check(not Eq.haveSameEqClass(ParentID, ChildID));
  revng_check(Eq.getEqClassID(ParentID) != Eq.getEqClassID(ChildID));
}

///\brief Test the case in which a node has many children (should not collapse)
BOOST_AUTO_TEST_CASE(TestMultipleChilds) {
  dla::LayoutTypeSystem TS;

  // Build TS
  LTSN *Parent = TS.createArtificialLayoutType();
  Parent->Size = 0U;
  unsigned ParentID = Parent->ID;

  LTSN *Child1 = addChildAtOffset(TS, Parent, /*offset=*/0U, /*size=*/8U);
  unsigned Child1ID = Child1->ID;

  LTSN *Child2 = addChildAtOffset(TS, Parent, /*offset=*/0U, /*size=*/16U);
  unsigned Child2ID = Child2->ID;

  // Run step
  runCollapseSingleChild(TS);

  // Check graph
  revng_check(TS.getNumLayouts() == 3);

  // Check Eq Classes
  dla::VectEqClasses &Eq = TS.getEqClasses();
  revng_check(Eq.getNumElements() == 3);
  revng_check(Eq.getNumClasses() == 3);
  revng_check(not Eq.isRemoved(ParentID) and not Eq.isRemoved(Child1ID)
              and not Eq.isRemoved(Child2ID));
  revng_check(not Eq.haveSameEqClass(ParentID, Child1ID)
              and not Eq.haveSameEqClass(ParentID, Child2ID));
  revng_check(Eq.getEqClassID(ParentID) != Eq.getEqClassID(Child1ID));
  revng_check(Eq.getEqClassID(ParentID) != Eq.getEqClassID(Child2ID));
}

///\brief Test the case in which there are multiple levels of single-childs to
/// be collapsed
BOOST_AUTO_TEST_CASE(TestMultiLevel) {
  dla::LayoutTypeSystem TS;

  // Build TS
  LTSN *Level0 = TS.createArtificialLayoutType();
  Level0->Size = 0U;
  unsigned Level0ID = Level0->ID;

  LTSN *Level1 = addChildAtOffset(TS, Level0, /*offset=*/0U, /*size=*/0U);
  unsigned Level1ID = Level1->ID;

  LTSN *Level2 = addChildAtOffset(TS, Level1, /*offset=*/0U, /*size=*/8U);
  unsigned Level2ID = Level2->ID;

  // Run step
  runCollapseSingleChild(TS);

  // Check graph
  revng_check(TS.getNumLayouts() == 1);
  LTSN *SurvivedNode = (*TS.getLayoutsRange().begin());
  revng_check(SurvivedNode->ID == Level0ID);
  revng_check(SurvivedNode->Size == 8U);

  // Check Eq Classes
  dla::VectEqClasses &Eq = TS.getEqClasses();
  revng_check(Eq.getNumElements() == 3);
  revng_check(Eq.getNumClasses() == 1);
  revng_check(not Eq.isRemoved(Level0ID) and not Eq.isRemoved(Level1ID)
              and not Eq.isRemoved(Level2ID));
  revng_check(Eq.haveSameEqClass(Level0ID, Level1ID)
              and Eq.haveSameEqClass(Level1ID, Level2ID));
  revng_check(Eq.getEqClassID(Level0ID) == Eq.getEqClassID(Level1ID)
              and Eq.getEqClassID(Level1ID) == Eq.getEqClassID(Level2ID));
}