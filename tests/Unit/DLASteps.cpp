//
// Copyright rev.ng Srls. See LICENSE.md for details.
//

#define BOOST_TEST_MODULE DLASteps
bool init_unit_test();

#include "boost/test/unit_test.hpp"

#include "revng-c/DataLayoutAnalysis/DLATypeSystem.h"
#include "revng-c/Utils/Utils.h"

#include "lib/DataLayoutAnalysis/Middleend/DLAStep.h"

using LTSN = dla::LayoutTypeSystemNode;

using namespace llvm;
using namespace dla;

template<typename StepT>
static void runStep(LayoutTypeSystem &TS) {
  // Enable expensive checks
  VerifyLog.enable();

  // Run Step
  dla::StepManager SM;
  revng_check(SM.addStep<StepT>());
  SM.run(TS);

  // Compress the equivalence classes
  dla::VectEqClasses &Eq = TS.getEqClasses();
  Eq.compress();
}

static LTSN *createRoot(LayoutTypeSystem &TS, unsigned Size = 0U) {
  LTSN *Root = TS.createArtificialLayoutType();
  Root->Size = Size;

  return Root;
}

static LTSN *addInstanceAtOffset(LayoutTypeSystem &TS,
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

static LTSN *
addInheritance(LayoutTypeSystem &TS, LTSN *Parent, unsigned ChildSize = 0U) {
  LTSN *Child = TS.createArtificialLayoutType();
  Child->Size = ChildSize;
  TS.addInheritanceLink(Parent, Child);

  return Child;
}

static LTSN *
addEquality(LayoutTypeSystem &TS, LTSN *Parent, unsigned ChildSize = 0U) {
  LTSN *Child = TS.createArtificialLayoutType();
  Child->Size = ChildSize;
  TS.addEqualityLink(Parent, Child);

  return Child;
}

void checkNode(const LayoutTypeSystem &TS,
               const LayoutTypeSystemNode *N,
               const unsigned ExpectedSize,
               const InterferingChildrenInfo ExpectedInfo,
               const std::set<unsigned> &ExpectedEqClass) {
  const dla::VectEqClasses &Eq = TS.getEqClasses();
  revng_check(N);
  revng_check(not Eq.isRemoved(N->ID));
  revng_check(ExpectedSize == N->Size);
  revng_check(ExpectedInfo == N->InterferingInfo);

  const auto &EqClass = Eq.computeEqClass(N->ID);
  revng_check(EqClass.size() == ExpectedEqClass.size());
  for (auto &Collapsed : EqClass)
    revng_check(ExpectedEqClass.contains(Collapsed));
}

// Test cases

// ----------------- CollapseIdentityAndInheritanceCC ---

///\brief Test an equality CC with a pointer edge
BOOST_AUTO_TEST_CASE(CollapseIdentityAndInheritanceCC_equality) {
  dla::LayoutTypeSystem TS;

  // Build TS
  LTSN *Root = createRoot(TS);
  LTSN *Node1 = addEquality(TS, Root);
  LTSN *Node2 = addEquality(TS, Root);
  LTSN *Node3 = addEquality(TS, Node2);
  TS.addEqualityLink(Node1, Node3);
  LTSN *PtrNode = createRoot(TS);
  TS.addPointerLink(Node3, PtrNode);

  runStep<dla::CollapseIdentityAndInheritanceCC>(TS);

  // Check that equality nodes have been collapsed
  revng_check(TS.getNumLayouts() == 2);
  revng_check(llvm::is_contained(TS.getLayoutsRange(), Node2));
  revng_check(llvm::is_contained(TS.getLayoutsRange(), PtrNode));
  revng_check(Node2->Successors.size() == 1);
  // Check that the pointer edge is still alive
  const auto &[Child, Tag] = *(llvm::children_edges<LTSN *>(Node2).begin());
  revng_check(Child == PtrNode);
  revng_check(Tag->getKind() == dla::TypeLinkTag::LK_Pointer);

  // Check Eq Classes
  dla::VectEqClasses &Eq = TS.getEqClasses();
  revng_check(Eq.getNumElements() == 5);
  revng_check(Eq.getNumClasses() == 2);
  checkNode(TS, Node2, 0, InterferingChildrenInfo::Unknown, { 0, 1, 2, 3 });
  checkNode(TS, PtrNode, 0, InterferingChildrenInfo::Unknown, { 4 });
}

///\brief Test an inheritance loop with a pointer edge
BOOST_AUTO_TEST_CASE(CollapseIdentityAndInheritanceCC_inheritance) {
  dla::LayoutTypeSystem TS;

  // Build TS
  LTSN *Root = createRoot(TS);
  LTSN *Node1 = addInheritance(TS, Root);
  LTSN *Node2 = addInheritance(TS, Root);
  LTSN *Node3 = addInheritance(TS, Node2);
  TS.addInheritanceLink(Node3, Root);
  LTSN *Node4 = addInheritance(TS, Node1);

  LTSN *PtrNode = createRoot(TS);
  TS.addPointerLink(Node4, PtrNode);

  runStep<dla::CollapseIdentityAndInheritanceCC>(TS);
  // Check that inheritance loops have been collapsed
  revng_check(TS.getNumLayouts() == 4);
  revng_check(llvm::is_contained(TS.getLayoutsRange(), Node3));
  revng_check(llvm::is_contained(TS.getLayoutsRange(), Node1));
  revng_check(llvm::is_contained(TS.getLayoutsRange(), Node4));
  revng_check(llvm::is_contained(TS.getLayoutsRange(), PtrNode));
  revng_check(Node3->Successors.size() == 1);
  // Check that the pointer edge is still alive
  const auto &[Child, Tag] = *(llvm::children_edges<LTSN *>(Node4).begin());
  revng_check(Child == PtrNode);
  revng_check(Tag->getKind() == dla::TypeLinkTag::LK_Pointer);

  // Check Eq Classes
  dla::VectEqClasses &Eq = TS.getEqClasses();
  revng_check(Eq.getNumElements() == 6);
  revng_check(Eq.getNumClasses() == 4);
  checkNode(TS, Node3, 0, InterferingChildrenInfo::Unknown, { 0, 2, 3 });
  checkNode(TS, Node1, 0, InterferingChildrenInfo::Unknown, { 1 });
  checkNode(TS, Node4, 0, InterferingChildrenInfo::Unknown, { 4 });
  checkNode(TS, PtrNode, 0, InterferingChildrenInfo::Unknown, { 5 });
}

///\brief Test an inheritance CC with a backward instance edge (loop)
BOOST_AUTO_TEST_CASE(CollapseIdentityAndInheritanceCC_instance) {
  dla::LayoutTypeSystem TS;

  // Build TS
  LTSN *Root = createRoot(TS);
  LTSN *Node1 = addInheritance(TS, Root);
  LTSN *Node2 = addInheritance(TS, Root);
  LTSN *Node3 = addInheritance(TS, Node2);
  TS.addInstanceLink(Node3, Root, OffsetExpression{});
  LTSN *Node4 = addInstanceAtOffset(TS, Node1, 0, 8);

  LTSN *PtrNode = createRoot(TS);
  TS.addPointerLink(Node4, PtrNode);

  // Run step
  runStep<dla::CollapseIdentityAndInheritanceCC>(TS);

  // Check that no node has been collapsed
  revng_check(TS.getNumLayouts() == 6);

  // Check that the pointer edge is still alive
  const auto &[Child, Tag] = *(llvm::children_edges<LTSN *>(Node4).begin());
  revng_check(Child == PtrNode);
  revng_check(Tag->getKind() == dla::TypeLinkTag::LK_Pointer);

  // Check that the instance back-edge has been removed
  revng_check(Node3->Successors.size() == 0);

  // Check Eq Classes
  dla::VectEqClasses &Eq = TS.getEqClasses();
  revng_check(Eq.getNumElements() == 6);
  revng_check(Eq.getNumClasses() == 6);
}

// ----------------- RemoveTransitiveInheritanceEdges ---

///\brief Test that transitive edges are removed and pointer edges ignored
BOOST_AUTO_TEST_CASE(RemoveTransitiveInheritanceEdges_basic) {
  dla::LayoutTypeSystem TS;

  // Build TS
  LTSN *Root = createRoot(TS);
  LTSN *Node1 = addInheritance(TS, Root);
  LTSN *Node2 = addInheritance(TS, Node1);
  LTSN *Node3 = addInheritance(TS, Node2);
  // Add transitive edges
  TS.addInheritanceLink(Root, Node2);
  TS.addInheritanceLink(Root, Node3);
  // Add pointer edge
  LTSN *PtrNode = createRoot(TS);
  TS.addPointerLink(Node3, PtrNode);
  TS.addInheritanceLink(Root, PtrNode);

  // Run step
  runStep<dla::RemoveTransitiveInheritanceEdges>(TS);

  // Check that no node has been collapsed
  revng_check(TS.getNumLayouts() == 5);
  // Check that the pointer edges are still alive
  const auto &[Child, Tag] = *(llvm::children_edges<LTSN *>(Node3).begin());
  revng_check(Child == PtrNode);
  revng_check(Tag->getKind() == dla::TypeLinkTag::LK_Pointer);
  // Check that transitive edges have been removed
  revng_check(Root->Successors.size() == 2);
  revng_check(llvm::is_contained(llvm::children<LTSN *>(Root), Node1));
  // The edge to the pointer node should not be removed
  revng_check(llvm::is_contained(llvm::children<LTSN *>(Root), PtrNode));

  // Check Eq Classes
  dla::VectEqClasses &Eq = TS.getEqClasses();
  revng_check(Eq.getNumElements() == 5);
  revng_check(Eq.getNumClasses() == 5);
  revng_check(not Eq.isRemoved(Node3->ID) and not Eq.isRemoved(PtrNode->ID));
}

// ----------------- MakeInheritanceTree ----------------

///\brief Test that inheritance "diamonds" are removed and pointer edges ignored
BOOST_AUTO_TEST_CASE(MakeInheritanceTree_diamond) {
  dla::LayoutTypeSystem TS;

  // Build TS
  LTSN *Root = createRoot(TS);
  LTSN *Node1 = addInheritance(TS, Root);
  LTSN *Node2 = addInheritance(TS, Root);
  LTSN *Node3 = addInheritance(TS, Node1);
  TS.addInheritanceLink(Node2, Node3);
  // Add pointer edge
  LTSN *PtrNode = createRoot(TS);
  TS.addPointerLink(Node3, PtrNode);

  runStep<dla::MakeInheritanceTree>(TS);
  // Check that nodes have been collapsed
  revng_check(TS.getNumLayouts() == 4);
  // Check that the pointer edges are still alive
  const auto &[Child, Tag] = *(llvm::children_edges<LTSN *>(Node3).begin());
  revng_check(Child == PtrNode);
  revng_check(Tag->getKind() == dla::TypeLinkTag::LK_Pointer);
  // Check diamonds have been collapsed
  revng_check(Root->Successors.size() == 1);
  revng_check(llvm::is_contained(llvm::children<LTSN *>(Root), Node1));
  revng_check(not llvm::is_contained(llvm::children<LTSN *>(Root), Node2));
  // Check Eq Classes
  dla::VectEqClasses &Eq = TS.getEqClasses();
  revng_check(Eq.getNumElements() == 5);
  revng_check(Eq.getNumClasses() == 4);
  checkNode(TS, Root, 0, InterferingChildrenInfo::Unknown, { 0 });
  checkNode(TS, Node1, 0, InterferingChildrenInfo::Unknown, { 1, 2 });
  checkNode(TS, Node3, 0, InterferingChildrenInfo::Unknown, { 3 });
  checkNode(TS, PtrNode, 0, InterferingChildrenInfo::Unknown, { 4 });
}

///\brief Test that multiple inheritance is collapsed
BOOST_AUTO_TEST_CASE(MakeInheritanceTree_virtualPostDom) {
  dla::LayoutTypeSystem TS;

  // Build TS
  LTSN *Root = createRoot(TS);
  LTSN *Node1 = addInheritance(TS, Root);
  LTSN *Node2 = addInheritance(TS, Root);
  LTSN *Node3 = addInheritance(TS, Node1);
  TS.addInheritanceLink(Node2, Node3);
  /*LTSN *Node4=*/addInheritance(TS, Root);
  // Add pointer edge
  LTSN *PtrNode = createRoot(TS);
  TS.addPointerLink(Node3, PtrNode);

  // Run step
  runStep<dla::MakeInheritanceTree>(TS);

  // Check that nodes have been collapsed
  revng_check(TS.getNumLayouts() == 3);
  // Check that the pointer edges are still alive
  const auto &[Child, Tag] = *(llvm::children_edges<LTSN *>(Node3).begin());
  revng_check(Child == PtrNode);
  revng_check(Tag->getKind() == dla::TypeLinkTag::LK_Pointer);
  // Check that all inheritance children have been collapsed
  revng_check(Root->Successors.size() == 1);
  revng_check(llvm::is_contained(llvm::children<LTSN *>(Root), Node3));
  // Check Eq Classes
  dla::VectEqClasses &Eq = TS.getEqClasses();
  revng_check(Eq.getNumElements() == 6);
  revng_check(Eq.getNumClasses() == 3);
  checkNode(TS, Root, 0, InterferingChildrenInfo::Unknown, { 0 });
  checkNode(TS, Node3, 0, InterferingChildrenInfo::Unknown, { 1, 2, 3, 4 });
  checkNode(TS, PtrNode, 0, InterferingChildrenInfo::Unknown, { 5 });
}

// ----------------- PruneLayoutNodesWithoutLayout ------

///\brief Test that branches without sized leaves are pruned
BOOST_AUTO_TEST_CASE(PruneLayoutNodesWithoutLayout_basic) {
  dla::LayoutTypeSystem TS;

  // Build TS
  LTSN *Root = createRoot(TS);
  std::vector<LTSN *> Level0 = {
    addInstanceAtOffset(TS, Root, /*offset=*/0, /*size=*/0),
    addInstanceAtOffset(TS, Root, /*offset=*/0, /*size=*/0)
  };
  std::vector<LTSN *> Level1 = {
    addInstanceAtOffset(TS, Level0[0], /*offset=*/0, /*size=*/0),
    addInstanceAtOffset(TS, Level0[0], /*offset=*/0, /*size=*/0),
    addInstanceAtOffset(TS, Level0[1], /*offset=*/0, /*size=*/0),
    addInstanceAtOffset(TS, Level0[1], /*offset=*/0, /*size=*/0)
  };
  std::vector<LTSN *> Leaves = {
    addInstanceAtOffset(TS, Level1[0], /*offset=*/0, /*size=*/8),
    addInstanceAtOffset(TS, Level1[1], /*offset=*/0, /*size=*/8),
    addInstanceAtOffset(TS, Level1[2], /*offset=*/0, /*size=*/0),
    addInstanceAtOffset(TS, Level1[3], /*offset=*/0, /*size=*/0)
  };
  // Add pointers
  std::vector<LTSN *> PtrNodes;
  for (LTSN *Leaf : Leaves) {
    LTSN *PtrNode = createRoot(TS);
    // PtrNode->Size = 8U;
    TS.addPointerLink(Leaf, PtrNode);
    PtrNodes.push_back(PtrNode);
  }

  // Run step
  runStep<dla::PruneLayoutNodesWithoutLayout>(TS);

  // Check final tree
  revng_check(TS.getNumLayouts() == 6);
  revng_check(Root->Successors.size() == 1);
  revng_check(llvm::is_contained(llvm::children<LTSN *>(Root), Level0[0]));
  revng_check(not llvm::is_contained(llvm::children<LTSN *>(Root), Level0[1]));

  revng_check(Level0[0]->Successors.size() == 2);
  revng_check(llvm::is_contained(llvm::children<LTSN *>(Level0[0]), Level1[0]));
  revng_check(llvm::is_contained(llvm::children<LTSN *>(Level0[0]), Level1[1]));

  revng_check(Level1[0]->Successors.size() == 1);
  revng_check(llvm::is_contained(llvm::children<LTSN *>(Level1[0]), Leaves[0]));
  revng_check(Level1[1]->Successors.size() == 1);
  revng_check(llvm::is_contained(llvm::children<LTSN *>(Level1[1]), Leaves[1]));

  revng_check(isLeaf(Leaves[0]) and isLeaf(Leaves[1]));

  // Check Eq Classes
  dla::VectEqClasses &Eq = TS.getEqClasses();
  revng_check(Eq.getNumElements() == 15);
  revng_check(Eq.getNumClasses() == 7);

  // All leaves should have been removed
  std::vector<unsigned> RemovedIDs = { 2, 5, 6, 9, 10, 11, 12, 13, 14 };
  for (auto ID : RemovedIDs)
    revng_check(Eq.isRemoved(ID));
}

// ----------------- CollapseSingleChild ----------------

///\brief Test nominal case with one parent and one instance child at offset 0
BOOST_AUTO_TEST_CASE(CollapseSingleChild_offsetZero) {
  dla::LayoutTypeSystem TS;

  // Build TS
  LTSN *Parent = createRoot(TS, 0U);
  /*LTSN *Child =*/addInstanceAtOffset(TS,
                                       Parent,
                                       /*offset=*/0U,
                                       /*size=*/8U);

  // Run step
  runStep<dla::CollapseSingleChild>(TS);

  // Check graph
  revng_check(TS.getNumLayouts() == 1);
  checkNode(TS, Parent, 8, InterferingChildrenInfo::Unknown, { 0, 1 });
}

///\brief Test the case where there is only one child but with offset > 0
BOOST_AUTO_TEST_CASE(CollapseSingleChild_offsetNonZero) {
  dla::LayoutTypeSystem TS;

  // Build TS
  LTSN *Parent = createRoot(TS, 0U);
  LTSN *Child = addInstanceAtOffset(TS,
                                    Parent,
                                    /*offset=*/8U,
                                    /*size=*/8U);

  // Run step
  runStep<dla::CollapseSingleChild>(TS);

  // Check graph
  revng_check(TS.getNumLayouts() == 2);
  checkNode(TS, Parent, 0, InterferingChildrenInfo::Unknown, { 0 });
  checkNode(TS, Child, 8, InterferingChildrenInfo::Unknown, { 1 });
}

///\brief Test the case in which a node has many children (should not collapse)
BOOST_AUTO_TEST_CASE(CollapseSingleChild_multiChild) {
  dla::LayoutTypeSystem TS;

  // Build TS
  LTSN *Parent = createRoot(TS, 0U);
  LTSN *Child1 = addInstanceAtOffset(TS,
                                     Parent,
                                     /*offset=*/0U,
                                     /*size=*/0U);
  addInstanceAtOffset(TS,
                      Child1,
                      /*offset=*/0U,
                      /*size=*/8U);
  LTSN *Child2 = addInstanceAtOffset(TS,
                                     Parent,
                                     /*offset=*/8U,
                                     /*size=*/0U);
  addInstanceAtOffset(TS,
                      Child2,
                      /*offset=*/0U,
                      /*size=*/8U);

  // Run step
  runStep<dla::CollapseSingleChild>(TS);

  // Check graph
  revng_check(TS.getNumLayouts() == 3);
  checkNode(TS, Parent, 0, InterferingChildrenInfo::Unknown, { 0 });
  checkNode(TS, Child1, 8, InterferingChildrenInfo::Unknown, { 1, 2 });
  checkNode(TS, Child2, 8, InterferingChildrenInfo::Unknown, { 3, 4 });
}

///\brief Test the case in which a node has many parents (should not collapse)
BOOST_AUTO_TEST_CASE(CollapseSingleChild_multiParent) {
  dla::LayoutTypeSystem TS;

  // Build TS
  LTSN *Parent1 = createRoot(TS, 0U);
  LTSN *Parent2 = createRoot(TS, 0U);
  LTSN *Child1 = addInstanceAtOffset(TS,
                                     Parent1,
                                     /*offset=*/8U,
                                     /*size=*/0U);
  TS.addInheritanceLink(Parent2, Child1);
  /*LTSN *Child2 =*/addInstanceAtOffset(TS,
                                        Child1,
                                        /*offset=*/0U,
                                        /*size=*/8U);

  // Run step
  runStep<dla::CollapseSingleChild>(TS);

  // Check graph
  revng_check(TS.getNumLayouts() == 3);
  checkNode(TS, Parent1, 0, InterferingChildrenInfo::Unknown, { 0 });
  checkNode(TS, Parent2, 0, InterferingChildrenInfo::Unknown, { 1 });
  checkNode(TS, Child1, 8, InterferingChildrenInfo::Unknown, { 2, 3 });
}

///\brief Test the case in which there are multiple levels of single-childs to
/// be collapsed
BOOST_AUTO_TEST_CASE(CollapseSingleChild_multiLevel) {
  dla::LayoutTypeSystem TS;

  // Build TS
  LTSN *Level0 = createRoot(TS, 0U);
  LTSN *Level1 = addInstanceAtOffset(TS,
                                     Level0,
                                     /*offset=*/0U,
                                     /*size=*/0U);
  LTSN *Level2 = addInstanceAtOffset(TS,
                                     Level1,
                                     /*offset=*/0U,
                                     /*size=*/0U);
  LTSN *Level3 = addInstanceAtOffset(TS,
                                     Level2,
                                     /*offset=*/0U,
                                     /*size=*/0U);
  /*LTSN *Level4 = */ addInstanceAtOffset(TS,
                                          Level3,
                                          /*offset=*/0U,
                                          /*size=*/8U);

  // Run step
  runStep<dla::CollapseSingleChild>(TS);

  // Check graph
  revng_check(TS.getNumLayouts() == 1);
  checkNode(TS, Level0, 8, InterferingChildrenInfo::Unknown, { 0, 1, 2, 3, 4 });
}

// ----------------- ComputeUpperMemberAccesses ---------

///\brief Test member access computation
BOOST_AUTO_TEST_CASE(ComputeUpperMemberAccesses_basic) {
  dla::LayoutTypeSystem TS;

  // Build TS
  LTSN *Root = createRoot(TS);
  Root->Size = 0U;

  LTSN *Child1 = createRoot(TS);
  Child1->Size = 8U;
  TS.addInstanceLink(Root, Child1, OffsetExpression{});

  LTSN *Child2 = createRoot(TS);
  Child2->Size = 8U;
  OffsetExpression MultiOE;
  MultiOE.Offset = 10U;
  MultiOE.Strides = { 10U };
  MultiOE.TripCounts = { 3U };
  TS.addInstanceLink(Root, Child2, std::move(MultiOE));

  LTSN *Child3 = createRoot(TS);
  Child3->Size = 16U;
  OffsetExpression SingleOE;
  SingleOE.Offset = 30U;
  TS.addInstanceLink(Root, Child3, std::move(SingleOE));

  LTSN *Root2 = createRoot(TS);
  LTSN *PtrNode = createRoot(TS);
  PtrNode->Size = 8U;
  TS.addInheritanceLink(Root2, PtrNode);

  // Add pointer edge
  TS.addPointerLink(PtrNode, Root);

  // Run step
  dla::StepManager SM;
  revng_check(SM.addStep<CollapseIdentityAndInheritanceCC>());
  revng_check(SM.addStep<ComputeUpperMemberAccesses>());
  SM.run(TS);
  // Compress the equivalence classes
  dla::VectEqClasses &Eq = TS.getEqClasses();
  Eq.compress();

  // Check remaining nodes
  revng_check(TS.getNumLayouts() == 6);
  // Check edges
  unsigned NInstance = 0U;
  for (auto &[Child, Tag] : llvm::children_edges<LTSN *>(Root)) {
    switch (Tag->getKind()) {
    case TypeLinkTag::LK_Instance:
      NInstance++;
      break;
    default:
      revng_check(false);
    }
  }
  revng_check(NInstance == 3);
  // Check size
  revng_check(Root->Size == 46);
  revng_check(Root2->Size == 8);

  // Check Eq Classes
  revng_check(Eq.getNumElements() == 6);
  revng_check(Eq.getNumClasses() == 6);
  revng_check(not Eq.isRemoved(PtrNode->ID));
}

// ----------------- ComputeNonInterferingComponents ----

///\brief Test union nested inside a struct
BOOST_AUTO_TEST_CASE(ComputeNonInterferingComponents_basic) {
  dla::LayoutTypeSystem TS;

  // Build TS
  LTSN *Root = createRoot(TS);
  Root->Size = 0U;

  LTSN *Child1 = createRoot(TS);
  Child1->Size = 8U;
  TS.addInstanceLink(Root, Child1, OffsetExpression{});

  LTSN *Child2 = createRoot(TS);
  Child2->Size = 8U;
  OffsetExpression MultiOE;
  MultiOE.Offset = 10U;
  MultiOE.Strides = { 10U };
  MultiOE.TripCounts = { 3U };
  TS.addInstanceLink(Root, Child2, std::move(MultiOE));

  LTSN *Child3 = createRoot(TS);
  Child3->Size = 16U;
  OffsetExpression SingleOE;
  SingleOE.Offset = 30U;
  TS.addInstanceLink(Root, Child3, std::move(SingleOE));

  LTSN *Root2 = createRoot(TS);
  LTSN *PtrNode = createRoot(TS);
  PtrNode->Size = 8U;
  TS.addInheritanceLink(Root2, PtrNode);
  TS.addPointerLink(PtrNode, Root);
  LTSN *UnionNode = addInstanceAtOffset(TS, Root2, 0U, 8U);

  // Run step
  VerifyLog.enable();
  dla::StepManager SM;
  revng_check(SM.addStep<CollapseIdentityAndInheritanceCC>());
  revng_check(SM.addStep<ComputeUpperMemberAccesses>());
  revng_check(SM.addStep<ComputeNonInterferingComponents>());
  SM.run(TS);
  // Compress the equivalence classes
  dla::VectEqClasses &Eq = TS.getEqClasses();
  Eq.compress();

  // Check remaining nodes
  revng_check(TS.getNumLayouts() == 8);
  // Check edges
  unsigned NInstance = 0U;
  for (auto &[Child, Tag] : llvm::children_edges<LTSN *>(Root)) {
    switch (Tag->getKind()) {
    case TypeLinkTag::LK_Instance:
      NInstance++;
      break;
    default:
      revng_check(false);
    }
  }
  revng_check(NInstance == 2);
  // Check size
  revng_check(Root->Size == 46);
  revng_check(Root2->Size == 8);
  revng_check(Root->InterferingInfo = AllChildrenAreNonInterfering);
  revng_check(Child1->InterferingInfo = AllChildrenAreNonInterfering);
  revng_check(Child2->InterferingInfo = AllChildrenAreNonInterfering);
  revng_check(Root2->InterferingInfo = AllChildrenAreInterfering);
  revng_check(PtrNode->InterferingInfo = AllChildrenAreNonInterfering);
  revng_check(UnionNode->InterferingInfo = AllChildrenAreNonInterfering);
  // Check new node
  const auto &[FinalChild1, Tag1] = *(Child2->Predecessors.begin());
  const auto &[FinalChild2, Tag2] = *(Child3->Predecessors.begin());
  revng_check(Tag1->getOffsetExpr().Offset == 0U);
  revng_check(Tag2->getOffsetExpr().Offset == 20U);

  revng_check(FinalChild1 == FinalChild2);

  // Check Eq Classes
  revng_check(Eq.getNumElements() == 8);
  revng_check(Eq.getNumClasses() == 8);
  revng_check(not Eq.isRemoved(PtrNode->ID));
}

// ----------------- Propagate to accessors --------------

BOOST_AUTO_TEST_CASE(PropagateToAccessors) {
  dla::LayoutTypeSystem TS;

  // Build TS
  LTSN *NodeA = createRoot(TS);
  LTSN *NodeB = addInheritance(TS, NodeA);
  LTSN *NodeC = addInheritance(TS, NodeA);
  LTSN *NodeD = addInstanceAtOffset(TS, NodeB, /*offset=*/0, /*size=*/0);
  /*LTSN *NodeE = */ addInstanceAtOffset(TS, NodeD, /*offset=*/0, /*size=*/8);
  LTSN *NodeF = addInstanceAtOffset(TS, NodeC, /*offset=*/8, /*size=*/0);
  /*LTSN *NodeG = */ addInstanceAtOffset(TS, NodeF, /*offset=*/0, /*size=*/8);

  // Run steps
  VerifyLog.enable();
  dla::StepManager SM;
  revng_check(SM.addStep<CollapseIdentityAndInheritanceCC>());
  revng_check(SM.addStep<MakeInheritanceTree>());
  revng_check(SM.addStep<CollapseSingleChild>());
  revng_check(SM.addStep<ComputeUpperMemberAccesses>());

  SM.run(TS);

  // Compress the equivalence classes
  dla::VectEqClasses &Eq = TS.getEqClasses();
  Eq.compress();

  // Check TS
  revng_check(TS.getNumLayouts() == 3);
  checkNode(TS, NodeA, 16, InterferingChildrenInfo::Unknown, { 0, 1, 2 });
  checkNode(TS, NodeD, 8, InterferingChildrenInfo::Unknown, { 3, 4 });
  checkNode(TS, NodeF, 8, InterferingChildrenInfo::Unknown, { 5, 6 });
}

// ----------------- Deduplicate Union Fields --------------

BOOST_AUTO_TEST_CASE(DeduplicateUnionFields_basic) {
  dla::LayoutTypeSystem TS;

  // Build TS
  LTSN *NodeA = createRoot(TS);
  LTSN *NodeB = addInheritance(TS, NodeA);
  /*LTSN *NodeD =*/addInstanceAtOffset(TS, NodeB, /*offset=*/0, /*size=*/8);
  /*LTSN *NodeE =*/addInstanceAtOffset(TS, NodeB, /*offset=*/8, /*size=*/8);

  LTSN *NodeC = addInstanceAtOffset(TS, NodeA, /*offset=*/0, /*size=*/16);
  LTSN *NodeF = addInstanceAtOffset(TS, NodeC, /*offset=*/0, /*size=*/8);
  LTSN *NodeG = addInstanceAtOffset(TS, NodeC, /*offset=*/8, /*size=*/8);

  LTSN *Node1 = addInstanceAtOffset(TS, NodeA, /*offset=*/0, /*size=*/12);
  LTSN *Node2 = addInstanceAtOffset(TS, Node1, /*offset=*/4, /*size=*/8);
  LTSN *Node3 = addInstanceAtOffset(TS, Node2, /*offset=*/0, /*size=*/4);
  LTSN *Node4 = addInstanceAtOffset(TS, Node2, /*offset=*/4, /*size=*/4);

  // Run steps
  VerifyLog.enable();
  dla::StepManager SM;
  revng_check(SM.addStep<CollapseIdentityAndInheritanceCC>());
  revng_check(SM.addStep<ComputeUpperMemberAccesses>());
  revng_check(SM.addStep<ComputeNonInterferingComponents>());
  revng_check(SM.addStep<DeduplicateUnionFields>());
  revng_check(SM.addStep<ComputeNonInterferingComponents>());

  SM.run(TS);

  // Compress the equivalence classes
  dla::VectEqClasses &Eq = TS.getEqClasses();
  Eq.compress();

  // Check TS
  revng_check(TS.getNumLayouts() == 8);
  checkNode(TS, NodeA, 16, AllChildrenAreInterfering, { 0 });
  checkNode(TS, NodeC, 16, AllChildrenAreNonInterfering, { 1, 4 });
  checkNode(TS, NodeF, 8, AllChildrenAreNonInterfering, { 2, 5 });
  checkNode(TS, NodeG, 8, AllChildrenAreNonInterfering, { 3, 6 });
  checkNode(TS, Node1, 12, AllChildrenAreNonInterfering, { 7 });
  checkNode(TS, Node2, 8, AllChildrenAreNonInterfering, { 8 });
  checkNode(TS, Node3, 4, AllChildrenAreNonInterfering, { 9 });
  checkNode(TS, Node4, 4, AllChildrenAreNonInterfering, { 10 });
}

BOOST_AUTO_TEST_CASE(DeduplicateUnionFields_diamond) {
  dla::LayoutTypeSystem TS;

  // Build TS
  LTSN *NodeA = createRoot(TS);
  LTSN *NodeB = addInheritance(TS, NodeA);
  NodeB->Size = 8;
  /*LTSN *NodeC =*/addInstanceAtOffset(TS, NodeA, /*offset=*/0, /*size=*/8);
  LTSN *NodeD = addInstanceAtOffset(TS, NodeA, /*offset=*/0, /*size=*/0);
  LTSN *NodeE = addInstanceAtOffset(TS, NodeA, /*offset=*/0, /*size=*/0);
  LTSN *NodeF = addInstanceAtOffset(TS, NodeA, /*offset=*/0, /*size=*/0);
  LTSN *NodeG = addInstanceAtOffset(TS, NodeA, /*offset=*/0, /*size=*/0);

  LTSN *NodeH = addInstanceAtOffset(TS, NodeD, /*offset=*/0, /*size=*/8);
  OffsetExpression OE{};
  OE.Offset = 0;
  TS.addInstanceLink(NodeE, NodeH, std::move(OE));

  LTSN *NodeI = addInstanceAtOffset(TS, NodeF, /*offset=*/0, /*size=*/8);
  OffsetExpression OE2{};
  OE2.Offset = 0;
  TS.addInstanceLink(NodeG, NodeI, std::move(OE2));

  // Run steps
  VerifyLog.enable();
  dla::StepManager SM;
  revng_check(SM.addStep<CollapseIdentityAndInheritanceCC>());
  revng_check(SM.addStep<ComputeUpperMemberAccesses>());
  revng_check(SM.addStep<ComputeNonInterferingComponents>());
  revng_check(SM.addStep<DeduplicateUnionFields>());
  revng_check(SM.addStep<ComputeNonInterferingComponents>());

  SM.run(TS);

  // Compress the equivalence classes
  dla::VectEqClasses &Eq = TS.getEqClasses();
  Eq.compress();

  // Check TS
  revng_check(TS.getNumLayouts() == 1);
  checkNode(TS,
            NodeA,
            8,
            AllChildrenAreNonInterfering,
            { 0, 1, 2, 3, 4, 5, 6, 7, 8 });
}

BOOST_AUTO_TEST_CASE(DeduplicateUnionFields_commonNodeSymmetric) {
  dla::LayoutTypeSystem TS;

  // Build TS
  LTSN *NodeUnion = createRoot(TS);
  LTSN *NodeA = addInstanceAtOffset(TS, NodeUnion, /*offset=*/0, /*size=*/0);
  LTSN *NodeB = addInstanceAtOffset(TS, NodeA, /*offset=*/0, /*size=*/8);
  LTSN *NodeC = addInstanceAtOffset(TS, NodeA, /*offset=*/0, /*size=*/0);
  /*LTSN *NodeD =*/addInstanceAtOffset(TS, NodeC, /*offset=*/8, /*size=*/8);

  LTSN *NodeA1 = addInstanceAtOffset(TS, NodeUnion, /*offset=*/0, /*size=*/0);
  LTSN *NodeC1 = addInstanceAtOffset(TS, NodeA1, /*offset=*/0, /*size=*/0);
  LTSN *NodeD1 = addInstanceAtOffset(TS, NodeC1, /*offset=*/8, /*size=*/8);
  OffsetExpression OE{};
  OE.Offset = 0;
  TS.addInstanceLink(NodeA1, NodeB, std::move(OE));

  // Run steps
  VerifyLog.enable();
  dla::StepManager SM;
  revng_check(SM.addStep<CollapseIdentityAndInheritanceCC>());
  revng_check(SM.addStep<ComputeUpperMemberAccesses>());
  revng_check(SM.addStep<ComputeNonInterferingComponents>());
  revng_check(SM.addStep<DeduplicateUnionFields>());
  revng_check(SM.addStep<ComputeNonInterferingComponents>());

  SM.run(TS);

  // Compress the equivalence classes
  dla::VectEqClasses &Eq = TS.getEqClasses();
  Eq.compress();

  // Check TS
  revng_check(TS.getNumLayouts() == 4);
  checkNode(TS, NodeUnion, 16, AllChildrenAreInterfering, { 0, 1, 5 });
  checkNode(TS, NodeB, 8, AllChildrenAreNonInterfering, { 2 });
  checkNode(TS, NodeC1, 16, AllChildrenAreNonInterfering, { 3, 6 });
  checkNode(TS, NodeD1, 8, AllChildrenAreNonInterfering, { 4, 7 });
}

BOOST_AUTO_TEST_CASE(DeduplicateUnionFields_commonNodeAsymmetric) {
  dla::LayoutTypeSystem TS;

  // Build TS
  LTSN *NodeUnion = createRoot(TS);
  LTSN *NodeA = addInstanceAtOffset(TS, NodeUnion, /*offset=*/0, /*size=*/0);
  /*LTSN *NodeB =*/addInstanceAtOffset(TS, NodeA, /*offset=*/0, /*size=*/8);
  LTSN *NodeC = addInstanceAtOffset(TS, NodeA, /*offset=*/0, /*size=*/0);
  LTSN *NodeD = addInstanceAtOffset(TS, NodeC, /*offset=*/4, /*size=*/8);

  LTSN *NodeA1 = addInstanceAtOffset(TS, NodeUnion, /*offset=*/0, /*size=*/0);
  LTSN *NodeC1 = addInstanceAtOffset(TS, NodeA1, /*offset=*/0, /*size=*/0);
  LTSN *NodeD1 = addInstanceAtOffset(TS, NodeC1, /*offset=*/4, /*size=*/8);
  OffsetExpression OE{};
  OE.Offset = 0;
  TS.addInstanceLink(NodeA1, NodeD, std::move(OE));

  // Run steps
  VerifyLog.enable();
  dla::StepManager SM;
  revng_check(SM.addStep<CollapseIdentityAndInheritanceCC>());
  revng_check(SM.addStep<ComputeUpperMemberAccesses>());
  revng_check(SM.addStep<ComputeNonInterferingComponents>());
  revng_check(SM.addStep<DeduplicateUnionFields>());
  revng_check(SM.addStep<ComputeNonInterferingComponents>());

  SM.run(TS);

  // Compress the equivalence classes
  dla::VectEqClasses &Eq = TS.getEqClasses();
  Eq.compress();

  // Check TS
  revng_check(TS.getNumLayouts() == 3);
  checkNode(TS, NodeUnion, 12, AllChildrenAreInterfering, { 0, 1, 5 });
  checkNode(TS, NodeC1, 12, AllChildrenAreNonInterfering, { 3, 6 });
  checkNode(TS, NodeD1, 8, AllChildrenAreNonInterfering, { 2, 4, 7 });
}

BOOST_AUTO_TEST_CASE(DeduplicateUnionFields_commonNodeAsymmetricCollapse) {
  dla::LayoutTypeSystem TS;

  // Build TS
  LTSN *NodeUnion = createRoot(TS);
  LTSN *NodeA = addInstanceAtOffset(TS, NodeUnion, /*offset=*/0, /*size=*/0);
  /*LTSN *NodeB =*/addInstanceAtOffset(TS, NodeA, /*offset=*/0, /*size=*/8);
  LTSN *NodeC = addInstanceAtOffset(TS, NodeA, /*offset=*/0, /*size=*/0);
  LTSN *NodeD = addInstanceAtOffset(TS, NodeC, /*offset=*/0, /*size=*/8);

  LTSN *NodeA1 = addInstanceAtOffset(TS, NodeUnion, /*offset=*/0, /*size=*/0);
  LTSN *NodeC1 = addInstanceAtOffset(TS, NodeA1, /*offset=*/0, /*size=*/0);
  LTSN *NodeD1 = addInstanceAtOffset(TS, NodeC1, /*offset=*/0, /*size=*/8);
  OffsetExpression OE{};
  OE.Offset = 0;
  TS.addInstanceLink(NodeA1, NodeD, std::move(OE));

  // Run steps
  VerifyLog.enable();
  dla::StepManager SM;
  revng_check(SM.addStep<CollapseIdentityAndInheritanceCC>());
  revng_check(SM.addStep<ComputeUpperMemberAccesses>());
  revng_check(SM.addStep<ComputeNonInterferingComponents>());
  revng_check(SM.addStep<DeduplicateUnionFields>());
  revng_check(SM.addStep<ComputeNonInterferingComponents>());

  SM.run(TS);

  // Compress the equivalence classes
  dla::VectEqClasses &Eq = TS.getEqClasses();
  Eq.compress();

  // Check TS
  revng_check(TS.getNumLayouts() == 3);
  checkNode(TS, NodeUnion, 8, AllChildrenAreInterfering, { 0, 1, 5 });
  checkNode(TS, NodeC1, 8, AllChildrenAreNonInterfering, { 3, 6 });
  checkNode(TS, NodeD1, 8, AllChildrenAreNonInterfering, { 2, 4, 7 });
}

// ----------------- Remove Conflicting edges --------------

BOOST_AUTO_TEST_CASE(RemoveConflictingEdges_basic) {
  dla::LayoutTypeSystem TS;

  // Build TS
  LTSN *NodeA = createRoot(TS);
  LTSN *NodeB = addInstanceAtOffset(TS, NodeA, /*offset=*/0, /*size=*/8);
  TS.addInheritanceLink(NodeA, NodeB);

  LTSN *Node1 = createRoot(TS);
  LTSN *Node2 = addInstanceAtOffset(TS, Node1, /*offset=*/4, /*size=*/8);
  TS.addInheritanceLink(Node1, Node2);

  // Run steps
  VerifyLog.enable();
  dla::StepManager SM;
  revng_check(SM.addStep<RemoveConflictingEdges>());

  SM.run(TS);

  // Compress the equivalence classes
  dla::VectEqClasses &Eq = TS.getEqClasses();
  Eq.compress();

  // Check TS
  revng_check(TS.getNumLayouts() == 4);
  revng_check(NodeA->Successors.size() == 1);
  revng_check(Node1->Successors.size() == 2);
}