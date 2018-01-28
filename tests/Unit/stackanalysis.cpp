/// \file stackanalysis.cpp
/// \brief Tests for StackAnalysis data structures

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes

// Boost includes
#define BOOST_TEST_MODULE StackAnalysis
#include <boost/test/unit_test.hpp>

// Local includes
#include "stackanalysis_impl.h"

using namespace StackAnalysis;

BOOST_TEST_DONT_PRINT_LOG_VALUE(ASID)
BOOST_TEST_DONT_PRINT_LOG_VALUE(ASSlot)
BOOST_TEST_DONT_PRINT_LOG_VALUE(std::vector<ASID>)

static_assert(ASID::RestOfTheStackID == ASID::LastStackID - 1,
              "We expect RST to be right before SP0");

const ASID SP0 = ASID::lastStackID();
const ASID SP1 = ASID(SP0.id() + 1);
const ASID GLB = ASID::globalID();
const ASID CPU = ASID::cpuID();
const ASID RST = ASID::restOfTheStackID();
const ASID DED = ASID::deadStackID();
const ASID Invalid = ASID::invalidID();

using ASVector = std::vector<ASID>;
ASVector toVector(ASSet S) {
  ASVector Result;
  std::copy(S.begin(), S.end(), std::back_inserter(Result));
  return Result;
}

BOOST_AUTO_TEST_CASE(TestAddressSpaceID) {
  // Test ordering
  BOOST_TEST(SP0.lowerThanOrEqual(SP0, Invalid));
  BOOST_TEST(!SP0.greaterThan(SP0, Invalid));
  BOOST_TEST(SP0.greaterThan(SP1, Invalid));
  BOOST_TEST(SP0.greaterThan(GLB, Invalid));

  BOOST_TEST(SP0.greaterThan(RST, Invalid));
  BOOST_TEST(RST.greaterThan(SP0, Invalid));
  BOOST_TEST(SP1.lowerThanOrEqual(RST, SP0));
  BOOST_TEST(SP0.greaterThan(RST, SP0));

  // Test moving and capping
  BOOST_TEST(SP0.getCallerStack(Invalid.id()) == SP1);
  BOOST_TEST(SP0.getCallerStack(SP0.id()) == RST);

  BOOST_TEST(SP0.shiftAddressSpaces(+1) == SP1);
  BOOST_TEST(SP1.shiftAddressSpaces(-1) == SP0);
  BOOST_TEST(SP0.shiftAddressSpaces(-1) == DED);
  BOOST_TEST(SP1.shiftAddressSpaces(-2) == DED);
  BOOST_TEST(CPU.shiftAddressSpaces(-1) == CPU);
  BOOST_TEST(CPU.shiftAddressSpaces(+1) == CPU);
  BOOST_TEST(GLB.shiftAddressSpaces(-1) == GLB);
  BOOST_TEST(GLB.shiftAddressSpaces(+1) == GLB);

  BOOST_TEST(GLB.cap(SP0.id()) == GLB);
  BOOST_TEST(SP0.cap(SP0.id()) == SP0);
  BOOST_TEST(SP1.cap(SP0.id()) == RST);
}

BOOST_AUTO_TEST_CASE(TestASFunction) {
  ASID GLB = ASID::globalID();
  ASID CPU = ASID::cpuID();

  // Test membership and adding element
  ASSet ASF = ASSet::singleElement(CPU);
  BOOST_TEST(toVector(ASF) == (ASVector { CPU }));
  BOOST_TEST(ASF[CPU] == true);
  BOOST_TEST(ASF[GLB] == false);

  ASF.add(GLB);
  BOOST_TEST(toVector(ASF) == (ASVector { GLB, CPU }));
  BOOST_TEST(ASF[CPU] == true);
  BOOST_TEST(ASF[GLB] == true);

  ASF.add(RST);
  BOOST_TEST(toVector(ASF) == (ASVector { GLB, CPU, RST }));
  BOOST_TEST(ASF.contains(SP1, SP0));
  BOOST_TEST(!ASF.contains(SP1, SP1));

  // Test lowerThanOrEqual
  ASSet Other = ASF;
  Other.add(SP0);
  BOOST_TEST(toVector(ASF) == (ASVector { GLB, CPU, RST }));
  BOOST_TEST(toVector(Other) == (ASVector { GLB, CPU, RST, SP0 }));
  BOOST_TEST(ASF.lowerThanOrEqual(Other, Invalid));

  ASF.add(SP1);
  BOOST_TEST(toVector(ASF) == (ASVector { GLB, CPU, RST, SP1 }));
  BOOST_TEST(ASF.greaterThan(Other, Invalid));
  BOOST_TEST(ASF.lowerThanOrEqual(Other, SP0));

  // Test combine
  ASSet Tmp;
  Tmp = Other;
  Tmp.combine(ASF, Invalid);
  BOOST_TEST(toVector(Tmp) == (ASVector { GLB, CPU, RST, SP0, SP1 }));

  Tmp = Other;
  Tmp.combine(ASF, SP0);
  BOOST_TEST(toVector(Tmp) == (ASVector { GLB, CPU, RST, SP0 }));

  ASSet A = ASSet::singleElement(CPU);
  ASSet B = ASSet::singleElement(SP1);
  Tmp = A;
  Tmp.combine(B, SP0);
  BOOST_TEST(toVector(Tmp) == (ASVector { CPU, RST }));

  Tmp = A;
  Tmp.combine(B, SP1);
  BOOST_TEST(toVector(Tmp) == (ASVector { CPU, SP1 }));

  // Test drop
  Tmp = A;
  Tmp.drop(CPU);
  BOOST_TEST(toVector(Tmp) == (ASVector { }));
  BOOST_TEST(Tmp.empty());

  // Test shiftAddressSpaces
  Tmp = Other;
  BOOST_TEST(toVector(Tmp) == (ASVector { GLB, CPU, RST, SP0 }));
  Tmp.shiftAddressSpaces(+1);
  BOOST_TEST(toVector(Tmp) == (ASVector { GLB, CPU, RST, SP1 }));
  Tmp.shiftAddressSpaces(-1);
  BOOST_TEST(toVector(Tmp) == (ASVector { GLB, CPU, RST, SP0 }));
  Tmp.shiftAddressSpaces(-1);
  BOOST_TEST(toVector(Tmp) == (ASVector { DED, GLB, CPU, RST }));

  // Test capAddressSpaces
  Tmp = Other;
  BOOST_TEST(toVector(Tmp) == (ASVector { GLB, CPU, RST, SP0 }));
  Tmp.capAddressSpaces(RST.id());
  BOOST_TEST(toVector(Tmp) == (ASVector { GLB, CPU, RST }));
  Tmp.add(SP1);
  BOOST_TEST(toVector(Tmp) == (ASVector { GLB, CPU, RST, SP1 }));
  Tmp.capAddressSpaces(SP0.id());
  BOOST_TEST(toVector(Tmp) == (ASVector { GLB, CPU, RST }));
}

BOOST_AUTO_TEST_CASE(TestASSlot) {
  // Test comparisons
  ASSlot SP1Slot(SP1, 0);
  ASSlot RSTSlot(RST, 0);
  BOOST_TEST(SP1Slot == SP1Slot);
  BOOST_TEST(SP1Slot != RSTSlot);
  BOOST_TEST(SP1Slot.greaterThan(RSTSlot, Invalid));
  BOOST_TEST(SP1Slot.lowerThanOrEqual(RSTSlot, SP0));

  // Test addition and masking
  SP1Slot.add(-5);
  BOOST_TEST(SP1Slot.offset() == -5);
  SP1Slot.add(+10);
  BOOST_TEST(SP1Slot.offset() == +5);
  SP1Slot.mask(1);
  BOOST_TEST(SP1Slot.offset() == +1);

  // Test flattening
  BOOST_TEST(toVector(SP1Slot.flatten()) == (ASVector { SP1 }));

  // Test usage in a map
  std::map<ASSlot, int> Map;
  Map[SP1Slot] = 0;
  BOOST_TEST(Map.count(SP1Slot) != 0U);
}
