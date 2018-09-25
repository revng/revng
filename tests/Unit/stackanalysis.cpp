/// \file stackanalysis.cpp
/// \brief Tests for StackAnalysis data structures

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Boost includes
#define BOOST_TEST_MODULE StackAnalysis
bool init_unit_test();
#include <boost/test/unit_test.hpp>

// Local includes
#include "Intraprocedural.h"

using namespace StackAnalysis;

BOOST_TEST_DONT_PRINT_LOG_VALUE(ASID)
BOOST_TEST_DONT_PRINT_LOG_VALUE(ASSlot)
BOOST_TEST_DONT_PRINT_LOG_VALUE(std::vector<ASID>)

const ASID SP0 = ASID::stackID();
const ASID GLB = ASID::globalID();
const ASID CPU = ASID::cpuID();
const ASID Invalid = ASID::invalidID();

using ASVector = std::vector<ASID>;

BOOST_AUTO_TEST_CASE(TestAddressSpaceID) {
  // Test ordering
  BOOST_TEST(SP0.lowerThanOrEqual(SP0));
  BOOST_TEST(not SP0.greaterThan(SP0));
  BOOST_TEST(SP0.greaterThan(GLB));
}

BOOST_AUTO_TEST_CASE(TestASSlot) {
  // Test comparisons
  ASSlot SP0Slot = ASSlot::create(SP0, 0);
  ASSlot CPUSlot = ASSlot::create(CPU, 0);
  BOOST_TEST(SP0Slot == SP0Slot);
  BOOST_TEST(SP0Slot != CPUSlot);
  BOOST_TEST(SP0Slot.greaterThan(CPUSlot));
  BOOST_TEST(not SP0Slot.lowerThanOrEqual(CPUSlot));

  // Test addition and masking
  SP0Slot.add(-5);
  BOOST_TEST(SP0Slot.offset() == -5);
  SP0Slot.add(+10);
  BOOST_TEST(SP0Slot.offset() == +5);
  SP0Slot.mask(1);
  BOOST_TEST(SP0Slot.offset() == +1);

  // Test usage in a map
  std::map<ASSlot, int> Map;
  Map[SP0Slot] = 0;
  BOOST_TEST(Map.count(SP0Slot) != 0U);
}
