/// \file sentinel.cpp
/// \brief Tests for ClassSentinel

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes

// Boost includes
#define BOOST_TEST_MODULE StackAnalysis
#include <boost/test/unit_test.hpp>

// Local includes
#include "classsentinel.h"

struct TestClass {
  ClassSentinel Sentinel;
};

BOOST_AUTO_TEST_CASE(Sentinel) {
  TestClass *DanglingPointer = nullptr;

  {
      TestClass Instance;
      BOOST_TEST(!Instance.Sentinel.isMoved());
      BOOST_TEST(!Instance.Sentinel.isDestroyed());

      TestClass OtherInstance = std::move(Instance);

      BOOST_TEST(Instance.Sentinel.isMoved());
      BOOST_TEST(!Instance.Sentinel.isDestroyed());

      DanglingPointer = &Instance;
  }

  BOOST_TEST(DanglingPointer->Sentinel.isDestroyed());
}
