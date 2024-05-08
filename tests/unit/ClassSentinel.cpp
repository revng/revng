/// \file ClassSentinel.cpp

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

#define BOOST_TEST_MODULE EarlyFunctionAnalysis
bool init_unit_test();
#include "boost/test/unit_test.hpp"

#include "revng/Support/ClassSentinel.h"
#include "revng/UnitTestHelpers/UnitTestHelpers.h"

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

#if !(defined(__OPTIMIZE__) || defined(__SANITIZE_ADDRESS__) \
      || (defined(__has_feature) && __has_feature(address_sanitizer)))
  BOOST_TEST(DanglingPointer->Sentinel.isDestroyed());
#else
  (void) DanglingPointer;
#endif
}
