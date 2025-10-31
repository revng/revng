/// \file RecursiveCoroutine2.cpp
/// Tests `RecursiveCoroutine`

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#define BOOST_TEST_MODULE RecursiveCoroutine2
bool init_unit_test();
#include "boost/test/unit_test.hpp"

#include "revng/ADT/RecursiveCoroutine.h"

namespace {

class InstanceCounter {
  int *Counter;

public:
  InstanceCounter(int &Counter) : Counter(&Counter) { ++*this->Counter; }

  InstanceCounter(const InstanceCounter &Other) : Counter(Other.Counter) {
    ++*this->Counter;
  }

  InstanceCounter &operator=(const InstanceCounter &Other) {
    --*this->Counter;
    this->Counter = Other.Counter;
    ++*this->Counter;
    return *this;
  }

  ~InstanceCounter() { --*this->Counter; }
};

class TestMutex {
  bool IsLocked = false;

public:
  void lock() {
    revng_check(not IsLocked);
    IsLocked = true;
  }

  void unlock() {
    revng_check(IsLocked);
    IsLocked = false;
  }
};

} // namespace

BOOST_AUTO_TEST_CASE(RecursiveCoroutineEvaluationOrder1) {
  TestMutex Mutex;
  int GuardedValue = 1;

  auto F = [&]() -> RecursiveCoroutine<int> {
    std::lock_guard Lock(Mutex);
    rc_return GuardedValue++;
  };

  auto G = [&](int Value) -> std::pair<int, int> {
    std::lock_guard Lock(Mutex);
    return { Value, GuardedValue++ };
  };

  std::pair<int, int> P = G(F());
  revng_check(P.first == 1);
  revng_check(P.second == 2);
}

BOOST_AUTO_TEST_CASE(RecursiveCoroutineEvaluationOrder2) {
  TestMutex Mutex;
  int GuardedValue = 1;

  auto F = [&]() -> RecursiveCoroutine<int> {
    std::lock_guard Lock(Mutex);
    rc_return GuardedValue++;
  };

  auto G = [&](int Value) -> RecursiveCoroutine<std::pair<int, int>> {
    std::lock_guard Lock(Mutex);
    rc_return{ Value, GuardedValue++ };
  };

  auto H = [&]() -> RecursiveCoroutine<std::pair<int, int>> {
    rc_return rc_recur G(rc_recur F());
  };

  std::pair<int, int> P = H();
  revng_check(P.first == 1);
  revng_check(P.second == 2);
}

BOOST_AUTO_TEST_CASE(RecursiveCoroutineEvaluationOrder3) {
  int Counter = 0;

  auto F = [&]() -> RecursiveCoroutine<InstanceCounter> { rc_return Counter; };

  auto G = [&](InstanceCounter) { revng_check(Counter == 1); };

  G(F());
}
