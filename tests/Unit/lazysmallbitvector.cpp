/// \file lazysmallbitvector.cpp
/// \brief Tests for LazySmallBitVector

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <algorithm>
#include <cstdint>
#include <iterator>
#include <vector>

// Boost includes
#define BOOST_TEST_MODULE LazySmallBitVector
bool init_unit_test();
#include <boost/test/unit_test.hpp>

// Local libraries includes
#include "revng/ADT/LazySmallBitVector.h"

static const unsigned FirstLargeBit = sizeof(uintptr_t) * CHAR_BIT;

// The following types don't have a << operator
BOOST_TEST_DONT_PRINT_LOG_VALUE(LazySmallBitVector)
BOOST_TEST_DONT_PRINT_LOG_VALUE(std::vector<unsigned>)

BOOST_AUTO_TEST_CASE(TestEmpty) {
  LazySmallBitVector Empty;
  BOOST_TEST(Empty.isZero());
}

BOOST_AUTO_TEST_CASE(TestSetAndRead) {
  LazySmallBitVector BitVector;

  // Test small
  BitVector.set(0);
  BOOST_TEST(BitVector.requiredBits() == 1U);
  BOOST_TEST(BitVector[0] == true);
  BOOST_TEST(BitVector[1] == false);
  BOOST_TEST(BitVector[1000] == false);
  BOOST_TEST(BitVector.isSmall());
  BOOST_TEST(!BitVector.isZero());

  BitVector.unset(0);
  BOOST_TEST(BitVector.isZero());
  BitVector.set(0);

  // Test large
  BitVector.set(FirstLargeBit);
  BOOST_TEST(BitVector.requiredBits() == FirstLargeBit + 1);
  BOOST_TEST(BitVector[0] == true);
  BOOST_TEST(BitVector[1] == false);
  BOOST_TEST(BitVector[1000] == false);
  BOOST_TEST(BitVector[FirstLargeBit] == true);
  BOOST_TEST(!BitVector.isSmall());
  BOOST_TEST(!BitVector.isZero());

  BitVector.unset(FirstLargeBit);
  BOOST_TEST(BitVector[FirstLargeBit] == false);
}

BOOST_AUTO_TEST_CASE(TestZero) {
  LazySmallBitVector BitVector;
  BitVector.set(0);
  BitVector.set(1);

  BitVector.zero(0, 1);
  BOOST_TEST(BitVector[0] == false);
  BOOST_TEST(!BitVector.isZero());

  BitVector.zero();
  BOOST_TEST(BitVector.isZero());
}

BOOST_AUTO_TEST_CASE(TestEquality) {
  LazySmallBitVector A;
  LazySmallBitVector B;

  // Test small
  A.set(1);
  BOOST_TEST(A != B);
  B.set(1);
  BOOST_REQUIRE_EQUAL(A, B);

  // Test large
  A.set(FirstLargeBit);
  BOOST_TEST(A != B);
  B.set(FirstLargeBit);
  BOOST_REQUIRE_EQUAL(A, B);

  // Test zero out
  A.zero();
  BOOST_TEST(A != B);
  B.zero();
  BOOST_REQUIRE_EQUAL(A, B);
}

BOOST_AUTO_TEST_CASE(TestBitwiseOperators) {
  // First loop iteration tests the small implementation, the second the large
  // one
  for (unsigned Start = 0; Start <= FirstLargeBit; Start += FirstLargeBit) {
    LazySmallBitVector A;
    LazySmallBitVector B;

    // Test and
    A.set(Start + 0);
    B = A;
    B.set(Start + 1);

    A &= B;
    BOOST_TEST(A.requiredBits() == Start + 1);
    BOOST_TEST(A[Start + 0] == true);

    A.zero();
    B.zero();

    A.set(Start + 0);
    B = A;
    B.set(Start + 1);

    B &= A;
    BOOST_REQUIRE_EQUAL(A, B);

    A.zero();
    B.zero();

    // Test or
    A.set(Start + 0);
    B = A;
    B.set(Start + 1);

    A |= B;
    BOOST_REQUIRE_EQUAL(A, B);

    A.zero();
    B.zero();

    // Test xor
    A.set(Start + 0);
    B = A;
    B.set(Start + 1);
    B.set(Start + 2);

    A ^= B;
    BOOST_TEST(A[Start + 0] == false);
    BOOST_TEST(A[Start + 1] == true);
    BOOST_TEST(A[Start + 2] == true);

    A.zero();
    B.zero();

    // Test shifts
    A.set(Start + 0);
    A <<= 1;
    BOOST_TEST(A[Start + 0] == false);
    BOOST_TEST(A[Start + 1] == true);
    A >>= 1;
    BOOST_TEST(A[Start + 0] == true);
    BOOST_TEST(A[Start + 1] == false);
  }
}

BOOST_AUTO_TEST_CASE(TestCopy) {
  for (unsigned Start = 0; Start <= FirstLargeBit; Start += FirstLargeBit) {
    LazySmallBitVector A;
    A.set(Start + 1);

    LazySmallBitVector B(A);
    BOOST_TEST(B[Start + 1] == true);

    LazySmallBitVector C = A;
    BOOST_TEST(C[Start + 1] == true);
  }
}

#include <boost/iterator/transform_iterator.hpp>

BOOST_AUTO_TEST_CASE(TestIterator) {
  std::vector<unsigned> Results;

  // Test small implementation
  LazySmallBitVector A;
  A.set(0);
  A.set(16);

  std::copy(A.begin(), A.end(), std::back_inserter(Results));
  BOOST_REQUIRE_EQUAL(Results, (std::vector<unsigned>{ 0, 16 }));
  Results.clear();

  // Test empty
  A.zero();
  std::copy(A.begin(), A.end(), std::back_inserter(Results));
  BOOST_TEST(Results.empty());

  // Test large implementation
  A.set(0);
  A.set(1000);
  A.set(16);

  std::copy(A.begin(), A.end(), std::back_inserter(Results));
  BOOST_REQUIRE_EQUAL(Results, (std::vector<unsigned>{ 0, 16, 1000 }));
}
