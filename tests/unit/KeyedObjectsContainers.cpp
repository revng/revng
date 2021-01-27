/// \file KeyedObjectsContainers.cpp
/// \brief Tests for MutableSet and SortedVector

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Boost includes
#define BOOST_TEST_MODULE KeyedObjectsContainers
bool init_unit_test();
#include "boost/test/unit_test.hpp"

// Local libraries includes
#include "revng/ADT/MutableSet.h"
#include "revng/ADT/SortedVector.h"
#include "revng/Support/Assert.h"

// Local includes
#include "TestKeyedObject.h"

template<typename T>
static void assertInsert(T &Set, uint64_t Key, uint64_t Value) {
  auto [It, Success] = Set.insert({ Key, Value });
  revng_check(It->key() == Key and It->value() == Value);
  revng_check(Success);
}

template<typename T>
void testSet() {
  T Set;
  revng_check(Set.empty());

  // Test insertion
  assertInsert(Set, 0x2000, 0xDEAD);
  assertInsert(Set, 0x1000, 0xDEADDEAD);
  assertInsert(Set, 0x3000, 0xDEADDEADDEAD);

  revng_check(not Set.empty());
  revng_check(Set.size() == 3);

  using IterationResultType = std::vector<std::pair<uint64_t, uint64_t>>;
  IterationResultType ExpectedResult{ { 0x1000, 0xDEADDEAD },
                                      { 0x2000, 0xDEAD },
                                      { 0x3000, 0xDEADDEADDEAD } };

  // Test forward iteration
  {
    IterationResultType IterationResult;
    for (const Element &SE : Set)
      IterationResult.emplace_back(SE.key(), SE.value());

    revng_check(IterationResult == ExpectedResult);
  }

  // Test backward iteration
  {
    IterationResultType IterationResult;
    for (const Element &SE : llvm::make_range(Set.rbegin(), Set.rend()))
      IterationResult.emplace_back(SE.key(), SE.value());
    std::reverse(ExpectedResult.begin(), ExpectedResult.end());
    revng_check(IterationResult == ExpectedResult);
    std::reverse(ExpectedResult.begin(), ExpectedResult.end());
  }

  // Test erase
  {
    revng_check(Set.find(0x1500) == Set.end());

    auto [It, Success] = Set.insert({ 0x1500, 0xEEEE });
    revng_check(Success);
    auto Next = Set.erase(It);
    revng_check(Next == Set.find(0x2000));

    revng_check(Set.find(0x1500) == Set.end());

    IterationResultType IterationResult;
    for (const Element &SE : Set)
      IterationResult.emplace_back(SE.key(), SE.value());
    revng_check(IterationResult == ExpectedResult);
  }

  // Test find
  {
    auto It = Set.find(0x1000);
    revng_check(It != Set.end());
    revng_check(It->key() == 0x1000);
    revng_check(It->value() == 0xDEADDEAD);
  }

  // insert vs insert_or_assign
  {
    auto [It, Success] = Set.insert({ 0x1000, 0x5555 });
    revng_check(Set.find(0x1000) == It);
    revng_check(not Success);
    revng_check(Set[0x1000].value() == 0xDEADDEAD);
  }

  {
    auto [It, Success] = Set.insert_or_assign({ 0x1000, 0x6666 });
    revng_check(not Success);
    revng_check(Set[0x1000].value() == 0x6666);
    Set.insert_or_assign({ 0x1000, 0xDEADDEAD });
  }

  // Test operator[] on existing element
  revng_check(Set[0x1000].key() == 0x1000);
  revng_check(Set[0x1000].value() == 0xDEADDEAD);

  // Test operator[] on new element
  revng_check(Set[0x2500].key() == 0x2500);
  revng_check(Set.size() == 4);
  revng_check(Set.find(0x2500) != Set.end());
  revng_check(Set[0x2500].value() == 0);

  // Test elements are mutable
  Set[0x2500].setValue(0x90);
  revng_check(Set[0x2500].value() == 0x90);

  // Test batch_insert
  {
    auto Inserter = Set.batch_insert();
    Inserter.insert({ 0x1000, 0x2222 });
    Inserter.insert({ 0x900, 0x1111 });
  }
  revng_check(Set[0x1000].value() == 0xDEADDEAD);
  revng_check(Set[0x900].value() == 0x1111);

  // Test batch_insert_or_assign
  {
    auto Inserter = Set.batch_insert_or_assign();
    Inserter.insert_or_assign({ 0x1000, 0x2222 });
    Inserter.insert_or_assign({ 0x900, 0x1111 });
  }
  revng_check(Set[0x1000].value() == 0x2222);
  revng_check(Set[0x900].value() == 0x1111);

  // Test clear
  Set.clear();

  revng_check(Set.empty());
  revng_check(Set.size() == 0);
}

BOOST_AUTO_TEST_CASE(TestMutableSet) {
  testSet<MutableSet<Element>>();
}

BOOST_AUTO_TEST_CASE(TestSortedVector) {
  testSet<SortedVector<Element>>();
}

BOOST_AUTO_TEST_CASE(TestUniqueLast) {
  using IntPair = std::pair<int, int>;
  using Vector = std::vector<IntPair>;
  Vector TheVector = {
    { 1, 1 }, { 1, 2 }, { 1, 3 }, { 2, 1 }, { 2, 2 }, { 2, 3 },
  };

  auto Comparator = [](const IntPair &LHS, const IntPair &RHS) {
    return LHS.first == RHS.first;
  };

  TheVector.erase(unique_last(TheVector.begin(), TheVector.end(), Comparator),
                  TheVector.end());

  Vector Expected = { { 1, 3 }, { 2, 3 } };
  revng_check(TheVector == Expected);
}
