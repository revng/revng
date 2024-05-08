/// \file KeyedObjectsContainers.cpp
/// Tests for MutableSet and SortedVector.

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

#define BOOST_TEST_MODULE KeyedObjectsContainers
bool init_unit_test();
#include "boost/test/unit_test.hpp"

#include "revng/ADT/MutableSet.h"
#include "revng/ADT/SortedVector.h"
#include "revng/ADT/TrackingContainer.h"
#include "revng/Support/Assert.h"

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

  // Test copy/move constructor/assign
  {
    T Original;
    T Copy(Original);
    Copy = Original;
    Copy = std::move(Original);

    T Moved(std::move(Copy));
  }

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
    revng_check(!Set.contains(0x1500));

    auto [It, Success] = Set.insert({ 0x1500, 0xEEEE });
    revng_check(Success);
    auto Next = Set.erase(It);
    revng_check(Next == Set.find(0x2000));

    revng_check(!Set.contains(0x1500));

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
  revng_check(Set.contains(0x2500));
  revng_check(Set[0x2500].value() == 0);

  // Test elements are mutable
  Set[0x2500].setValue(0x90);
  revng_check(Set[0x2500].value() == 0x90);

  // Test batch_insert
  {
    auto Inserter = Set.batch_insert();
    Inserter.insert({ 0x1100, 0x2222 });
    Inserter.insert({ 0x900, 0x1111 });
  }
  revng_check(Set[0x1000].value() == 0xDEADDEAD);
  revng_check(Set[0x900].value() == 0x1111);

  // Test batch_insert_or_assign
  {
    auto Inserter = Set.batch_insert_or_assign();
    Inserter.insert_or_assign({ 0x1000, 0x2222 });
    Inserter.insert_or_assign({ 0x900, 0x3333 });
  }
  revng_check(Set[0x1000].value() == 0x2222);
  revng_check(Set[0x1100].value() == 0x2222);
  revng_check(Set[0x900].value() == 0x3333);

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

template<typename T>
bool isSerializationStable(T &&Original) {
  std::string Buffer;

  {
    llvm::raw_string_ostream Stream(Buffer);
    llvm::yaml::Output YAMLOutput(Stream);
    YAMLOutput << Original;
  }

  T Deserialized;
  llvm::yaml::Input YAMLInput(Buffer);
  YAMLInput >> Deserialized;

  return Original == Deserialized;
}

BOOST_AUTO_TEST_CASE(TestYAMLSerializationStability) {
  revng_check(isSerializationStable(SortedVector<int>{ 4, 19, 7 }));
  revng_check(isSerializationStable(MutableSet<int>{ 4, 19, 7 }));

  SortedVector<Element> TestSV{ { 1, 2 }, { 2, 3 } };
  revng_check(isSerializationStable(std::move(TestSV)));

  MutableSet<Element> TestMS{ { 1, 2 }, { 2, 3 } };
  revng_check(isSerializationStable(std::move(TestMS)));
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

template<template<typename...> class T>
static void testAt() {
  revng::TrackingContainer<T<int>> Vector;
  int One(1);
  Vector.insert(One);
  Vector.clearTracking();
  const auto &Reference = Vector;
  Reference.at(One);

  auto TrackingResult = Reference.getTrackingResult();
  revng_check(TrackingResult.InspectedKeys.size() == 1);
  revng_check(TrackingResult.InspectedKeys.contains(1));
}

BOOST_AUTO_TEST_CASE(TrackingContainerVectorAt) {
  testAt<SortedVector>();
  testAt<MutableSet>();
}

template<template<typename...> class T>
static void testVectorCount() {
  revng::TrackingContainer<T<int>> Vector;
  Vector.insert(1);
  Vector.clearTracking();
  const auto &Reference = Vector;
  Reference.count(2);

  auto TrackingResult = Reference.getTrackingResult();
  revng_check(TrackingResult.InspectedKeys.size() == 1);
  revng_check(not TrackingResult.Exact);
  revng_check(TrackingResult.InspectedKeys.contains(2));
}

BOOST_AUTO_TEST_CASE(TrackingContainerVectorCount) {
  testVectorCount<SortedVector>();
  testVectorCount<MutableSet>();
}

template<template<typename...> class T>
static void testTryGet() {
  revng::TrackingContainer<T<int>> Vector;
  Vector.insert(1);
  Vector.clearTracking();
  const auto &Reference = Vector;
  const int *Result = nullptr;
  Result = Reference.tryGet(2);
  revng_check(Result == nullptr);
  Result = Reference.tryGet(1);
  revng_check(Result != nullptr && *Result == 1);

  auto TrackingResult = Reference.getTrackingResult();
  revng_check(TrackingResult.InspectedKeys == std::set<int>({ 1, 2 }));
  revng_check(not TrackingResult.Exact);
}

BOOST_AUTO_TEST_CASE(TrackingContainerVectorTryGet) {
  testTryGet<SortedVector>();
  testTryGet<MutableSet>();
}

template<template<typename...> class T>
static void testBeginEnd() {
  revng::TrackingContainer<SortedVector<int>> Vector;
  Vector.insert(1);
  Vector.clearTracking();
  const auto &Reference = Vector;
  llvm::find(Reference, 1);

  auto TrackingResult = Reference.getTrackingResult();
  revng_check(TrackingResult.InspectedKeys.size() == 0);
  revng_check(TrackingResult.Exact);
}

BOOST_AUTO_TEST_CASE(TrackingContainerVectorBeginEnd) {
  testBeginEnd<SortedVector>();
  testBeginEnd<MutableSet>();
}

template<template<typename...> class T>
static void testSetCount() {
  revng::TrackingContainer<SortedVector<int>> Vector;
  Vector.insert(1);
  Vector.clearTracking();
  const auto &Reference = Vector;
  Reference.count(1);

  auto TrackingResult = Reference.getTrackingResult();
  revng_check(TrackingResult.InspectedKeys == std::set<int>({ 1 }));
  revng_check(not TrackingResult.Exact);
}

BOOST_AUTO_TEST_CASE(TrackingContainerSetCount) {
  testSetCount<SortedVector>();
  testSetCount<MutableSet>();
}

template<template<typename...> class T>
static void testPush() {
  revng::TrackingContainer<SortedVector<int>> Vector;
  Vector.insert(1);
  Vector.clearTracking();
  const auto &Reference = Vector;
  Reference.count(1);

  Reference.trackingPush();

  auto TrackingResult = Reference.getTrackingResult();
  revng_check(TrackingResult.InspectedKeys == std::set<int>({ 1 }));
  revng_check(not TrackingResult.Exact);

  Reference.trackingPop();

  TrackingResult = Reference.getTrackingResult();
  revng_check(TrackingResult.InspectedKeys == std::set<int>({ 1 }));
  revng_check(not TrackingResult.Exact);

  Reference.trackingPop();

  TrackingResult = Reference.getTrackingResult();
  revng_check(TrackingResult.InspectedKeys.empty());
  revng_check(not TrackingResult.Exact);
}

BOOST_AUTO_TEST_CASE(TrackingContainerPush) {
  testPush<SortedVector>();
  testPush<MutableSet>();
}

static_assert(KeyedObjectContainer<TrackingSortedVector<int>>);
static_assert(KeyedObjectContainer<revng::TrackingContainer<MutableSet<int>>>);
