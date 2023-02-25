/// \file ZipMapIterator.cpp
/// \brief Tests for ZipMapIterator

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <iterator>
#include <map>
#include <set>

#define BOOST_TEST_MODULE ZipMapIterator
bool init_unit_test();
#include "boost/test/unit_test.hpp"

#include "revng/ADT/MutableSet.h"
#include "revng/ADT/STLExtras.h"
#include "revng/ADT/SmallMap.h"
#include "revng/ADT/SortedVector.h"
#include "revng/ADT/ZipMapIterator.h"
#include "revng/UnitTestHelpers/UnitTestHelpers.h"

// Local includes
#include "TestKeyedObject.h"

using namespace llvm;

template<typename T, typename = void>
struct KeyContainer {};

template<MapLike T>
struct KeyContainer<T> {
  using key_type = typename T::key_type;

  static void insert(T &Container, typename T::key_type Key) {
    Container.insert({ Key, typename T::mapped_type() });
  }

  static auto find(T &Container, typename T::key_type &Key) {
    return &*Container.find(Key);
  }

  static void sort(T &) {}
};

template<typename T>
concept SetOrKOC = SetLike<T> or KeyedObjectContainer<T>;

template<SetOrKOC T>
struct KeyContainer<T> {
  using key_type = const typename T::key_type;

  static void insert(T &Container, key_type Key) { Container.insert(Key); }

  static auto find(T &Container, key_type Key) { return &*Container.find(Key); }

  static void sort(T &) {}
};

template<VectorOfPairs T>
struct KeyContainer<T> {
  using key_type = typename T::value_type::first_type;
  using value_type = std::conditional_t<std::is_const<T>::value,
                                        const typename T::value_type,
                                        typename T::value_type>;
  using mapped_type = typename value_type::second_type;

  static void insert(T &Container, key_type Key) {
    Container.push_back({ Key, mapped_type() });
  }

  static auto find(T &Container, key_type Key) {
    auto Condition = [Key](const value_type &Value) {
      return Value.first == Key;
    };
    return &*std::find_if(Container.begin(), Container.end(), Condition);
  }

  static void sort(T &Container) {
    static_assert(not std::is_const<T>::value, "");
    using non_const_value_type = std::pair<std::remove_const_t<key_type>,
                                           mapped_type>;
    auto Less = [](const non_const_value_type &This,
                   const non_const_value_type &Other) {
      return This.first < Other.first;
    };
    using vector = std::vector<non_const_value_type>;
    auto &NonConst = *reinterpret_cast<vector *>(&Container);
    std::sort(NonConst.begin(), NonConst.end(), Less);
  }
};

template<typename LeftType, typename RightType>
static void
compare(LeftType &Left,
        RightType &Right,
        std::vector<std::pair<Optional<int>, Optional<int>>> &&Expected) {
  using LeftKE = KeyContainer<LeftType>;
  using RightKE = KeyContainer<RightType>;
  using left_pointer = element_pointer_t<LeftType>;
  using right_pointer = element_pointer_t<RightType>;
  std::vector<std::pair<left_pointer, right_pointer>> Result;
  std::copy(zipmap_begin(Left, Right),
            zipmap_end(Left, Right),
            std::back_inserter(Result));

  revng_check(Result.size() == Expected.size());

  auto FindLeft = [&Expected, &Left](unsigned I) {
    return Expected[I].first ? LeftKE::find(Left, *Expected[I].first) : nullptr;
  };

  auto FindRight = [&Expected, &Right](unsigned I) {
    return Expected[I].second ? RightKE::find(Right, *Expected[I].second) :
                                nullptr;
  };

  for (unsigned I = 0; I < Result.size(); I++) {
    std::pair<left_pointer, right_pointer> X{ FindLeft(I), FindRight(I) };
    revng_check(Result[I] == X);
  }
}

template<typename LeftMap, typename RightMap>
void run() {
  LeftMap A;
  RightMap B;
  LeftMap &ARef = A;
  RightMap &BRef = B;

  using LeftKC = KeyContainer<LeftMap>;
  using RightKC = KeyContainer<RightMap>;

  LeftKC::insert(A, 1);
  LeftKC::insert(A, 2);
  LeftKC::insert(A, 4);
  LeftKC::insert(A, 5);
  LeftKC::sort(A);

  RightKC::insert(B, 1);
  RightKC::insert(B, 3);
  RightKC::insert(B, 4);
  RightKC::insert(B, 7);
  RightKC::sort(B);

  compare(ARef,
          BRef,
          {
            { { 1 }, { 1 } },
            { { 2 }, {} },
            { {}, { 3 } },
            { { 4 }, { 4 } },
            { { 5 }, {} },
            { {}, { 7 } },
          });

  LeftKC::insert(A, 0);
  LeftKC::sort(A);
  compare(A,
          B,
          {
            { { 0 }, {} },
            { { 1 }, { 1 } },
            { { 2 }, {} },
            { {}, { 3 } },
            { { 4 }, { 4 } },
            { { 5 }, {} },
            { {}, { 7 } },
          });

  RightKC::insert(B, -1);
  RightKC::sort(B);
  compare(A,
          B,
          {
            { {}, { -1 } },
            { { 0 }, {} },
            { { 1 }, { 1 } },
            { { 2 }, {} },
            { {}, { 3 } },
            { { 4 }, { 4 } },
            { { 5 }, {} },
            { {}, { 7 } },
          });
}

template<typename Map>
void runSame() {
  run<Map, Map>();
}

BOOST_AUTO_TEST_CASE(TestStdMap) {
  runSame<std::map<int, long>>();
}

BOOST_AUTO_TEST_CASE(TestStdSet) {
  runSame<std::set<int>>();
}

BOOST_AUTO_TEST_CASE(TestMutableSet) {
  runSame<MutableSet<int>>();
}

BOOST_AUTO_TEST_CASE(TestSortedVector) {
  runSame<SortedVector<int>>();
}

BOOST_AUTO_TEST_CASE(TestStdVectorPair) {
  runSame<std::vector<std::pair<const int, long>>>();
}

BOOST_AUTO_TEST_CASE(TestSmallMap) {
  runSame<SmallMap<int, long, 4>>();
}

BOOST_AUTO_TEST_CASE(TestSortedVectorAndMutableSet) {
  run<SortedVector<int>, MutableSet<int>>();
}
