/// \file ZipMapIterator.cpp
/// \brief Tests for ZipMapIterator

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <iterator>
#include <map>
#include <set>

// Boost includes
#define BOOST_TEST_MODULE ZipMapIterator
bool init_unit_test();
#include <boost/test/unit_test.hpp>

// Local libraries includes
#include "revng/ADT/SmallMap.h"
#include "revng/ADT/ZipMapIterator.h"
#include "revng/UnitTestHelpers/UnitTestHelpers.h"

using namespace llvm;

template<typename T>
static void
compare(T &Left,
        T &Right,
        std::vector<std::pair<Optional<int>, Optional<int>>> &&Expected) {
  using KE = KeyContainer<T>;
  using pointer = typename KE::pointer;
  std::vector<std::pair<pointer, pointer>> Result;
  std::copy(zipmap_begin(Left, Right),
            zipmap_end(Left, Right),
            std::back_inserter(Result));

  revng_check(Result.size() == Expected.size());

  auto FindLeft = [&Expected, &Left](unsigned I) {
    return Expected[I].first ? KE::find(Left, *Expected[I].first) : nullptr;
  };

  auto FindRight = [&Expected, &Right](unsigned I) {
    return Expected[I].second ? KE::find(Right, *Expected[I].second) : nullptr;
  };

  for (unsigned I = 0; I < Result.size(); I++) {
    revng_check(Result[I] == std::make_pair(FindLeft(I), FindRight(I)));
  }
}

template<typename Map>
void run() {
  Map A, B;
  const Map &ARef = A;
  const Map &BRef = B;

  using KC = KeyContainer<Map>;

  KC::insert(A, 1);
  KC::insert(A, 2);
  KC::insert(A, 4);
  KC::insert(A, 5);
  KC::sort(A);

  KC::insert(B, 1);
  KC::insert(B, 3);
  KC::insert(B, 4);
  KC::insert(B, 7);
  KC::sort(B);

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

  KC::insert(A, 0);
  KC::sort(A);
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

  KC::insert(B, -1);
  KC::sort(B);
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

BOOST_AUTO_TEST_CASE(TestStdMap) {
  run<std::map<int, long>>();
}

BOOST_AUTO_TEST_CASE(TestStdSet) {
  run<std::set<int>>();
}

BOOST_AUTO_TEST_CASE(TestStdVectorPair) {
  run<std::vector<std::pair<const int, long>>>();
}

BOOST_AUTO_TEST_CASE(TestSmallMap) {
  run<SmallMap<int, long, 4>>();
}
