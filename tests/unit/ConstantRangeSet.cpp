/// \file ConstantrangeSet.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#define BOOST_TEST_MODULE ConstantrangeSet
bool init_unit_test();
#include "boost/test/unit_test.hpp"

#include "revng/ADT/ConstantRangeSet.h"

BOOST_AUTO_TEST_CASE(TestEnumerate) {
  using CRS = ConstantRangeSet;

  auto Range = [](uint32_t Start, uint32_t End) {
    return CRS({ { 32, Start }, { 32, End } });
  };

  revng_check(Range(0xFFFFFFFF, 0).size() == 1);
  llvm::ConstantRange AlmostAll({ 32, 0 }, { 32, 0xFFFFFFFF });
  revng_check(not AlmostAll.isFullSet());
  revng_check(not AlmostAll.isEmptySet());
  revng_check(not AlmostAll.isWrappedSet());
  revng_check(Range(0, 0xFFFFFFFF).size().getLimitedValue() == 0xFFFFFFFF);

  llvm::ConstantRange ZeroFive({ 4, 0 }, { 4, 5 });

  std::pair<CRS, std::vector<uint64_t>> Ranges[] = {
    { CRS(), {} },
    { CRS(4, true), { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 } },
    { CRS(4, false), {} },
    { Range(10, 20), { 10, 11, 12, 13, 14, 15, 16, 17, 18, 19 } },
    { Range(10, 20).unionWith(Range(30, 40)),
      { 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
        30, 31, 32, 33, 34, 35, 36, 37, 38, 39 } },
    { CRS({ { 8, 250 }, { 8, 5 } }),
      { 0, 1, 2, 3, 4, 250, 251, 252, 253, 254, 255 } },
    { Range(10, 20).unionWith(Range(30, 40)).unionWith(Range(15, 35)),
      { 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
        25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39 } },
    { CRS(ZeroFive), { 0, 1, 2, 3, 4 } },
    { CRS(ZeroFive.inverse()), { 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 } },
    { CRS(ZeroFive.inverse()).intersectWith(CRS({ { 4, 10 }, { 4, 13 } })),
      { 10, 11, 12 } },
    { Range(0xFFFFFFFF, 0), { 0xFFFFFFFF } }
  };

  for (auto &P : Ranges) {
    const ConstantRangeSet &Range = P.first;
    const std::vector<uint64_t> &Expected = P.second;

    Range.dump();
    dbg << ": ";

    unsigned I = 0;
    auto It = Range.begin();
    auto End = Range.end();
    while (It != End) {
      uint64_t Value = (*It).getLimitedValue();
      revng_check(Value == Expected[I]);
      dbg << " " << Value;
      ++It;
      I++;
    }

    revng_check(I == Expected.size());

    dbg << "\n";
  }
}
