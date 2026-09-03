//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#define BOOST_TEST_MODULE Model
bool init_unit_test();
#include "boost/test/unit_test.hpp"

#include "revng/Model/Binary.h"
#include "revng/Model/PrimitiveType.h"

// Each test case is a pair of numbers. The first number is arbitrary,
// the second is the size of the smallest primitive that can hold it.

struct Case {
  uint64_t From;
  uint64_t To;
};

static constexpr std::array<Case, 26> NumericPrimitives = {
  // < 8
  Case(1, 1),
  Case(2, 2),
  Case(3, 4),
  Case(4, 4),
  Case(5, 8),
  Case(6, 8),
  Case(7, 8),
  Case(8, 8),

  // 16
  Case(9, 16),
  Case(10, 16),
  Case(11, 16),
  Case(12, 16),
  Case(13, 16),
  Case(14, 16),
  Case(15, 16),
  Case(16, 16),

  // 32
  Case(17, 32),
  Case(23, 32),
  Case(29, 32),
  Case(31, 32),
  Case(32, 32),

  // 64
  Case(33, 64),
  Case(41, 64),
  Case(57, 64),
  Case(63, 64),
  Case(64, 64),

  // Anything bigger asserts, as we do not support such primitives!
};

static constexpr std::array<Case, 16> FloatPrimitives = {
  // < 8
  Case(1, 2),
  Case(2, 2),
  Case(3, 4),
  Case(4, 4),
  Case(5, 8),
  Case(6, 8),
  Case(7, 8),
  Case(8, 8),

  // 10 and 12
  Case(9, 10),
  Case(10, 10),
  Case(11, 12),
  Case(12, 12),

  // 16
  Case(13, 16),
  Case(14, 16),
  Case(15, 16),
  Case(16, 16),

  // Anything bigger asserts, as we do not support such primitives!
};

static constexpr std::array<Case, 26> GenericPrimitives = {
  // < 8
  Case(1, 1),
  Case(2, 2),
  Case(3, 4),
  Case(4, 4),
  Case(5, 8),
  Case(6, 8),
  Case(7, 8),
  Case(8, 8),

  // 10 and 12
  Case(9, 10),
  Case(10, 10),
  Case(11, 12),
  Case(12, 12),

  // 16
  Case(13, 16),
  Case(14, 16),
  Case(15, 16),
  Case(16, 16),

  // 32
  Case(17, 32),
  Case(23, 32),
  Case(29, 32),
  Case(31, 32),
  Case(32, 32),

  // 64
  Case(33, 64),
  Case(41, 64),
  Case(57, 64),
  Case(63, 64),
  Case(64, 64),

  // Anything bigger asserts, as we do not support such primitives!
};

static void testPrimitiveImpl(const Case &Test,
                              model::PrimitiveKind::Values Kind) {
  auto P = model::PrimitiveType::makeBigEnoughFor(Kind, Test.From);
  if (Test.To != *P->size()) {
    std::string Error = "Creating a primitive big enough to hold a "
                        + std::to_string(Test.From) + "-byte `" + toString(Kind)
                        + "` produced a " + std::to_string(*P->size())
                        + "-byte primitive instead of the expected "
                        + std::to_string(Test.To) + "-byte one.";
    revng_abort(Error.c_str());
  }
}

BOOST_AUTO_TEST_CASE(TestPrimitiveSizeSelection) {
  for (const auto &Test : NumericPrimitives)
    testPrimitiveImpl(Test, model::PrimitiveKind::Unsigned);

  for (const auto &Test : FloatPrimitives)
    testPrimitiveImpl(Test, model::PrimitiveKind::Float);

  for (const auto &Test : GenericPrimitives)
    testPrimitiveImpl(Test, model::PrimitiveKind::Generic);
}
