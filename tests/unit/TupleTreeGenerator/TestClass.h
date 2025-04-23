#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <cstdint>
#include <vector>

#include "revng/ADT/SortedVector.h"

#include "Generated/Early/TestClass.h"

namespace ttgtest {
class TestClass : public ttgtest::generated::TestClass {
public:
  using ttgtest::generated::TestClass::TestClass;
};
} // namespace ttgtest

#include "Generated/Late/TestClass.h"
