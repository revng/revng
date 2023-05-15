/// \file UpcastablePointer.cpp
/// \brief

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/ADT/UpcastablePointer.h"

template<typename T>
concept UniquePtrLike = requires(T A, typename T::pointer B) {
  std::is_pointer_v<typename T::pointer>;
  typename T::element_type;
  { A.get() } -> std::same_as<typename T::pointer>;
  { A.reset(B) } -> std::same_as<void>;
  { T() };
  { T(B) };
};

class TestClass {
public:
  static bool classof(TestClass *) { return true; }
};

template<>
struct concrete_types_traits<TestClass> {
  using type = std::tuple<TestClass>;
};

// Test UniquePtrLike
static_assert(UniquePtrLike<std::unique_ptr<int>>);
static_assert(not UniquePtrLike<TestClass>);
static_assert(UniquePtrLike<UpcastablePointer<TestClass>>);

static_assert(Upcastable<TestClass>);

// Test UpcastablePointerLike
static_assert(UpcastablePointerLike<TestClass *>);
static_assert(UpcastablePointerLike<std::unique_ptr<TestClass>>);
static_assert(UpcastablePointerLike<UpcastablePointer<TestClass>>);

static_assert(std::is_default_constructible_v<UpcastablePointer<TestClass>>);
static_assert(std::is_copy_assignable_v<UpcastablePointer<TestClass>>);
static_assert(std::is_copy_constructible_v<UpcastablePointer<TestClass>>);
static_assert(std::is_move_assignable_v<UpcastablePointer<TestClass>>);
static_assert(std::is_move_constructible_v<UpcastablePointer<TestClass>>);

int main() {
  return 0;
}
