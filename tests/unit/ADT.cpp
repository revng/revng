/// \file ADT.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#define BOOST_TEST_MODULE RegisterUsageAnalyses
bool init_unit_test();
#include "boost/test/unit_test.hpp"

#include "revng/ADT/CompilationTime.h"
#include "revng/ADT/Concepts.h"
#include "revng/ADT/ConstexprString.h"
#include "revng/ADT/STLExtras.h"

//
// CompilationTime.h
//

using namespace std::string_view_literals;

template<size_t Count>
consteval size_t fullSize(std::array<std::string_view, Count> Components,
                          std::string_view Separator) {
  size_t Result = Separator.size() * Count;
  compile_time::repeat<Count>([&Result, &Components]<size_t Index> {
    Result += std::get<Index>(Components).size();
  });
  return Result;
}

inline constexpr std::array Components = { "instruction"sv,
                                           "0x401000:Code_x86_64"sv,
                                           "0x402000:Code_x86_64"sv,
                                           "0x403000:Code_x86_64"sv };
static_assert(fullSize(Components, "/"sv) == 75);

//
// Concepts.h
//

static_assert(SpecializationOf<std::pair<int, long>, std::pair>);
static_assert(StrictSpecializationOf<std::pair<int, long>, std::pair>);
static_assert(SpecializationOf<const std::pair<int, long>, std::pair>);
static_assert(StrictSpecializationOf<const std::pair<int, long>, std::pair>);

static_assert(SpecializationOf<std::string, std::basic_string>);
static_assert(StrictSpecializationOf<std::string, std::basic_string>);
static_assert(not SpecializationOf<std::string, std::basic_string_view>);
static_assert(not StrictSpecializationOf<std::string, std::basic_string_view>);

using Alias = std::pair<int, long>;
static_assert(SpecializationOf<Alias, std::pair>);
static_assert(StrictSpecializationOf<Alias, std::pair>);

template<typename Type>
struct InheritanceT : std::pair<int, Type> {};
struct PublicInheritance : public InheritanceT<long> {};
struct PrivateInheritance : private InheritanceT<long> {};
struct ProtectedInheritance : protected InheritanceT<long> {};

static_assert(SpecializationOf<PublicInheritance, std::pair>);
static_assert(SpecializationOf<PublicInheritance, InheritanceT>);
static_assert(not SpecializationOf<PrivateInheritance, std::pair>);
static_assert(not SpecializationOf<ProtectedInheritance, std::pair>);

static_assert(not StrictSpecializationOf<PublicInheritance, std::pair>);
static_assert(not StrictSpecializationOf<PublicInheritance, InheritanceT>);
static_assert(not StrictSpecializationOf<PrivateInheritance, std::pair>);
static_assert(not StrictSpecializationOf<ProtectedInheritance, std::pair>);

//
// ConstexprString.h
//

template<ConstexprString String>
struct StringParametrizedTrait {
  static constexpr std::string_view value = String;
};

static_assert(StringParametrizedTrait<"value">::value == "value");

BOOST_AUTO_TEST_CASE(ReachingDefinitionsTest) {
}

//
// STLExtras.h
//

consteval int takeAsTupleExample() {
  std::array<int, 6> Data = { 42, 43, 44, 45, 46, 47 };

  {
    // Edit some elements.
    auto &&[First, Second, Third] = takeAsTuple<3>(Data);
    ++First;
    Second = 2;
    Third = 3;
  }

  {
    // Read multiple elements through a view.
    auto &&[Second, Third] = takeAsTuple<2>(Data | std::views::drop(1));
    return Second + Third;
  }
}

static_assert(takeAsTupleExample() == 5);

constexpr bool test() {
  constexpr std::array Input{ 1, 2, 3, 5, 7, 8, 10, 1 };

  auto IsOdd = [](int I) { return I % 2 == 1; };

  // The "copy" approach works fine for vectors
  std::vector<int> InserterVectorOutput;
  std::ranges::copy(Input | std::views::filter(IsOdd),
                    std::back_inserter(InserterVectorOutput));

  // But it starts looking a lot uglier when applied to other containers.
  // (using vector in this test because `std::set` only became `constexpr` in
  // c++23, but the idea is the same: you cannot use `std::back_inserter` with
  // sets).
  using FakeSet = std::/* set */ vector<int>;
  FakeSet InserterSetOutput;
  std::ranges::copy(Input | std::views::filter(IsOdd),
                    std::inserter(InserterSetOutput,
                                  InserterSetOutput.begin()));

  // On the other hand, `revng::to` just works (tm) for everything that can be
  // constructed from two iterators. And even looks nicer to boot.
  FakeSet ToOutput = Input | std::views::filter(IsOdd) | revng::to<FakeSet>();

  return ToOutput.size() == InserterVectorOutput.size()
         && InserterVectorOutput.size() == InserterSetOutput.size();
}

static_assert(test());
