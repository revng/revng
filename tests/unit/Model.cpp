/// \file Model.cpp
/// \brief

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#define BOOST_TEST_MODULE Model
bool init_unit_test();
#include "boost/test/unit_test.hpp"

#include "revng/Model/Binary.h"
#include "revng/Support/MetaAddress/KeyTraits.h"

using namespace model;

BOOST_AUTO_TEST_CASE(TestIntrospection) {
  using namespace llvm;

  Function TheFunction(MetaAddress::invalid());

  // Use get
  TheFunction.Name = "FunctionName";
  revng_check(get<1>(TheFunction) == "FunctionName");

  // Test std::tuple_size
  static_assert(std::tuple_size<Function>::value >= 2);

  // Test TupleLikeTraits
  using TLT = TupleLikeTraits<Function>;
  static_assert(std::is_same_v<std::tuple_element_t<1, Function>,
                               decltype(TheFunction.Name)>);
  revng_check(StringRef(TLT::name()) == "model::Function");
  revng_check(StringRef(TLT::fieldName<1>()) == "Name");
}

BOOST_AUTO_TEST_CASE(TestPathAccess) {
  Binary TheBinary;
  using FunctionsType = decltype(TheBinary.Functions);
  auto *FirstField = getByPath<FunctionsType>(KeyIntVector{ 0 }, TheBinary);
  revng_check(FirstField == &TheBinary.Functions);

  auto *FunctionsField = getByPath<FunctionsType>("/Functions", TheBinary);
  revng_check(FunctionsField == &TheBinary.Functions);

  // Test non existing field
  revng_check(getByPath<FunctionsType>("/Function", TheBinary) == nullptr);

  // Test non existing entry in container
  revng_check(getByPath<Function>("/Functions/:Invalid", TheBinary) == nullptr);

  // Test existing entry in container
  Function &F = TheBinary.Functions[MetaAddress::invalid()];
  revng_check(getByPath<Function>("/Functions/:Invalid", TheBinary) == &F);
}

BOOST_AUTO_TEST_CASE(TestStringPathConversion) {
  revng_check(stringAsPath<Binary>("/").value() == KeyIntVector{});
  revng_check(stringAsPath<Binary>("/Functions").value() == KeyIntVector{ 0 });

  KeyIntVector FiveZeros{ 0, 0, 0, 0, 0 };
  auto MaybeInvalidFunctionPath = stringAsPath<Binary>("/Functions/:Invalid");
  revng_check(MaybeInvalidFunctionPath.value() == FiveZeros);

  KeyIntVector FiveZerosAndOne{ 0, 0, 0, 0, 00, 1 };
  auto MaybePath = stringAsPath<Binary>("/Functions/:Invalid/Name");
  revng_check(MaybePath.value() == FiveZerosAndOne);

  auto CheckRoundTrip = [](const char *String) {
    auto Path = stringAsPath<Binary>(String).value();
    auto StringAgain = pathAsString<Binary>(Path);
    revng_check(StringAgain == String);
  };

  CheckRoundTrip("/Functions");
  CheckRoundTrip("/Functions/:Invalid");
  CheckRoundTrip("/Functions/:Invalid/Entry");
  CheckRoundTrip("/Functions/0x1000:Code_arm/Entry");
  CheckRoundTrip("/Functions/0x1000:Code_arm/CFG/"
                 "0x2000:Code_arm-0x3000:Code_arm/Start");
}

BOOST_AUTO_TEST_CASE(TestPathMatcher) {
  auto ARM1000 = MetaAddress::fromString("0x1000:Code_arm");
  auto ARM2000 = MetaAddress::fromString("0x2000:Code_arm");
  auto ARM3000 = MetaAddress::fromString("0x3000:Code_arm");

  //
  // Single matcher
  //
  {
    auto Matcher = PathMatcher::create<Binary>("/Functions/*/Entry").value();

    auto ARM1000EntryPath = pathAsString<Binary>(Matcher.apply(ARM1000));
    revng_check(ARM1000EntryPath == "/Functions/0x1000:Code_arm/Entry");

    auto MaybeToMatch = stringAsPath<Binary>("/Functions/0x1000:Code_arm/"
                                             "Entry");
    auto MaybeMatch = Matcher.match<MetaAddress>(MaybeToMatch.value());
    revng_check(MaybeMatch);
    revng_check(std::get<0>(*MaybeMatch) == ARM1000);
  }

  //
  // Double matcher
  //
  {
    auto MaybeMatcher = PathMatcher::create<Binary>("/Functions/*/CFG/*/Start");
    auto Matcher = MaybeMatcher.value();

    using BlockKeyPair = std::pair<MetaAddress, MetaAddress>;
    BlockKeyPair BlockKey = { ARM2000, ARM3000 };
    auto BlockKeyName = getNameFromYAMLScalar(BlockKey);
    revng_check(BlockKeyName == "0x2000:Code_arm-0x3000:Code_arm");

    auto ARM1000EntryPath = Matcher.apply(ARM1000, BlockKey);
    auto ARM1000EntryPathAsString = pathAsString<Binary>(ARM1000EntryPath);
    auto ExpectedName = ("/Functions/0x1000:Code_arm/CFG/"
                         "0x2000:Code_arm-0x3000:Code_arm/Start");
    revng_check(ARM1000EntryPathAsString == ExpectedName);

    auto Match = Matcher.match<MetaAddress, BlockKeyPair>(ARM1000EntryPath);
    revng_check(Match);
    revng_check(std::get<0>(*Match) == ARM1000);
    revng_check(std::get<1>(*Match) == BlockKey);

    {
      auto Path = stringAsPath<Binary>("/Functions");
      revng_check((not Matcher.match<MetaAddress, BlockKeyPair>(Path.value())));
    }

    {
      auto Path = stringAsPath<Binary>("/Functions/:Invalid");
      revng_check((not Matcher.match<MetaAddress, BlockKeyPair>(Path.value())));
    }
  }
}
