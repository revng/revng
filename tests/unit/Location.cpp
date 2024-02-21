//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#define BOOST_TEST_MODULE location
bool init_unit_test();
#include "boost/test/unit_test.hpp"

#include "revng/Model/Binary.h"
#include "revng/Pipeline/Location.h"
#include "revng/Pipes/Ranks.h"
#include "revng/Support/BasicBlockID.h"

BOOST_AUTO_TEST_SUITE(RevngLocationInfrastructure);

BOOST_AUTO_TEST_CASE(DefinitionStyles) {
  namespace ranks = revng::ranks;

  const auto A0 = MetaAddress::fromString("0x1:Generic64");
  const auto A1 = BasicBlockID::fromString("0x2:Generic64");
  const auto A2 = MetaAddress::fromString("0x3:Generic64");
  auto Location = pipeline::location(ranks::Instruction, A0, A1, A2);

  constexpr auto S = "/instruction/0x1:Generic64/0x2:Generic64/0x3:Generic64";
  auto MaybeInstruction = pipeline::locationFromString(ranks::Instruction, S);
  revng_check(MaybeInstruction.has_value());

  revng_check(Location.tuple() == MaybeInstruction->tuple());
  revng_check(Location.toString() == S);
}

BOOST_AUTO_TEST_CASE(MetaAddressAsTheKey) {
  namespace ranks = revng::ranks;

  const auto A0 = MetaAddress::fromString("0x123:Generic64");
  const auto A1 = BasicBlockID::fromString("0x456:Generic64");
  const auto A2 = MetaAddress::fromString("0x789:Generic64");
  auto Location = pipeline::location(ranks::Instruction, A0, A1, A2);

  revng_check(std::get<0>(Location.at(ranks::Function)).address() == 0x123);
  revng_check(Location.at(ranks::BasicBlock).start().address() == 0x456);
  revng_check(Location.at(ranks::Instruction).address() == 0x789);

  constexpr auto Expected = "/instruction/0x123:Generic64/"
                            "0x456:Generic64/0x789:Generic64";
  revng_check(Location.toString() == Expected);
  revng_check(Location.toString() == serializeToString(Location));
}

static model::DefinitionReference makeFunction(model::Binary &Model) {
  model::CABIFunctionDefinition Function;
  Function.CustomName() = "my_cool_func";
  Function.OriginalName() = "Function_at_0x40012f:Code_x86_64";
  Function.ABI() = model::ABI::SystemV_x86_64;

  using UTD = model::UpcastableTypeDefinition;
  auto Ptr = UTD::make<model::CABIFunctionDefinition>(std::move(Function));

  return Model.recordNewType(std::move(Ptr));
}

BOOST_AUTO_TEST_CASE(TypeIDAsTheKey) {
  TupleTree<model::Binary> NewModel;
  auto Function = makeFunction(*NewModel);

  namespace ranks = revng::ranks;
  auto FieldLocation = pipeline::location(ranks::UnionField,
                                          Function.get()->key(),
                                          2);

  auto Path = NewModel->getDefinitionReference(FieldLocation.at(ranks::Type));
  revng_check(Path.getConst()->name() == "my_cool_func");

  std::string ID = Function.toString();
  ID = ID.substr(ID.find_last_of('/') + 1);
  revng_check(FieldLocation.toString() == "/union-field/" + ID + "/2");
  revng_check(FieldLocation.toString() == serializeToString(FieldLocation));
}

BOOST_AUTO_TEST_CASE(Serialization) {
  constexpr std::array<std::string_view, 4> TestCases{
    // a list of unrelated serialized locations to use as sources of truth
    "/binary",
    "/instruction/0x12:Generic64/0x34:Generic64/0x56:Generic64",
    "/type/1026-PrimitiveDefinition",
    "/raw-byte-range/0x78:Generic64/0x90:Generic64"
  };

  using pipeline::locationFromString;
  namespace ranks = revng::ranks;
  for (auto TestCase : TestCases) {
    bool ParsedOnce = false;

    if (auto Binary = locationFromString(ranks::Binary, TestCase)) {
      revng_check(TestCase == TestCases[0]);
      revng_check(ParsedOnce == false);
      ParsedOnce = true;

      revng_check(TestCase == Binary->toString());
    }

    if (auto Instruction = locationFromString(ranks::Instruction, TestCase)) {
      revng_check(TestCase == TestCases[1]);
      revng_check(ParsedOnce == false);
      ParsedOnce = true;

      auto FunctionMetaAddress = std::get<0>(Instruction->at(ranks::Function));
      revng_check(FunctionMetaAddress.address() == 0x12);
      revng_check(Instruction->at(ranks::BasicBlock).start().address() == 0x34);
      revng_check(Instruction->at(ranks::Instruction).address() == 0x56);

      revng_check(TestCase == Instruction->toString());
    }

    if (auto Type = locationFromString(ranks::Type, TestCase)) {
      revng_check(TestCase == TestCases[2]);
      revng_check(ParsedOnce == false);
      ParsedOnce = true;

      revng_check(std::get<uint64_t>(Type->at(ranks::Type)) == 1026);

      revng_check(TestCase == Type->toString());
    }

    if (auto RawRange = locationFromString(ranks::RawByteRange, TestCase)) {
      revng_check(TestCase == TestCases[3]);
      revng_check(ParsedOnce == false);
      ParsedOnce = true;

      revng_check(RawRange->at(ranks::RawByte).address() == 0x78);
      revng_check(RawRange->at(ranks::RawByteRange).address() == 0x90);

      revng_check(TestCase == RawRange->toString());
    }

    revng_check(!locationFromString(ranks::Function, TestCase).has_value());
    revng_check(!locationFromString(ranks::BasicBlock, TestCase).has_value());
    revng_check(!locationFromString(ranks::StructField, TestCase).has_value());
    revng_check(!locationFromString(ranks::UnionField, TestCase).has_value());
    revng_check(!locationFromString(ranks::EnumEntry, TestCase).has_value());
    revng_check(!locationFromString(ranks::RawByte, TestCase).has_value());

    using namespace std::string_literals;
    revng_check(ParsedOnce,
                ("Parsing of '"s + TestCase.data() + "' failed").c_str());
  }
}

BOOST_AUTO_TEST_SUITE_END();
