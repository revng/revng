//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

#define BOOST_TEST_MODULE TupleTreeGenerator
bool init_unit_test();

#include "boost/test/unit_test.hpp"

#include "llvm/Support/YAMLTraits.h"

#include "revng/TupleTree/VisitsImpl.h"
#include "revng/UnitTestHelpers/UnitTestHelpers.h"

#include "TestClass.h"
#include "TestEnum.h"

/// Ensures that TestClass can be YAML serialized and deserialized
BOOST_AUTO_TEST_CASE(YAMLSerializationRoundTripTest) {
  using namespace ttgtest;
  TestClass ReferenceInstance;
  ReferenceInstance.RequiredField() = 1;
  ReferenceInstance.OptionalField() = 2;
  ReferenceInstance.EnumField() = ttgtest::TestEnum::MemberOne;
  ReferenceInstance.SequenceField() = { 1, 2, 3, 4, 5 };
  using RefType = TupleTreeReference<uint64_t, TestClass>;
  ReferenceInstance.ReferenceField() = RefType::fromString(&ReferenceInstance,
                                                           "/SequenceField/1");

  revng_assert(ReferenceInstance.ReferenceField().isValid());
  revng_assert(ReferenceInstance.ReferenceField().get());

  std::string Buffer;
  llvm::raw_string_ostream OutputStream(Buffer);
  llvm::yaml::Output YAMLOutput(OutputStream);

  YAMLOutput << ReferenceInstance;
  std::cout << Buffer;

  TestClass DeserializedInstance;
  llvm::yaml::Input YamlInput(Buffer);
  YamlInput >> DeserializedInstance;

  revng_assert(ReferenceInstance == DeserializedInstance);
}
