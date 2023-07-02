/// \file DiffInvalidationEvent.cpp
/// Tests for C API of revng-pipeline.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/Binary.h"
#include "revng/Pipes/Kinds.h"
#include "revng/Support/MetaAddress.h"

#define BOOST_TEST_MODULE PipelineC
bool init_unit_test();
#include "boost/test/unit_test.hpp"

#include "revng/UnitTestHelpers/UnitTestHelpers.h"

using namespace revng::kinds;
using namespace pipeline;

BOOST_AUTO_TEST_SUITE(ModelInvalidationDiffSuite)

BOOST_AUTO_TEST_CASE(RootInvalidationTest) {
  model::Binary Empty;
  model::Binary New;

  Context Ctx;
  MetaAddress Address(0x1000, MetaAddressType::Code_aarch64);
  New.ExtraCodeAddresses().insert(Address);

  TargetsList ToRemove;
  GlobalTupleTreeDiff Event(diff(Empty, New));
  Root.getInvalidations(Ctx, ToRemove, Event);
  BOOST_TEST(ToRemove.size() == 1);
  BOOST_TEST(&ToRemove.front().getKind() == &Root);
}

BOOST_AUTO_TEST_SUITE_END()
