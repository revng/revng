/// \file Pipeline.cpp
/// Tests for C API of revng-pipeline.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/PipelineC/PipelineC.h"

#define BOOST_TEST_MODULE PipelineC
bool init_unit_test();
#include "boost/test/unit_test.hpp"

#include "revng/UnitTestHelpers/UnitTestHelpers.h"

const auto PipelineTextContent =
  R"(Component: revng-test
Containers:
 - Name:            strings-1
   Type:            string-container
 - Name:            strings-2
   Type:            string-container
Branches:
  -  Steps:
     - Name:            first-step
       Pipes:
         - Type:             copy-pipe
           UsedContainers: [strings-1, strings-2]
)";

static rp_manager *Manager;

struct Fixture {
public:
  Fixture() {
    std::string S = "fakeName";
    const char *Array[1] = { S.data() };
    const char *PipelineText[1] = { PipelineTextContent };
    const char *LibToLoad[] = {};

    rp_initialize(1, Array, 0, {});
    Manager = rp_manager_create_from_string(1, PipelineText, 0, {}, "");
    revng_check(Manager != nullptr);
  }

  ~Fixture() {
    rp_manager_destroy(Manager);
    rp_shutdown();
  }
};

BOOST_AUTO_TEST_SUITE(PipelineCTestSuite, *boost::unit_test::fixture<Fixture>())

BOOST_AUTO_TEST_CASE(CAPILoadTest) {
  rp_manager_get_step_from_name(Manager, "begin");
  rp_manager_get_step_from_name(Manager, "first-step");
  BOOST_TEST(rp_manager_get_kind_from_name(Manager, "MISSING") == nullptr);
  BOOST_TEST(rp_manager_get_kind_from_name(Manager, "root") != nullptr);
}

BOOST_AUTO_TEST_SUITE_END()
