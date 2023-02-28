/// \file Pipeline.cpp
/// \brief Tests for C API of revng-pipeline

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/PipelineC/PipelineC.h"

#define BOOST_TEST_MODULE PipelineC
bool init_unit_test();
#include "boost/test/unit_test.hpp"

#include "revng/UnitTestHelpers/UnitTestHelpers.h"

const auto PipelineTextContent =
  R"(Containers:
 - Name:            Strings1
   Type:            StringContainer
 - Name:            Strings2
   Type:            StringContainer
Branches:
  -  Steps:
     - Name:            FirstStep
       Pipes:
         - Type:             CopyPipe
           UsedContainers: [Strings1, Strings2]
)";

static rp_manager *Runner;

struct Fixture {
public:
  Fixture() {
    std::string S = "fakeName";
    const char *Array[1] = { S.data() };
    const char *PipelineText[1] = { PipelineTextContent };
    const char *LibToLoad[] = {};

    rp_initialize(1, Array, 0, LibToLoad, 0, {});
    Runner = rp_manager_create_from_string(1, PipelineText, 0, {}, "");
    revng_check(Runner != nullptr);
  }

  ~Fixture() {
    rp_manager_destroy(Runner);
    rp_shutdown();
  }
};

BOOST_AUTO_TEST_SUITE(PipelineCTestSuite, *boost::unit_test::fixture<Fixture>())

BOOST_AUTO_TEST_CASE(CAPILoadTest) {
  BOOST_TEST(rp_manager_steps_count(Runner) == 2UL);
  auto *FirstStep = rp_manager_get_step(Runner, 0);
  BOOST_TEST(rp_manager_containers_count(Runner) == 2UL);

  BOOST_TEST(rp_manager_step_name_to_index(Runner, "begin") == 0UL);
  BOOST_TEST(rp_manager_step_name_to_index(Runner, "FirstStep") == 1UL);
  BOOST_TEST(rp_manager_get_kind_from_name(Runner, "MISSING") == nullptr);
  BOOST_TEST(rp_manager_get_kind_from_name(Runner, "Root") != nullptr);
}

BOOST_AUTO_TEST_SUITE_END()
