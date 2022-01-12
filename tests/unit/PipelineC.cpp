/// \file Pipeline.cpp
/// \brief Tests for Auto Pipe

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
Steps:
 - Name:            FirstStep
   Pipes:
     - Name:             CopyPipe
       UsedContainers: [Strings1, Strings2]
)";

static rp_manager *runner;

struct Fixture {
public:
  Fixture() {
    std::string S = "fakeName";
    char *Array[1] = { S.data() };
    const char *PipelineText[1] = { PipelineTextContent };
    const char *LibToLoad[] = {};

    rp_initialize(1, Array, 0, LibToLoad);
    runner = rp_manager_create_from_string(1, PipelineText, 0, {}, "");
    BOOST_TEST(runner != nullptr);
  }

  ~Fixture() { rp_manager_destroy(runner); }
};

BOOST_AUTO_TEST_SUITE(s, *boost::unit_test::fixture<Fixture>())

BOOST_AUTO_TEST_CASE(CAPILoadTest) {
  BOOST_TEST(rp_manager_steps_count(runner) == 2);
  auto *FirstStep = rp_manager_get_step(runner, 0);
  BOOST_TEST(rp_manager_containers_count(runner) == 2);

  BOOST_TEST(rp_manager_step_name_to_index(runner, "FirstStep") == 0);
  BOOST_TEST(rp_manager_step_name_to_index(runner, "End") == 1);
  BOOST_TEST(rp_manager_get_kind_from_name(runner, "MISSING") == nullptr);
  BOOST_TEST(rp_manager_get_kind_from_name(runner, "Root") != nullptr);
  BOOST_TEST(rp_step_get_container_from_name(FirstStep, "Strings1"));
  BOOST_TEST(rp_step_get_container_from_name(FirstStep, "Strings2"));
}

BOOST_AUTO_TEST_SUITE_END()
