//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/PipelineC/PipelineC.h"
#include "revng/PipelineC/Tracing/Private.h"
#include "revng/PipelineC/Tracing/Trace.h"
#include "revng/Support/PathList.h"
#include "revng/Support/TemporaryFile.h"

#define BOOST_TEST_MODULE PipelineCTracing
bool init_unit_test();
#include "boost/test/unit_test.hpp"

#include "revng/UnitTestHelpers/UnitTestHelpers.h"

using namespace revng;

struct Fixture {
public:
  Fixture() {
    llvm::StringRef RootPath = getCurrentRoot();
    std::string LibPath = joinPath(RootPath,
                                   "lib/librevngStringContainerLibrary.so");
    std::string PipelinePath = joinPath(RootPath, "tests/pipeline/Tracing.yml");

    std::vector<std::string> ArgvStorage = { "",
                                             "-load=" + LibPath,
                                             "-pipeline-path=" + PipelinePath };
    std::vector<const char *> Argv;
    for (std::string &Arg : ArgvStorage)
      Argv.push_back(Arg.c_str());

    rp_initialize(Argv.size(), Argv.data(), 0, {});
  }

  ~Fixture() { rp_shutdown(); }
};

static void verifyTrace(tracing::Trace &Trace) {
  BOOST_TEST(Trace.Version == 1ULL);
  BOOST_TEST(Trace.Commands.size() == 4ULL);
  BOOST_TEST(Trace.Commands[0].Name == "rp_manager_create");
  BOOST_TEST(Trace.Commands[1].Name == "rp_manager_get_step_from_name");
  BOOST_TEST(Trace.Commands[2].Name == "rp_manager_get_step_from_name");
  BOOST_TEST(Trace.Commands[3].Name == "rp_manager_destroy");

  BOOST_TEST(Trace.Commands[0].Arguments.size() == 3ULL);
  BOOST_TEST(Trace.Commands[1].Arguments.size() == 2ULL);
  BOOST_TEST(Trace.Commands[2].Arguments.size() == 2ULL);
  BOOST_TEST(Trace.Commands[3].Arguments.size() == 1ULL);

  const std::string &ManagerAddress = Trace.Commands[0].Result;
  BOOST_TEST(Trace.Commands[1].Arguments[0].getScalar() == ManagerAddress);
  BOOST_TEST(Trace.Commands[2].Arguments[0].getScalar() == ManagerAddress);
  BOOST_TEST(Trace.Commands[3].Arguments[0].getScalar() == ManagerAddress);

  BOOST_TEST(Trace.Commands[1].Arguments[1].getScalar() == "begin");
  BOOST_TEST(Trace.Commands[2].Arguments[1].getScalar() == "first-step");
}

BOOST_AUTO_TEST_SUITE(PipelineCTracingTestSuite,
                      *boost::unit_test::fixture<Fixture>())

BOOST_AUTO_TEST_CASE(PipelineCTraceTest) {
  std::string Buffer;

  {
    llvm::raw_string_ostream OS(Buffer);
    tracing::setTracing(&OS);

    rp_manager *Manager = rp_manager_create(0, {}, "");
    rp_manager_get_step_from_name(Manager, "begin");
    rp_manager_get_step_from_name(Manager, "first-step");
    rp_manager_destroy(Manager);

    tracing::setTracing(nullptr);
  }

  tracing::Trace Trace;
  {
    auto MemoryBuffer = llvm::MemoryBuffer::getMemBuffer(Buffer);
    auto MaybeTrace = tracing::Trace::fromBuffer(*MemoryBuffer);
    BOOST_TEST(!!MaybeTrace);

    Trace = *MaybeTrace;
  }

  verifyTrace(Trace);

  Buffer.clear();
  {
    llvm::raw_string_ostream OS(Buffer);
    tracing::setTracing(&OS);

    llvm::cantFail(Trace.run());

    tracing::setTracing(nullptr);
  }

  tracing::Trace Trace2;
  {
    auto MemoryBuffer = llvm::MemoryBuffer::getMemBuffer(Buffer);
    auto MaybeTrace = tracing::Trace::fromBuffer(*MemoryBuffer);
    BOOST_TEST(!!MaybeTrace);

    Trace2 = *MaybeTrace;
  }

  verifyTrace(Trace2);
}

BOOST_AUTO_TEST_SUITE_END()
