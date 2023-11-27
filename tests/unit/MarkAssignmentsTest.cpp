//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

#include <compare>
#include <map>
#include <set>
#include <string>
#include <vector>

#define BOOST_TEST_MODULE MarkAssignmentsTest
bool init_unit_test();
#include "boost/test/unit_test.hpp"

#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"

#include "revng/Support/IRHelpers.h"
#include "revng/UnitTestHelpers/LLVMTestHelpers.h"

#include "revng-c/Support/FunctionTags.h"

#include "lib/IRCanonicalization/MarkAssignments/MarkAssignments.h"

using namespace MarkAssignments;
using namespace MarkAssignments::Reasons;

struct BBAssignmentFlags {
  std::string BBName;
  std::vector<Reasons::Values> InstrFlags;
};

using ExpectedFlagsType = std::vector<BBAssignmentFlags>;

static void runTestOnFunctionWithExpected(const char *Body,
                                          const ExpectedFlagsType &Expected) {
  MarkLog.enable();
  revng_log(MarkLog, "Running Test On Function: " << Body);

  llvm::LLVMContext C;
  std::unique_ptr<llvm::Module> M = loadModule(C, Body);

  // main is present and has a body
  llvm::Function *F = M->getFunction("main");
  revng_check(nullptr != F);
  revng_check(not F->empty());

  llvm::Function *Opaque = M->getFunction("opaque");
  revng_check(nullptr != Opaque);
  FunctionTags::WritesMemory.addTo(Opaque);

  auto Results = selectAssignments(*F);

  // We expect a vector of results for each BasicBlock in F, otherwise the
  // Expected results are not well formed.
  revng_check(F->size() == Expected.size());

  for (const auto &[BB, BBFlags] : llvm::zip_first(*F, Expected)) {
    const auto &[BBName, InstrFlagVec] = BBFlags;
    // Name matches
    revng_check(BB.hasName() and BB.getName().str() == BBName);
    // Same number of instructions
    revng_check(BB.size() == InstrFlagVec.size());
    for (const auto &[I, ExpectedFlag] : llvm::zip_first(BB, InstrFlagVec)) {
      // Check that the expected flags match the computed one.
      // Using Results[] may create a None flag if non was computed but that's
      // not a problem.
      revng_log(MarkLog, dumpToString(&I));
      revng_log(MarkLog,
                "Results[&I].value() = " << Results[&I].value()
                                         << " ExpectedFlag = " << ExpectedFlag);
      revng_check(Results[&I].value() == ExpectedFlag);
    }
  }
}

using InstrNameToLoadNamesMap = std::map<std::string, std::set<std::string>>;

BOOST_AUTO_TEST_CASE(AlwaysAssignNoUses) {

  const char *EmptyBody = R"LLVM(
  %unused = inttoptr i64 4294967296 to i64*
  unreachable
  )LLVM";

  ExpectedFlagsType ExpectedFlags{
    BBAssignmentFlags{ "initial_block", { AlwaysAssign, None } },
  };

  runTestOnFunctionWithExpected(EmptyBody, ExpectedFlags);
}

BOOST_AUTO_TEST_CASE(RetNone) {

  const char *Body = R"LLVM(
  ret void
  )LLVM";

  ExpectedFlagsType ExpectedFlags{
    BBAssignmentFlags{ "initial_block", { None } },
  };

  runTestOnFunctionWithExpected(Body, ExpectedFlags);
}

BOOST_AUTO_TEST_CASE(InstrinsicWithoutSideEffects) {

  const char *Body = R"LLVM(
  %bswapped = call i64 @llvm.bswap.i64(i64 1231231231)
  unreachable
  )LLVM";

  ExpectedFlagsType ExpectedFlags{
    BBAssignmentFlags{ "initial_block", { AlwaysAssign, None } },
  };

  runTestOnFunctionWithExpected(Body, ExpectedFlags);
}

BOOST_AUTO_TEST_CASE(CallSideEffectsAndNoUses) {

  const char *Body = R"LLVM(
  %opaqued = call i64 @opaque(i64 1231231231)
  unreachable
  )LLVM";

  ExpectedFlagsType ExpectedFlags{
    BBAssignmentFlags{ "initial_block",
                       { AlwaysAssign | HasSideEffects, None } },
  };

  runTestOnFunctionWithExpected(Body, ExpectedFlags);
}

BOOST_AUTO_TEST_CASE(StoreSideEffects) {

  const char *Body = R"LLVM(
  store i64 123, i64 * inttoptr (i64 4294967296 to i64*)
  unreachable
  )LLVM";

  ExpectedFlagsType ExpectedFlags{
    BBAssignmentFlags{ "initial_block", { HasSideEffects, None } },
  };

  runTestOnFunctionWithExpected(Body, ExpectedFlags);
}

BOOST_AUTO_TEST_CASE(ManyUses) {

  const char *Body = R"LLVM(
  %pointer = inttoptr i64 4294967296 to i64*
  %loaded = load i64, i64 * %pointer
  %reloaded = load i64, i64 * %pointer
  unreachable
  )LLVM";

  ExpectedFlagsType ExpectedFlags{
    BBAssignmentFlags{ "initial_block",
                       { None, AlwaysAssign, AlwaysAssign, None } },
  };

  runTestOnFunctionWithExpected(Body, ExpectedFlags);
}

BOOST_AUTO_TEST_CASE(BranchNone) {

  const char *Body = R"LLVM(
  br label %next

  next:
  unreachable
  )LLVM";

  ExpectedFlagsType ExpectedFlags{
    BBAssignmentFlags{ "initial_block", { None } },
    BBAssignmentFlags{ "next", { None } },
  };

  runTestOnFunctionWithExpected(Body, ExpectedFlags);
}

BOOST_AUTO_TEST_CASE(Interfering) {

  const char *Body = R"LLVM(
  %pointer = inttoptr i64 4294967296 to i64*
  %loaded = load i64, i64 * %pointer
  store i64 undef, i64 * %pointer
  store i64 %loaded, i64 * %pointer
  unreachable
  )LLVM";

  ExpectedFlagsType ExpectedFlags{
    BBAssignmentFlags{ "initial_block",
                       { None,
                         HasInterferingSideEffects,
                         HasSideEffects,
                         HasSideEffects,
                         None } },
  };

  runTestOnFunctionWithExpected(Body, ExpectedFlags);
}

BOOST_AUTO_TEST_CASE(NonInterfering) {

  const char *Body = R"LLVM(
  %pointer = inttoptr i64 4294967296 to i64*
  %loaded = load i64, i64 * %pointer
  store i64 %loaded, i64 * %pointer
  unreachable
  )LLVM";

  ExpectedFlagsType ExpectedFlags{
    BBAssignmentFlags{ "initial_block", { None, None, HasSideEffects, None } },
  };

  runTestOnFunctionWithExpected(Body, ExpectedFlags);
}

BOOST_AUTO_TEST_CASE(ComplexNonInterfering) {

  const char *Body = R"LLVM(
  %pointer = inttoptr i64 4294967296 to i64*
  %loaded = load i64, i64 * %pointer
  %sum1 = add i64 %loaded, 4
  %sum2 = add i64 %sum1, 4
  store i64 %sum2, i64 * %pointer
  unreachable
  )LLVM";

  ExpectedFlagsType ExpectedFlags{
    BBAssignmentFlags{
      /*.BBName =*/"initial_block",
      /*.InstrFlags =*/{ None, None, None, None, HasSideEffects, None } },
  };

  runTestOnFunctionWithExpected(Body, ExpectedFlags);
}

BOOST_AUTO_TEST_CASE(DoubleIndirectionWithoutSideEffects) {

  const char *Body = R"LLVM(
  %ptrtoptr = inttoptr i64 4294967296 to i64**
  %ptr = load i64*, i64 **%ptrtoptr
  %value = load i64, i64 *%ptr
  %call = call i64 @opaque(i64 %value)
  unreachable
  )LLVM";

  ExpectedFlagsType ExpectedFlags{
    BBAssignmentFlags{ /*.BBName =*/"initial_block",
                       /*.InstrFlags =*/{ None,
                                          None,
                                          None,
                                          AlwaysAssign | HasSideEffects,
                                          None } },
  };

  runTestOnFunctionWithExpected(Body, ExpectedFlags);
}

BOOST_AUTO_TEST_CASE(DoubleIndirectionWithSideEffects) {

  const char *Body = R"LLVM(
  %ptrtoptr = inttoptr i64 4294967296 to i64**
  %ptr = load i64*, i64 **%ptrtoptr
  %call = call i64 @opaque(i64 1032143324)
  %value = load i64, i64 *%ptr
  %othercall = call i64 @opaque(i64 %value)
  unreachable
  )LLVM";

  ExpectedFlagsType ExpectedFlags{
    BBAssignmentFlags{ /*.BBName =*/"initial_block",
                       /*.InstrFlags =*/{ None,
                                          HasInterferingSideEffects,
                                          AlwaysAssign | HasSideEffects,
                                          None,
                                          AlwaysAssign | HasSideEffects,
                                          None } },
  };

  runTestOnFunctionWithExpected(Body, ExpectedFlags);
}

BOOST_AUTO_TEST_CASE(ConditionalIsNotAssigned1) {
  const char *Body = R"LLVM(
  %pointer0 = inttoptr i64 4294967000 to i64*
  %value0 = load i64, i64 *%pointer0
  %pointer1 = inttoptr i64 4294967000 to i64*
  %value1 = load i64, i64 *%pointer1
  %pointer2 = inttoptr i64 4294967200 to i64*
  %value2 = load i64, i64 *%pointer2
  %cmp = icmp ult i64 %value1, %value2
  br i1 %cmp, label %smaller, label %greater

  smaller:
  br label %end

  greater:
  br label %end

  end:
  store i64 %value0, i64 * %pointer0
  unreachable
  )LLVM";

  ExpectedFlagsType ExpectedFlags{
    BBAssignmentFlags{ /*.BBName =*/"initial_block",
                       /*.InstrFlags =*/
                       {
                         None,
                         None,
                         None,
                         None,
                         None,
                         None,
                         None,
                         None,
                       } },
    BBAssignmentFlags{ /*.BBName =*/"smaller",
                       /*.InstrFlags =*/{ None } },
    BBAssignmentFlags{ /*.BBName =*/"greater",
                       /*.InstrFlags =*/{ None } },
    BBAssignmentFlags{ /*.BBName =*/"end",
                       /*.InstrFlags =*/{ HasSideEffects, None } },
  };

  runTestOnFunctionWithExpected(Body, ExpectedFlags);
}

BOOST_AUTO_TEST_CASE(ConditionalIsNotAssigned2) {
  const char *Body = R"LLVM(
  %pointer0 = inttoptr i64 4294967000 to i64*
  %value0 = load i64, i64 *%pointer0
  %pointer1 = inttoptr i64 4294967000 to i64*
  %value1 = load i64, i64 *%pointer1
  %pointer2 = inttoptr i64 4294967200 to i64*
  %value2 = load i64, i64 *%pointer2
  %cmp = icmp ult i64 %value1, %value2
  br i1 %cmp, label %smaller, label %greater

  smaller:
  br label %end

  greater:
  store i64 %value0, i64 * %pointer0
  br label %end

  end:
  unreachable
  )LLVM";

  ExpectedFlagsType ExpectedFlags{
    BBAssignmentFlags{ /*.BBName =*/"initial_block",
                       /*.InstrFlags =*/
                       {
                         None,
                         None,
                         None,
                         None,
                         None,
                         None,
                         None,
                         None,
                       } },
    BBAssignmentFlags{ /*.BBName =*/"smaller",
                       /*.InstrFlags =*/{ None } },
    BBAssignmentFlags{ /*.BBName =*/"greater",
                       /*.InstrFlags =*/{ HasSideEffects, None } },
    BBAssignmentFlags{ /*.BBName =*/"end",
                       /*.InstrFlags =*/{ None } },
  };

  runTestOnFunctionWithExpected(Body, ExpectedFlags);
}

BOOST_AUTO_TEST_CASE(AssignPendingInThenBranch) {
  const char *Body = R"LLVM(
  %pointer0 = inttoptr i64 4294967000 to i64*
  %value0 = load i64, i64 *%pointer0
  %pointer1 = inttoptr i64 4294967000 to i64*
  %value1 = load i64, i64 *%pointer1
  %pointer2 = inttoptr i64 4294967200 to i64*
  %value2 = load i64, i64 *%pointer2
  %cmp = icmp ult i64 %value1, %value2
  br i1 %cmp, label %smaller, label %greater

  smaller:
  store i64 undef, i64 *undef
  br label %end

  greater:
  br label %end

  end:
  %constant = add i64 %value0, 0
  unreachable
  )LLVM";

  ExpectedFlagsType ExpectedFlags{
    BBAssignmentFlags{ /*.BBName =*/"initial_block",
                       /*.InstrFlags =*/
                       {
                         None,
                         HasInterferingSideEffects,
                         None,
                         None,
                         None,
                         None,
                         None,
                         None,
                       } },
    BBAssignmentFlags{ /*.BBName =*/"smaller",
                       /*.InstrFlags =*/{ HasSideEffects, None } },
    BBAssignmentFlags{ /*.BBName =*/"greater",
                       /*.InstrFlags =*/{ None } },
    BBAssignmentFlags{ /*.BBName =*/"end",
                       /*.InstrFlags =*/{ AlwaysAssign, None } },
  };

  runTestOnFunctionWithExpected(Body, ExpectedFlags);
}

BOOST_AUTO_TEST_CASE(AssignPendingInElseBranch) {
  const char *Body = R"LLVM(
  %pointer0 = inttoptr i64 4294967000 to i64*
  %value0 = load i64, i64 *%pointer0
  %pointer1 = inttoptr i64 4294967000 to i64*
  %value1 = load i64, i64 *%pointer1
  %pointer2 = inttoptr i64 4294967200 to i64*
  %value2 = load i64, i64 *%pointer2
  %cmp = icmp ult i64 %value1, %value2
  br i1 %cmp, label %smaller, label %greater

  smaller:
  br label %end

  greater:
  store i64 undef, i64 *undef
  br label %end

  end:
  %constant = add i64 %value0, 0
  unreachable
  )LLVM";

  ExpectedFlagsType ExpectedFlags{
    BBAssignmentFlags{ /*.BBName =*/"initial_block",
                       /*.InstrFlags =*/
                       {
                         None,
                         HasInterferingSideEffects,
                         None,
                         None,
                         None,
                         None,
                         None,
                         None,
                       } },
    BBAssignmentFlags{ /*.BBName =*/"smaller",
                       /*.InstrFlags =*/{ None } },
    BBAssignmentFlags{ /*.BBName =*/"greater",
                       /*.InstrFlags =*/{ HasSideEffects, None } },
    BBAssignmentFlags{ /*.BBName =*/"end",
                       /*.InstrFlags =*/{ AlwaysAssign, None } },
  };

  runTestOnFunctionWithExpected(Body, ExpectedFlags);
}

BOOST_AUTO_TEST_CASE(Loop) {
  const char *Body = R"LLVM(
  %initptr = inttoptr i64 4294967000 to i64*
  %initval = load i64, i64 *%initptr
  store i64 %initval, i64 * inttoptr (i64 1000 to i64*)
  br label %head

  head:
  %loopingvar = load i64, i64 * inttoptr (i64 1000 to i64*)
  br label %tail

  tail:
  %loopptr = inttoptr i64 4294967200 to i64*
  %loopval = load i64, i64 *%loopptr
  %cmp = icmp ult i64 undef, %loopingvar
  store i64 %loopval, i64 * inttoptr (i64 1000 to i64*)
  br i1 %cmp, label %head, label %end

  end:
  unreachable
  )LLVM";

  ExpectedFlagsType ExpectedFlags{
    BBAssignmentFlags{ /*.BBName =*/"initial_block",
                       /*.InstrFlags =*/
                       {
                         None,
                         None,
                         HasSideEffects,
                         None,
                       } },
    BBAssignmentFlags{ /*.BBName =*/"head",
                       /*.InstrFlags =*/{ None, None } },
    BBAssignmentFlags{ /*.BBName =*/"tail",
                       /*.InstrFlags =*/{ None,
                                          None,
                                          HasInterferingSideEffects,
                                          HasSideEffects,
                                          None } },
    BBAssignmentFlags{ /*.BBName =*/"end",
                       /*.InstrFlags =*/{ None } },
  };

  runTestOnFunctionWithExpected(Body, ExpectedFlags);
}
