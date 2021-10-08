/// \file MarkForSerializationTest.cpp
/// \brief Test for MarkForSerialization

//
// Copyright rev.ng Srls. See LICENSE.md for details.
//

#include <compare>
#include <map>
#include <set>
#include <string>
#include <vector>

#define BOOST_TEST_MODULE MarkForSerializationTest
bool init_unit_test();
#include "boost/test/unit_test.hpp"

#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"

#include "revng/Support/IRHelpers.h"
#include "revng/UnitTestHelpers/LLVMTestHelpers.h"

#include "revng-c/MarkForSerialization/MarkAnalysis.h"
#include "revng-c/MarkForSerialization/MarkForSerializationPass.h"

struct BBSerializationFlags {
  std::string BBName;
  std::vector<SerializationReason> InstrFlags;
};

using ExpectedDuplicatesType = std::vector<unsigned>;
using ExpectedFlagsType = std::vector<BBSerializationFlags>;

static void
runTestOnFunctionWithExpected(const char *Body,
                              const ExpectedDuplicatesType &ExpectedDups,
                              const ExpectedFlagsType &Expected) {
  MarkLog.enable();
  revng_log(MarkLog, "Running Test On Function: " << Body);

  llvm::LLVMContext C;
  std::unique_ptr<llvm::Module> M = loadModule(C, Body);
  revng_check(not llvm::verifyModule(*M, &llvm::dbgs()));

  // main is present and has a body
  const llvm::Function *F = M->getFunction("main");
  revng_check(nullptr != F);
  revng_check(not F->empty());

  SerializationMap Results;
  MarkAnalysis::Analysis Mark(*F, Results);
  Mark.initialize();
  Mark.run();

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

BOOST_AUTO_TEST_CASE(AlwaysSerializeNoUses) {

  const char *EmptyBody = R"LLVM(
  unreachable
  )LLVM";

  ExpectedDuplicatesType ExpectedDups{ 1 };

  ExpectedFlagsType ExpectedFlags{
    BBSerializationFlags{ "initial_block", { AlwaysSerialize } },
  };

  runTestOnFunctionWithExpected(EmptyBody, ExpectedDups, ExpectedFlags);
}

BOOST_AUTO_TEST_CASE(Ret) {

  const char *Body = R"LLVM(
  ret void
  )LLVM";

  ExpectedDuplicatesType ExpectedDups{ 1 };

  ExpectedFlagsType ExpectedFlags{
    BBSerializationFlags{ "initial_block", { AlwaysSerialize } },
  };

  runTestOnFunctionWithExpected(Body, ExpectedDups, ExpectedFlags);
}

BOOST_AUTO_TEST_CASE(CallSideEffects) {

  const char *Body = R"LLVM(
  %bswapped = call i64 @llvm.bswap.i64(i64 1231231231)
  unreachable
  )LLVM";

  ExpectedDuplicatesType ExpectedDups{ 1 };

  ExpectedFlagsType ExpectedFlags{
    BBSerializationFlags{
      "initial_block", { HasSideEffects | AlwaysSerialize, AlwaysSerialize } },
  };

  runTestOnFunctionWithExpected(Body, ExpectedDups, ExpectedFlags);
}

BOOST_AUTO_TEST_CASE(StoreSideEffects) {

  const char *Body = R"LLVM(
  store i64 123, i64 * inttoptr (i64 4294967296 to i64*)
  unreachable
  )LLVM";

  ExpectedDuplicatesType ExpectedDups{ 1 };

  ExpectedFlagsType ExpectedFlags{
    BBSerializationFlags{
      "initial_block", { HasSideEffects | AlwaysSerialize, AlwaysSerialize } },
  };

  runTestOnFunctionWithExpected(Body, ExpectedDups, ExpectedFlags);
}

BOOST_AUTO_TEST_CASE(AllocaNeedsLocalVar) {

  const char *Body = R"LLVM(
  %alloca = alloca i64, align 8
  unreachable
  )LLVM";

  ExpectedDuplicatesType ExpectedDups{ 1 };

  ExpectedFlagsType ExpectedFlags{
    BBSerializationFlags{
      "initial_block",
      { NeedsLocalVarToComputeExpr | AlwaysSerialize, AlwaysSerialize } },
  };

  runTestOnFunctionWithExpected(Body, ExpectedDups, ExpectedFlags);
}

BOOST_AUTO_TEST_CASE(InsertValue) {

  const char *Body = R"LLVM(
  %a = insertvalue {i64, i32} undef, i64 1, 0
  unreachable
  )LLVM";

  ExpectedDuplicatesType ExpectedDups{ 1 };

  ExpectedFlagsType ExpectedFlags{
    BBSerializationFlags{
      "initial_block",
      { NeedsLocalVarToComputeExpr | NeedsManyStatements | AlwaysSerialize,
        AlwaysSerialize } },
  };

  runTestOnFunctionWithExpected(Body, ExpectedDups, ExpectedFlags);
}

BOOST_AUTO_TEST_CASE(ManyUses) {

  const char *Body = R"LLVM(
  %pointer = inttoptr i64 4294967296 to i64*
  %loaded = load i64, i64 * %pointer
  %reloaded = load i64, i64 * %pointer
  unreachable
  )LLVM";

  ExpectedDuplicatesType ExpectedDups{ 1 };

  ExpectedFlagsType ExpectedFlags{
    BBSerializationFlags{
      "initial_block",
      { HasManyUses, AlwaysSerialize, AlwaysSerialize, AlwaysSerialize } },
  };

  runTestOnFunctionWithExpected(Body, ExpectedDups, ExpectedFlags);
}

BOOST_AUTO_TEST_CASE(BranchNone) {

  const char *Body = R"LLVM(
  br label %next

  next:
  unreachable
  )LLVM";

  ExpectedDuplicatesType ExpectedDups{ 1, 1 };

  ExpectedFlagsType ExpectedFlags{
    BBSerializationFlags{ "initial_block", { None } },
    BBSerializationFlags{ "next", { AlwaysSerialize } },
  };

  runTestOnFunctionWithExpected(Body, ExpectedDups, ExpectedFlags);
}

BOOST_AUTO_TEST_CASE(Interfering) {

  const char *Body = R"LLVM(
  %pointer = inttoptr i64 4294967296 to i64*
  %loaded = load i64, i64 * %pointer
  store i64 undef, i64 * %pointer
  store i64 %loaded, i64 * %pointer
  unreachable
  )LLVM";

  ExpectedDuplicatesType ExpectedDups{ 1 };

  ExpectedFlagsType ExpectedFlags{
    BBSerializationFlags{ "initial_block",
                          { HasManyUses,
                            HasInterferingSideEffects,
                            AlwaysSerialize | HasSideEffects,
                            AlwaysSerialize | HasSideEffects,
                            AlwaysSerialize } },
  };

  runTestOnFunctionWithExpected(Body, ExpectedDups, ExpectedFlags);
}

BOOST_AUTO_TEST_CASE(NonInterfering) {

  const char *Body = R"LLVM(
  %pointer = inttoptr i64 4294967296 to i64*
  %loaded = load i64, i64 * %pointer
  store i64 %loaded, i64 * %pointer
  unreachable
  )LLVM";

  ExpectedDuplicatesType ExpectedDups{ 1 };

  ExpectedFlagsType ExpectedFlags{
    BBSerializationFlags{ "initial_block",
                          { HasManyUses,
                            None,
                            HasSideEffects | AlwaysSerialize,
                            AlwaysSerialize } },
  };

  runTestOnFunctionWithExpected(Body, ExpectedDups, ExpectedFlags);
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

  ExpectedDuplicatesType ExpectedDups{ 1 };

  ExpectedFlagsType ExpectedFlags{
    BBSerializationFlags{ /*.BBName =*/"initial_block",
                          /*.InstrFlags =*/{ HasManyUses,
                                             None,
                                             None,
                                             None,
                                             AlwaysSerialize | HasSideEffects,
                                             AlwaysSerialize } },
  };

  runTestOnFunctionWithExpected(Body, ExpectedDups, ExpectedFlags);
}

BOOST_AUTO_TEST_CASE(DoubleIndirectionWithoutSideEffects) {

  const char *Body = R"LLVM(
  %ptrtoptr = inttoptr i64 4294967296 to i64**
  %ptr = load i64*, i64 **%ptrtoptr
  %value = load i64, i64 *%ptr
  %call = call i64 @llvm.bswap.i64(i64 %value)
  unreachable
  )LLVM";

  ExpectedDuplicatesType ExpectedDups{ 1 };

  ExpectedFlagsType ExpectedFlags{
    BBSerializationFlags{ /*.BBName =*/"initial_block",
                          /*.InstrFlags =*/{ None,
                                             None,
                                             None,
                                             AlwaysSerialize | HasSideEffects,
                                             AlwaysSerialize } },
  };

  runTestOnFunctionWithExpected(Body, ExpectedDups, ExpectedFlags);
}

BOOST_AUTO_TEST_CASE(DoubleIndirectionWithSideEffects) {

  const char *Body = R"LLVM(
  %ptrtoptr = inttoptr i64 4294967296 to i64**
  %ptr = load i64*, i64 **%ptrtoptr
  %call = call i64 @llvm.bswap.i64(i64 1032143324)
  %value = load i64, i64 *%ptr
  %othercall = call i64 @llvm.bswap.i64(i64 %value)
  unreachable
  )LLVM";

  ExpectedDuplicatesType ExpectedDups{ 1 };

  ExpectedFlagsType ExpectedFlags{
    BBSerializationFlags{ /*.BBName =*/"initial_block",
                          /*.InstrFlags =*/{ None,
                                             HasInterferingSideEffects,
                                             AlwaysSerialize | HasSideEffects,
                                             None,
                                             AlwaysSerialize | HasSideEffects,
                                             AlwaysSerialize } },
  };

  runTestOnFunctionWithExpected(Body, ExpectedDups, ExpectedFlags);
}

BOOST_AUTO_TEST_CASE(ConditionalIsNotSerialized1) {
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

  ExpectedDuplicatesType ExpectedDups{ 1, 1, 1, 1 };

  ExpectedFlagsType ExpectedFlags{
    BBSerializationFlags{ /*.BBName =*/"initial_block",
                          /*.InstrFlags =*/
                          {
                            HasManyUses,
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                          } },
    BBSerializationFlags{ /*.BBName =*/"smaller",
                          /*.InstrFlags =*/{ None } },
    BBSerializationFlags{ /*.BBName =*/"greater",
                          /*.InstrFlags =*/{ None } },
    BBSerializationFlags{
      /*.BBName =*/"end",
      /*.InstrFlags =*/{ HasSideEffects | AlwaysSerialize, AlwaysSerialize } },
  };

  runTestOnFunctionWithExpected(Body, ExpectedDups, ExpectedFlags);
}

BOOST_AUTO_TEST_CASE(ConditionalIsNotSerialized2) {
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

  ExpectedDuplicatesType ExpectedDups{ 1, 1, 1, 1 };

  ExpectedFlagsType ExpectedFlags{
    BBSerializationFlags{ /*.BBName =*/"initial_block",
                          /*.InstrFlags =*/
                          {
                            HasManyUses,
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                          } },
    BBSerializationFlags{ /*.BBName =*/"smaller",
                          /*.InstrFlags =*/{ None } },
    BBSerializationFlags{
      /*.BBName =*/"greater",
      /*.InstrFlags =*/{ HasSideEffects | AlwaysSerialize, None } },
    BBSerializationFlags{ /*.BBName =*/"end",
                          /*.InstrFlags =*/{ AlwaysSerialize } },
  };

  runTestOnFunctionWithExpected(Body, ExpectedDups, ExpectedFlags);
}

BOOST_AUTO_TEST_CASE(SerializePendingInThenBranch) {
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

  ExpectedDuplicatesType ExpectedDups{ 1, 1, 1, 1 };

  ExpectedFlagsType ExpectedFlags{
    BBSerializationFlags{ /*.BBName =*/"initial_block",
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
    BBSerializationFlags{
      /*.BBName =*/"smaller",
      /*.InstrFlags =*/{ HasSideEffects | AlwaysSerialize, None } },
    BBSerializationFlags{ /*.BBName =*/"greater",
                          /*.InstrFlags =*/{ None } },
    BBSerializationFlags{
      /*.BBName =*/"end",
      /*.InstrFlags =*/{ AlwaysSerialize, AlwaysSerialize } },
  };

  runTestOnFunctionWithExpected(Body, ExpectedDups, ExpectedFlags);
}

BOOST_AUTO_TEST_CASE(SerializePendingInElseBranch) {
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

  ExpectedDuplicatesType ExpectedDups{ 1, 1, 1, 1 };

  ExpectedFlagsType ExpectedFlags{
    BBSerializationFlags{ /*.BBName =*/"initial_block",
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
    BBSerializationFlags{ /*.BBName =*/"smaller",
                          /*.InstrFlags =*/{ None } },
    BBSerializationFlags{
      /*.BBName =*/"greater",
      /*.InstrFlags =*/{ HasSideEffects | AlwaysSerialize, None } },
    BBSerializationFlags{
      /*.BBName =*/"end",
      /*.InstrFlags =*/{ AlwaysSerialize, AlwaysSerialize } },
  };

  runTestOnFunctionWithExpected(Body, ExpectedDups, ExpectedFlags);
}

BOOST_AUTO_TEST_CASE(Loop) {
  const char *Body = R"LLVM(
  %initptr = inttoptr i64 4294967000 to i64*
  %initval = load i64, i64 *%initptr
  br label %head

  head:
  %loopingvar = phi i64 [ %initval, %initial_block ], [ %loopval, %tail ]
  br label %tail

  tail:
  %loopptr = inttoptr i64 4294967200 to i64*
  %loopval = load i64, i64 *%loopptr
  %cmp = icmp ult i64 undef, %loopingvar
  br i1 %cmp, label %head, label %end

  end:
  unreachable
  )LLVM";

  ExpectedDuplicatesType ExpectedDups{ 1, 1, 1, 1 };

  ExpectedFlagsType ExpectedFlags{
    BBSerializationFlags{ /*.BBName =*/"initial_block",
                          /*.InstrFlags =*/
                          {
                            None,
                            None,
                            None,
                          } },
    BBSerializationFlags{ /*.BBName =*/"head",
                          /*.InstrFlags =*/{ None, None } },
    BBSerializationFlags{ /*.BBName =*/"tail",
                          /*.InstrFlags =*/{ None, None, None, None } },
    BBSerializationFlags{ /*.BBName =*/"end",
                          /*.InstrFlags =*/{ AlwaysSerialize } },
  };

  runTestOnFunctionWithExpected(Body, ExpectedDups, ExpectedFlags);
}
