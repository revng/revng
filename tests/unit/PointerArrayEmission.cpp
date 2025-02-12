/// \file PointerArrayEmission.cpp
/// Tests `getNamedCInstance`

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#define BOOST_TEST_MODULE PointerArrayEmission
bool init_unit_test();
#include "boost/test/unit_test.hpp"

#include "revng/Model/Binary.h"
#include "revng/Support/Assert.h"
#include "revng/TypeNames/PTMLCTypeBuilder.h"
#include "revng/UnitTestHelpers/UnitTestHelpers.h"

// Tests whether the way we emit pointers and arrays is reasonable.
// Each layer of the types builds on the previous ones, so the test get more
// complex gradually.
//
// \note the easiest way to understand one of these is to see it fail: just edit
//       one of the strings and watch what it prints: it'll show what
//       we emitted, what we should have, and how the type in question looks
//       like in the model.
//
// \note also, notice that there is no special logic surrounding function
//       pointers, it's because we emit those as typedef, which leads to them
//       having a concrete name to fall onto, making their emission trivial.
BOOST_AUTO_TEST_CASE(PointerArrayEmission) {
  // Maps types to their expected output.
  std::vector<std::pair<const model::UpcastableType &, std::string>> Tests;

  TupleTree<model::Binary> Binary = {};

  auto Void = model::PrimitiveType::makeVoid();
  Tests.emplace_back(Void, "void test");

  auto Int = model::PrimitiveType::makeConstSigned(4);
  Tests.emplace_back(Int, "const int32_t test");

  auto &&[TypedefDef, Typedef] = Binary->makeTypedefDefinition(Void.copy());
  Tests.emplace_back(Typedef, "typedef_0 test");

  auto VoidP = model::PointerType::make(Void.copy(), 8);
  Tests.emplace_back(VoidP, "void *test");
  auto VoidCP = model::PointerType::makeConst(Void.copy(), 8);
  Tests.emplace_back(VoidCP, "void *const test");
  auto IntP = model::PointerType::make(Int.copy(), 8);
  Tests.emplace_back(IntP, "const int32_t *test");
  auto IntCP = model::PointerType::makeConst(Int.copy(), 8);
  Tests.emplace_back(IntCP, "const int32_t *const test");
  auto TypedefP = model::PointerType::make(Typedef.copy(), 8);
  Tests.emplace_back(TypedefP, "typedef_0 *test");
  auto TypedefCP = model::PointerType::makeConst(Typedef.copy(), 8);
  Tests.emplace_back(TypedefCP, "typedef_0 *const test");

  auto VoidPA = model::ArrayType::make(VoidP.copy(), 15);
  Tests.emplace_back(VoidPA, "void *test[15]");
  auto VoidCPA = model::ArrayType::make(VoidCP.copy(), 16);
  Tests.emplace_back(VoidCPA, "void *const test[16]");
  auto IntPA = model::ArrayType::make(IntP.copy(), 17);
  Tests.emplace_back(IntPA, "const int32_t *test[17]");
  auto IntCPA = model::ArrayType::make(IntCP.copy(), 18);
  Tests.emplace_back(IntCPA, "const int32_t *const test[18]");
  auto TypedefPA = model::ArrayType::make(TypedefP.copy(), 19);
  Tests.emplace_back(TypedefPA, "typedef_0 *test[19]");
  auto TypedefCPA = model::ArrayType::make(TypedefCP.copy(), 20);
  Tests.emplace_back(TypedefCPA, "typedef_0 *const test[20]");

  auto VoidPAA = model::ArrayType::make(VoidPA.copy(), 42);
  Tests.emplace_back(VoidPAA, "void *test[42][15]");
  auto VoidCPAA = model::ArrayType::make(VoidCPA.copy(), 41);
  Tests.emplace_back(VoidCPAA, "void *const test[41][16]");
  auto IntPAA = model::ArrayType::make(IntPA.copy(), 40);
  Tests.emplace_back(IntPAA, "const int32_t *test[40][17]");
  auto IntCPAA = model::ArrayType::make(IntCPA.copy(), 39);
  Tests.emplace_back(IntCPAA, "const int32_t *const test[39][18]");
  auto TypedefPAA = model::ArrayType::make(TypedefPA.copy(), 38);
  Tests.emplace_back(TypedefPAA, "typedef_0 *test[38][19]");
  auto TypedefCPAA = model::ArrayType::make(TypedefCPA.copy(), 37);
  Tests.emplace_back(TypedefCPAA, "typedef_0 *const test[37][20]");

  auto VoidPAACP = model::PointerType::makeConst(VoidPAA.copy(), 8);
  Tests.emplace_back(VoidPAACP,
                     "artificial_wrapper_"
                     "array_42_of_"
                     "array_15_of_"
                     "ptr_to_"
                     "void *const test");
  auto VoidCPAACP = model::PointerType::makeConst(VoidCPAA.copy(), 8);
  Tests.emplace_back(VoidCPAACP,
                     "artificial_wrapper_"
                     "array_41_of_"
                     "array_16_of_"
                     "const_ptr_to_"
                     "void *const test");
  auto IntPAAP = model::PointerType::make(IntPAA.copy(), 8);
  Tests.emplace_back(IntPAAP,
                     "artificial_wrapper_"
                     "array_40_of_"
                     "array_17_of_"
                     "ptr_to_"
                     "const_int32_t *test");
  auto IntCPAAP = model::PointerType::make(IntCPAA.copy(), 8);
  Tests.emplace_back(IntCPAAP,
                     "artificial_wrapper_"
                     "array_39_of_"
                     "array_18_of_"
                     "const_ptr_to_"
                     "const_int32_t *test");
  auto TypedefPAACP = model::PointerType::makeConst(TypedefPAA.copy(), 8);
  Tests.emplace_back(TypedefPAACP,
                     "artificial_wrapper_"
                     "array_38_of_"
                     "array_19_of_"
                     "ptr_to_"
                     "typedef_0 *const test");
  auto TypedefCPAAP = model::PointerType::make(TypedefCPAA.copy(), 8);
  Tests.emplace_back(TypedefCPAAP,
                     "artificial_wrapper_"
                     "array_37_of_"
                     "array_20_of_"
                     "const_ptr_to_"
                     "typedef_0 *test");

  auto FinalV = model::ArrayType::make(VoidPAACP.copy(), 1);
  Tests.emplace_back(FinalV,
                     "artificial_wrapper_"
                     "array_42_of_"
                     "array_15_of_"
                     "ptr_to_"
                     "void *const test[1]");
  auto FinalVC = model::ArrayType::make(VoidCPAACP.copy(), 2);
  Tests.emplace_back(FinalVC,
                     "artificial_wrapper_"
                     "array_41_of_"
                     "array_16_of_"
                     "const_ptr_to_"
                     "void *const test[2]");
  auto FinalI = model::ArrayType::make(IntPAAP.copy(), 3);
  Tests.emplace_back(FinalI,
                     "artificial_wrapper_"
                     "array_40_of_"
                     "array_17_of_"
                     "ptr_to_"
                     "const_int32_t *test[3]");
  auto FinalIC = model::ArrayType::make(IntCPAAP.copy(), 4);
  Tests.emplace_back(FinalIC,
                     "artificial_wrapper_"
                     "array_39_of_"
                     "array_18_of_"
                     "const_ptr_to_"
                     "const_int32_t *test[4]");
  auto FinalT = model::ArrayType::make(TypedefPAACP.copy(), 5);
  Tests.emplace_back(FinalT,
                     "artificial_wrapper_"
                     "array_38_of_"
                     "array_19_of_"
                     "ptr_to_"
                     "typedef_0 *const test[5]");
  auto FinalTC = model::ArrayType::make(TypedefCPAAP.copy(), 6);
  Tests.emplace_back(FinalTC,
                     "artificial_wrapper_"
                     "array_37_of_"
                     "array_20_of_"
                     "const_ptr_to_"
                     "typedef_0 *test[6]");

  auto Extra1 = model::PointerType::makeConst(FinalIC.copy(), 8);
  Tests.emplace_back(Extra1,
                     "artificial_wrapper_"
                     "array_4_of_"
                     "ptr_to_"
                     "array_39_of_"
                     "array_18_of_"
                     "const_ptr_to_"
                     "const_int32_t *const test");
  auto Extra2 = model::PointerType::makeConst(Extra1.copy(), 8);
  Tests.emplace_back(Extra2,
                     "artificial_wrapper_"
                     "array_4_of_"
                     "ptr_to_"
                     "array_39_of_"
                     "array_18_of_"
                     "const_ptr_to_"
                     "const_int32_t *const *const test");
  auto Extra3 = model::PointerType::make(Extra2.copy(), 8);
  Tests.emplace_back(Extra3,
                     "artificial_wrapper_"
                     "array_4_of_"
                     "ptr_to_"
                     "array_39_of_"
                     "array_18_of_"
                     "const_ptr_to_"
                     "const_int32_t *const *const *test");
  auto Extra4 = model::PointerType::make(Extra3.copy(), 8);
  Tests.emplace_back(Extra4,
                     "artificial_wrapper_"
                     "array_4_of_"
                     "ptr_to_"
                     "array_39_of_"
                     "array_18_of_"
                     "const_ptr_to_"
                     "const_int32_t *const *const **test");
  auto Extra5 = model::PointerType::make(Extra4.copy(), 8);
  Tests.emplace_back(Extra5,
                     "artificial_wrapper_"
                     "array_4_of_"
                     "ptr_to_"
                     "array_39_of_"
                     "array_18_of_"
                     "const_ptr_to_"
                     "const_int32_t *const *const ***test");
  auto Extra6 = model::PointerType::makeConst(Extra5.copy(), 8);
  Tests.emplace_back(Extra6,
                     "artificial_wrapper_"
                     "array_4_of_"
                     "ptr_to_"
                     "array_39_of_"
                     "array_18_of_"
                     "const_ptr_to_"
                     "const_int32_t *const *const ****const test");

  std::string FailureLog;
  ptml::CTypeBuilder B(llvm::nulls(), *Binary, /* EnableTaglessMode = */ true);
  for (auto &&[Type, ExpectedOutput] : Tests) {
    std::string ActualOutput = B.getNamedCInstance(*Type, "test").str().str();
    if (ActualOutput != ExpectedOutput) {
      FailureLog += "Output of `getNamedCInstance` (\"" + ActualOutput
                    + "\")\n";
      FailureLog += "didn't match the expectations (\"" + ExpectedOutput
                    + "\")\n";
      FailureLog += "for\n" + toString(Type) + "\n\n";
    }
  }

  revng_check(FailureLog.empty(), FailureLog.c_str());
}
