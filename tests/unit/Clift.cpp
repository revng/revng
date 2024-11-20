/// \file Clift.cpp
/// Tests for the Clift Dialect

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <cstdlib>

#define BOOST_TEST_MODULE Clift
bool init_unit_test();
#include "boost/test/unit_test.hpp"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "revng/UnitTestHelpers/UnitTestHelpers.h"
#include "revng/mlir/Dialect/Clift/IR/Clift.h"
#include "revng/mlir/Dialect/Clift/IR/CliftAttributes.h"
#include "revng/mlir/Dialect/Clift/IR/CliftOps.h"

using namespace mlir::clift;

class CliftTest {
public:
  CliftTest() :
    module(mlir::ModuleOp::create(mlir::UnknownLoc::get(&context))),
    builder(module.getBodyRegion()) {

    registry.insert<CliftDialect>();
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  }

  void canonicalize() {
    mlir::PassManager manager(&context);
    manager.addPass(mlir::createCanonicalizerPass());
    BOOST_ASSERT(manager.run(module).succeeded());
  }

  auto getDiagnosticEmitter() {
    return [&]() { return module.emitError(); };
  }

protected:
  mlir::DialectRegistry registry;
  mlir::MLIRContext context;
  std::unique_ptr<mlir::Diagnostic> diagnostic;
  mlir::ModuleOp module;
  mlir::OpBuilder builder;
};

BOOST_FIXTURE_TEST_SUITE(CliftTestSuite, CliftTest)

//===-------------------------------- Types -------------------------------===//

BOOST_AUTO_TEST_CASE(CanWalkPointerTypes) {
  auto Type = PrimitiveType::get(&context,
                                 PrimitiveKind::GenericKind,
                                 4,
                                 mlir::BoolAttr::get(&context, false));

  auto Ptr = PointerType::get(&context,
                              Type,
                              8,
                              mlir::BoolAttr::get(&context, false));
  size_t Count = 0;
  Ptr.walkSubTypes([&](mlir::Type Underlying) {
    BOOST_TEST((Underlying == Type));
    Count++;
  });
  BOOST_TEST(Count == 1);
}

//===----------------------------- Statements -----------------------------===//

BOOST_AUTO_TEST_CASE(LabelsWithoutGoToMustBeTriviallyDead) {
  auto label = builder.create<MakeLabelOp>(builder.getUnknownLoc());
  builder.create<AssignLabelOp>(builder.getUnknownLoc(), label);

  BOOST_ASSERT(not module.getBody()->getOperations().empty());

  canonicalize();
  BOOST_ASSERT(module.getBody()->getOperations().empty());
}

BOOST_AUTO_TEST_CASE(LabelsWithGoToMustBeAlive) {
  auto label = builder.create<MakeLabelOp>(builder.getUnknownLoc());
  builder.create<AssignLabelOp>(builder.getUnknownLoc(), label);
  builder.create<GoToOp>(builder.getUnknownLoc(), label);

  BOOST_CHECK(not module.getBody()->getOperations().empty());

  canonicalize();
  BOOST_TEST(module.getBody()->getOperations().size() == 3);
}

BOOST_AUTO_TEST_CASE(LabelsWithAGoToWithoutAssignMustFail) {
  auto label = builder.create<MakeLabelOp>(builder.getUnknownLoc());
  builder.create<GoToOp>(builder.getUnknownLoc(), label);

  BOOST_TEST(mlir::verify(module).failed());
}

BOOST_AUTO_TEST_CASE(SwitchWithMismatchedCaseValuesAndRegionsIsNotValid) {
  const int64_t CaseValues[] = { 1 };
  builder.create<SwitchOp>(builder.getUnknownLoc(),
                           /*case_values=*/CaseValues,
                           /*casesCount=*/2);

  BOOST_TEST(mlir::verify(module).failed());
}

BOOST_AUTO_TEST_SUITE_END()
