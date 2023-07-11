/// \file Clift.cpp
/// Tests for the Clift Dialect

//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

#include <cstdlib>

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "revng-c/mlir/Dialect/Clift/IR/Clift.h"
#include "revng-c/mlir/Dialect/Clift/IR/CliftOps.h"

#define BOOST_TEST_MODULE Clift
bool init_unit_test();
#include "boost/test/unit_test.hpp"

class CliftTest {
public:
  CliftTest() :
    module(mlir::ModuleOp::create(mlir::UnknownLoc::get(&context))),
    builder(module.getBodyRegion()) {

    registry.insert<mlir::clift::CliftDialect>();
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  }

  void canonicalize() {
    mlir::PassManager manager(&context);
    manager.addPass(mlir::createCanonicalizerPass());
    BOOST_ASSERT(manager.run(module).succeeded());
  }

protected:
  mlir::DialectRegistry registry;
  mlir::MLIRContext context;
  std::unique_ptr<mlir::Diagnostic> diagnostic;
  mlir::ModuleOp module;
  mlir::OpBuilder builder;
};

BOOST_FIXTURE_TEST_SUITE(CliftTestSuite, CliftTest)
BOOST_AUTO_TEST_CASE(LabelsWithoutGoToMustBeTriviallyDead) {
  auto label = builder
                 .create<mlir::clift::MakeLabelOp>(builder.getUnknownLoc());
  builder.create<mlir::clift::AssignLabelOp>(builder.getUnknownLoc(), label);

  BOOST_ASSERT(not module.getBody()->getOperations().empty());

  canonicalize();
  BOOST_ASSERT(module.getBody()->getOperations().empty());
}

BOOST_AUTO_TEST_CASE(LabelsWithGoToMustBeAlive) {
  auto label = builder
                 .create<mlir::clift::MakeLabelOp>(builder.getUnknownLoc());
  builder.create<mlir::clift::AssignLabelOp>(builder.getUnknownLoc(), label);
  builder.create<mlir::clift::GoToOp>(builder.getUnknownLoc(), label);

  BOOST_CHECK(not module.getBody()->getOperations().empty());

  canonicalize();
  BOOST_TEST(module.getBody()->getOperations().size() == 3);
}

BOOST_AUTO_TEST_CASE(LabelsWithAGoToWithoutAssignMustFail) {
  auto label = builder
                 .create<mlir::clift::MakeLabelOp>(builder.getUnknownLoc());
  builder.create<mlir::clift::GoToOp>(builder.getUnknownLoc(), label);

  BOOST_TEST(mlir::verify(module).failed());
}

BOOST_AUTO_TEST_SUITE_END()
