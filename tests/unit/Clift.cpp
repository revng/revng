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

#include "revng-c/mlir/Dialect/Clift/IR/Clift.h"
#include "revng-c/mlir/Dialect/Clift/IR/CliftAttributes.h"
#include "revng-c/mlir/Dialect/Clift/IR/CliftOps.h"

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

BOOST_AUTO_TEST_CASE(CliftModuleCannotContainExtraneousTypes) {
  auto cliftModule = builder
                       .create<mlir::clift::ModuleOp>(builder.getUnknownLoc());
  builder.createBlock(&cliftModule.getBody());
  cliftModule->setAttr("dc",
                       mlir::TypeAttr::get(mlir::IndexType::get(&context)));
  BOOST_TEST(cliftModule.verify().failed());
}

BOOST_AUTO_TEST_CASE(CliftModuleCannotContaintTypesWithSameID) {
  using namespace mlir::clift;
  auto cliftModule = builder.create<ModuleOp>(builder.getUnknownLoc());
  builder.createBlock(&cliftModule.getBody());
  auto TrueAttr = mlir::BoolAttr::get(&context, true);
  auto T1 = mlir::TypeAttr::get(DefinedType::get(&context,
                                                 StructType::get(&context, 1),
                                                 TrueAttr));
  auto T2 = mlir::TypeAttr::get(DefinedType::get(&context,
                                                 UnionType::get(&context, 1),
                                                 TrueAttr));
  cliftModule->setAttr("dc", T1);
  cliftModule->setAttr("dc2", T2);
  BOOST_TEST(cliftModule.verify().failed());
}

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

BOOST_AUTO_TEST_CASE(UnionAndStructsCantContainThemself) {
  auto UnionAttrT = mlir::clift::UnionType::get(builder.getContext(), 0);
  using DefinedT = mlir::clift::DefinedType;
  auto UnionT = DefinedT::get(builder.getContext(),
                              UnionAttrT,
                              mlir::BoolAttr::get(builder.getContext(), false));

  // Just check that you can't make a field out of a forward declared type, a
  // pointer to the type must be used instead.
  BOOST_TEST(mlir::clift::FieldAttr::verify(getDiagnosticEmitter(),
                                            0,
                                            UnionT,
                                            "field")
               .failed());
}

BOOST_AUTO_TEST_CASE(UnionAndStructsCantContainFunctions) {
  using namespace mlir::clift;
  auto VoidT = PrimitiveType::getVoid(builder.getContext(), 0);
  auto FunctionT = FunctionAttr::get(0, VoidT);
  auto UnionT = DefinedType::get(builder.getContext(),
                                 FunctionT,
                                 mlir::BoolAttr::get(builder.getContext(),
                                                     false));

  BOOST_TEST(mlir::clift::FieldAttr::verify(getDiagnosticEmitter(),
                                            0,
                                            UnionT,
                                            "field")
               .failed());
}

BOOST_AUTO_TEST_CASE(FunctionTypesCantContainVoidArgs) {
  auto VoidType = mlir::clift::PrimitiveType::getVoid(builder.getContext(), 0);

  BOOST_TEST(mlir::clift::FunctionArgumentAttr::verify(getDiagnosticEmitter(),
                                                       VoidType,
                                                       "")
               .failed());
}

BOOST_AUTO_TEST_CASE(FunctionTypesCantContainFunctionTypes) {
  using namespace mlir::clift;
  auto FunctionT = FunctionAttr::get(0,
                                     PrimitiveType::getVoid(builder
                                                              .getContext(),
                                                            0));

  using DefinedT = mlir::clift::DefinedType;
  auto DefinedType = DefinedT::get(builder.getContext(),
                                   FunctionT,
                                   mlir::BoolAttr::get(builder.getContext(),
                                                       false));

  BOOST_TEST(mlir::clift::FunctionAttr::verify(getDiagnosticEmitter(),
                                               1,
                                               "",
                                               DefinedType,
                                               {})
               .failed());
}

using namespace mlir::clift;
BOOST_AUTO_TEST_CASE(PrimitiveTypesDefaultToNonConst) {
  auto Type = PrimitiveType::get(&context,
                                 mlir::clift::PrimitiveKind::GenericKind,
                                 4,
                                 mlir::BoolAttr::get(&context, false));
  BOOST_TEST((Type.isConst() == false));
}

BOOST_AUTO_TEST_CASE(CanWalkPointerTypes) {
  auto Type = PrimitiveType::get(&context,
                                 mlir::clift::PrimitiveKind::GenericKind,
                                 4,
                                 mlir::BoolAttr::get(&context, false));

  auto Ptr = mlir::clift::PointerType::get(&context,
                                           Type,
                                           8,
                                           mlir::BoolAttr::get(&context,
                                                               false));
  size_t Count = 0;
  Ptr.walkSubTypes([&](mlir::Type Underlying) {
    BOOST_TEST((Underlying == Type));
    Count++;
  });
  BOOST_TEST(Count == 1);
}

BOOST_AUTO_TEST_SUITE_END()
