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

BOOST_AUTO_TEST_CASE(CliftModuleCannotContainExtraneousTypes) {
  auto cliftModule = builder.create<ModuleOp>(builder.getUnknownLoc());
  builder.createBlock(&cliftModule.getBody());
  cliftModule->setAttr("dc",
                       mlir::TypeAttr::get(mlir::IndexType::get(&context)));
  BOOST_TEST(cliftModule.verify().failed());
}

BOOST_AUTO_TEST_CASE(CliftModuleCannotContaintTypesWithSameID) {
  auto cliftModule = builder.create<ModuleOp>(builder.getUnknownLoc());
  builder.createBlock(&cliftModule.getBody());
  auto TrueAttr = mlir::BoolAttr::get(&context, true);
  auto T1 = mlir::TypeAttr::get(DefinedType::get(&context,
                                                 StructTypeAttr::get(&context,
                                                                     1),
                                                 TrueAttr));
  auto T2 = mlir::TypeAttr::get(DefinedType::get(&context,
                                                 UnionTypeAttr::get(&context,
                                                                    1),
                                                 TrueAttr));
  cliftModule->setAttr("dc", T1);
  cliftModule->setAttr("dc2", T2);
  BOOST_TEST(cliftModule.verify().failed());
}

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

BOOST_AUTO_TEST_CASE(UnionAndStructsCantDirectlyContainThemselves) {
  auto UnionAttrT = UnionTypeAttr::get(builder.getContext(), 0);
  auto UnionT = DefinedType::get(builder.getContext(),
                                 UnionAttrT,
                                 mlir::BoolAttr::get(builder.getContext(),
                                                     false));

  FieldAttr UnionFields[] = {
    FieldAttr::get(builder.getContext(), 0, UnionT, "x")
  };
  (void) UnionTypeAttr::get(builder.getContext(), 0, "u", UnionFields);

  auto CliftModule = builder.create<ModuleOp>(builder.getUnknownLoc());
  mlir::OpBuilder CliftModuleBuilder(CliftModule.getBody());
  CliftModuleBuilder.create<UndefOp>(builder.getUnknownLoc(), UnionT);
  BOOST_TEST(mlir::verify(CliftModule).failed());
}

BOOST_AUTO_TEST_CASE(UnionAndStructsCantIndirectlyContainThemselves) {
  auto UnionAttrT = UnionTypeAttr::get(builder.getContext(), 0);
  auto UnionT = DefinedType::get(builder.getContext(),
                                 UnionAttrT,
                                 mlir::BoolAttr::get(builder.getContext(),
                                                     false));

  auto StructAttrT = StructTypeAttr::get(builder.getContext(), 1);
  auto StructT = DefinedType::get(builder.getContext(),
                                  StructAttrT,
                                  mlir::BoolAttr::get(builder.getContext(),
                                                      false));

  FieldAttr UnionFields[] = {
    FieldAttr::get(builder.getContext(), 0, StructT, "x")
  };
  (void) UnionTypeAttr::get(builder.getContext(), 0, "u", UnionFields);

  FieldAttr StructFields[] = {
    FieldAttr::get(builder.getContext(), 0, UnionT, "y")
  };
  (void) StructTypeAttr::get(builder.getContext(), 1, "s", 1, StructFields);

  auto CliftModule = builder.create<ModuleOp>(builder.getUnknownLoc());
  mlir::OpBuilder CliftModuleBuilder(CliftModule.getBody());
  CliftModuleBuilder.create<UndefOp>(builder.getUnknownLoc(), StructT);
  BOOST_TEST(mlir::verify(CliftModule).failed());
}

BOOST_AUTO_TEST_CASE(UnionAndStructsCanContainThemselvesViaIndirection) {
  auto UnionAttrT = UnionTypeAttr::get(builder.getContext(), 0);
  auto UnionT = DefinedType::get(builder.getContext(),
                                 UnionAttrT,
                                 mlir::BoolAttr::get(builder.getContext(),
                                                     false));

  auto UnionPtrT = PointerType::get(builder.getContext(),
                                    UnionT,
                                    8,
                                    mlir::BoolAttr::get(builder.getContext(),
                                                        false));

  FieldAttr UnionFields[] = {
    FieldAttr::get(builder.getContext(), 0, UnionPtrT, "p")
  };
  (void) UnionTypeAttr::get(builder.getContext(), 0, "u", UnionFields);

  auto CliftModule = builder.create<ModuleOp>(builder.getUnknownLoc());
  mlir::OpBuilder CliftModuleBuilder(CliftModule.getBody());
  CliftModuleBuilder.create<UndefOp>(builder.getUnknownLoc(), UnionT);
  BOOST_TEST(mlir::verify(CliftModule).succeeded());
}

BOOST_AUTO_TEST_CASE(UnionAndStructsCantContainFunctions) {
  auto VoidT = PrimitiveType::getVoid(builder.getContext(), 0);
  auto FunctionT = DefinedType::get(builder.getContext(),
                                    FunctionTypeAttr::get(0, VoidT),
                                    mlir::BoolAttr::get(builder.getContext(),
                                                        false));

  BOOST_TEST(FieldAttr::verify(getDiagnosticEmitter(), 0, FunctionT, "field")
               .failed());
}

BOOST_AUTO_TEST_CASE(FunctionTypesCantContainVoidArgs) {
  auto VoidType = PrimitiveType::getVoid(builder.getContext(), 0);

  BOOST_TEST(FunctionArgumentAttr::verify(getDiagnosticEmitter(), VoidType, "")
               .failed());
}

BOOST_AUTO_TEST_CASE(FunctionTypesCantContainFunctionTypes) {
  auto FunctionT = FunctionTypeAttr::get(0,
                                         PrimitiveType::getVoid(builder
                                                                  .getContext(),
                                                                0));

  auto DefinedT = DefinedType::get(builder.getContext(),
                                   FunctionT,
                                   mlir::BoolAttr::get(builder.getContext(),
                                                       false));

  BOOST_TEST(FunctionTypeAttr::verify(getDiagnosticEmitter(),
                                      1,
                                      "",
                                      DefinedT,
                                      {})
               .failed());
}

using namespace mlir::clift;
BOOST_AUTO_TEST_CASE(PrimitiveTypesDefaultToNonConst) {
  auto Type = PrimitiveType::get(&context,
                                 PrimitiveKind::GenericKind,
                                 4,
                                 mlir::BoolAttr::get(&context, false));
  BOOST_TEST((Type.isConst() == false));
}

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

BOOST_AUTO_TEST_SUITE_END()
