/// \file Pipeline.cpp
/// \brief Tests for revng-pipeline

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <algorithm>
#include <memory>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include "revng/Pipeline/AllRegistries.h"
#include "revng/Pipeline/ContainerEnumerator.h"
#include "revng/Pipeline/ContainerFactorySet.h"
#include "revng/Pipeline/Context.h"
#include "revng/Pipeline/Contract.h"
#include "revng/Pipeline/Errors.h"
#include "revng/Pipeline/GenericLLVMPipe.h"
#include "revng/Pipeline/Kind.h"
#include "revng/Pipeline/LLVMContainerFactory.h"
#include "revng/Pipeline/LLVMGlobalKindBase.h"
#include "revng/Pipeline/Loader.h"
#include "revng/Pipeline/PathComponent.h"
#include "revng/Pipeline/Runner.h"
#include "revng/Pipeline/Target.h"
#include "revng/Support/Assert.h"

static char LLVMName = ' ';

using ExampleLLVMInspectalbeContainer = pipeline::LLVMContainerBase<&LLVMName>;

template<>
const char ExampleLLVMInspectalbeContainer::ID = '0';

#define BOOST_TEST_MODULE Pipeline
bool init_unit_test();
#include "boost/test/unit_test.hpp"

#include "revng/UnitTestHelpers/UnitTestHelpers.h"

using namespace pipeline;
using namespace std;
using namespace llvm;
using KE = Exactness::Values;

static Rank Root("Root");
static Rank FunctionRank("Function", Root);
class RootKindType : public LLVMKind {
public:
  RootKindType() : LLVMKind("RootKind", &Root) { revng_assert(depth() == 0); };

  std::optional<Target>
  symbolToTarget(const llvm::Function &Symbol) const override {
    if (Symbol.getName() == "root")
      return Target(*this);
    return std::nullopt;
  }

  TargetsList compactTargets(const Context &Ctx,
                             TargetsList::List &Targets) const override {
    return Targets;
  }

  ~RootKindType() override {}
};

static RootKindType RootKind;
static Kind RootKind2("RootKind2", RootKind, &Root);
static Kind RootKind3("RootKind3", &Root);
static Kind FunctionKind("FunctionKind", &FunctionRank);

static std::string CName = "ContainerName";

class MapContainer : public Container<MapContainer> {
public:
  MapContainer(std::map<Target, int> Map, llvm::StringRef Name) :
    Container<MapContainer>(Name, "application/x.test.map"),
    Map(std::move(Map)) {}
  MapContainer(llvm::StringRef Name) :
    Container<MapContainer>(Name, "application/x.test.map"), Map() {}

  ~MapContainer() override = default;

  unique_ptr<ContainerBase>
  cloneFiltered(const TargetsList &Container) const final {
    return make_unique<MapContainer>(Map, this->name());
  }

  bool contains(const Target &T) const {
    if (T.getPathComponents().size() != 0
        and T.getPathComponents().back().isAll())
      return Map.count(Target({ "f1" }, FunctionKind))
             and Map.count(Target({ "f2" }, FunctionKind));
    bool Contained = Map.count(T);
    return Contained;
  }

  const static Target AllTargets;

  TargetsList enumerate() const final {
    TargetsList ToReturn;

    if (contains(AllTargets)) {
      ToReturn.emplace_back(AllTargets);
      return ToReturn;
    }

    for (const auto &Target : Map)
      ToReturn.push_back(Target.first);

    return ToReturn;
  }

  bool remove(const TargetsList &Targets) override {

    if (Targets.contains(AllTargets)) {
      Map.clear();
      return true;
    }

    bool RemovedAll = true;
    for (const auto &Target : Targets)
      RemovedAll = remove(Target) && RemovedAll;

    return RemovedAll;
  }

  bool remove(const Target &Target) {
    if (Map.find(Target) == Map.end())
      return false;

    Map.erase(Target);
    return true;
  }

  llvm::Error
  extractOne(llvm::raw_ostream &OS, const Target &Target) const override {
    revng_abort();
    return llvm::Error::success();
  }

  static char ID;

  auto &get(Target Target) { return Map[std::move(Target)]; }
  const auto &get(const Target &Target) const {
    return Map.find(std::move(Target))->second;
  }
  auto &getMap() const { return Map; }
  auto &getMap() { return Map; }

  llvm::Error storeToDisk(llvm::StringRef Path) const override {
    SavedData = Map;
    return llvm::Error::success();
  }

  llvm::Error loadFromDisk(llvm::StringRef Path) override {
    Map = SavedData;
    return llvm::Error::success();
  }

  llvm::Error serialize(llvm::raw_ostream &OS) const final {

    return llvm::Error::success();
  }

  llvm::Error deserialize(const llvm::MemoryBuffer &Buffer) final {

    return llvm::Error::success();
  }

  /// Must reset the state of the container to the just built state
  void clear() final {}

private:
  std::map<Target, int> Map;
  mutable std::map<Target, int> SavedData;

private:
  void mergeBackImpl(MapContainer &&Container) override {
    Container.Map.merge(std::move(this->Map));
    this->Map = std::move(Container.Map);
  }
};

const Target MapContainer::AllTargets = Target({ PathComponent::all() },
                                               FunctionKind);

char MapContainer::ID;

static const Target ExampleTarget({}, RootKind);

struct Fixture {
  Fixture() {
    Rank::init();
    Kind::init();
  }
};

BOOST_AUTO_TEST_SUITE(PipelineTestSuite, *boost::unit_test::fixture<Fixture>())

BOOST_AUTO_TEST_CASE(ContainerIsa) {
  std::map<Target, int> Map;
  Map[ExampleTarget] = 1;

  auto Ptr = make_unique<MapContainer>(move(Map), "dont_care");
  ContainerBase *BasePtr = Ptr.get();
  BOOST_TEST(llvm::isa<MapContainer>(BasePtr));
  BOOST_TEST(llvm::cast<MapContainer>(BasePtr) != nullptr);
  BOOST_TEST(Ptr->get(ExampleTarget) == 1);
}

static ContainerFactory getMapFactoryContainer() {

  return
    [](llvm::StringRef Name) { return std::make_unique<MapContainer>(Name); };
}

BOOST_AUTO_TEST_CASE(ContainersCanBeCreated) {
  ContainerSet Containers;
  auto Factory = getMapFactoryContainer();
  Containers.add(CName, Factory, Factory("RandomName"));
  BOOST_TEST(Containers.contains(CName));
  auto &Container = llvm::cast<MapContainer>(Containers[CName]);
  Container.get(ExampleTarget) = 1;
  BOOST_TEST(cast<MapContainer>(Containers.at(CName)).get(ExampleTarget) == 1);
}

class TestPipe {

public:
  static constexpr auto Name = "TestPipe";

  std::vector<ContractGroup> getContract() const {
    return { ContractGroup(RootKind, KE::Exact, 0, RootKind2, 0) };
  }

  void run(const Context &, const MapContainer &Source, MapContainer &Target) {
    for (const auto &Element : Source.getMap())
      if (&Element.first.getKind() == &RootKind) {
        pipeline::Target NewTar(RootKind2);
        Target.get(NewTar) = Element.second;
      }
  }
};

BOOST_AUTO_TEST_CASE(PipeCanBeWrapper) {
  Context Ctx;
  MapContainer Map("RandomName");
  Map.get({ {}, RootKind }) = 1;
  TestPipe Enf;
  Enf.run(Ctx, Map, Map);
  BOOST_TEST(Map.get({ {}, RootKind2 }) == 1);
}

BOOST_AUTO_TEST_CASE(InputOutputContractExactPassForward) {
  ContainerToTargetsMap Targets;
  Targets[CName].emplace_back(Target({}, RootKind));

  ContractGroup Contract1(RootKind, KE::Exact);
  Contract1.deduceResults(Targets, { CName });
  BOOST_TEST((&Targets[CName][0].getKind() == &RootKind));
  BOOST_TEST((Targets[CName][0].kindExactness() == KE::Exact));
}

BOOST_AUTO_TEST_CASE(InputOutputContractExactExactForward) {
  ContainerToTargetsMap Targets;
  Targets[CName].emplace_back(Target({}, RootKind));
  ContractGroup Contract1(RootKind, KE::Exact, 0, RootKind2, 0);
  Contract1.deduceResults(Targets, { CName });
  BOOST_TEST((&Targets[CName][0].getKind() == &RootKind2));
  BOOST_TEST((Targets[CName][0].kindExactness() == KE::Exact));
}

BOOST_AUTO_TEST_CASE(InputOutputContractMultiLine) {
  ContainerToTargetsMap Targets;
  Targets["third"].emplace_back(Target({}, RootKind));

  Contract FirstPart(RootKind2,
                     Exactness::Exact,
                     0,
                     RootKind,
                     2,
                     InputPreservation::Preserve);
  Contract SecondPart(RootKind3,
                      Exactness::Exact,
                      1,
                      RootKind,
                      2,
                      InputPreservation::Preserve);
  ContractGroup Contract({ FirstPart, SecondPart });

  auto Res = Contract.deduceRequirements(Targets,
                                         { "first", "second", "third" });

  BOOST_TEST(Res["first"].size() == 1);
  BOOST_TEST(Res["second"].size() == 1);
}

BOOST_AUTO_TEST_CASE(InputOutputContractDerivedPassForward) {
  ContainerToTargetsMap Targets;
  Targets[CName].emplace_back(Target({}, RootKind2));

  ContractGroup Contract1(RootKind, KE::DerivedFrom);
  Contract1.deduceResults(Targets, { CName });
  BOOST_TEST((&Targets[CName][0].getKind() == &RootKind2));
  BOOST_TEST((Targets[CName][0].kindExactness() == KE::Exact));
}

BOOST_AUTO_TEST_CASE(InputOutputContractDerivedExactForward) {
  ContainerToTargetsMap Targets;
  Targets[CName].emplace_back(Target({}, RootKind2));

  ContractGroup Contract1(RootKind, KE::DerivedFrom, 0, RootKind, 0);
  Contract1.deduceResults(Targets, { CName });
  BOOST_TEST((&Targets[CName][0].getKind() == &RootKind));
  BOOST_TEST((Targets[CName][0].kindExactness() == KE::Exact));
}

BOOST_AUTO_TEST_CASE(InputOutputContractExactPassBackward) {
  ContainerToTargetsMap Targets;
  Targets[CName].emplace_back(Target({}, RootKind));

  ContractGroup Contract1(RootKind, KE::Exact);
  auto Res = Contract1.deduceRequirements(Targets, { CName });
  BOOST_TEST((&Res[CName][0].getKind() == &RootKind));
  BOOST_TEST((Res[CName][0].kindExactness() == KE::Exact));
}

BOOST_AUTO_TEST_CASE(InputOutputContractExactExactBackward) {
  ContainerToTargetsMap Targets;
  Targets[CName].emplace_back(Target({}, RootKind2));

  ContractGroup Contract1(RootKind, KE::Exact, 0, RootKind2, 0);
  auto Res = Contract1.deduceRequirements(Targets, { CName });
  BOOST_TEST((&Res[CName][0].getKind() == &RootKind));
  BOOST_TEST((Res[CName][0].kindExactness() == KE::Exact));
}

BOOST_AUTO_TEST_CASE(InputOutputContractDerivedPassBackward) {
  ContainerToTargetsMap Targets;
  Targets[CName].emplace_back(Target({}, RootKind2));

  ContractGroup Contract1(RootKind, KE::DerivedFrom);
  auto Res = Contract1.deduceRequirements(Targets, { CName });
  BOOST_TEST((&Res[CName][0].getKind() == &RootKind2));
  BOOST_TEST((Res[CName][0].kindExactness() == KE::Exact));
}

BOOST_AUTO_TEST_CASE(InputOutputContractDerivedExactBackward) {
  ContainerToTargetsMap Targets;
  Targets[CName].emplace_back(Target({}, RootKind2));

  ContractGroup Contract1(RootKind, KE::DerivedFrom, 0, RootKind2, 0);
  auto Res = Contract1.deduceRequirements(Targets, { CName });
  BOOST_TEST((&Res[CName][0].getKind() == &RootKind));
  BOOST_TEST((Res[CName][0].kindExactness() == KE::DerivedFrom));
}

BOOST_AUTO_TEST_CASE(InputOutputContractExactExactFineGrainedBackward) {
  ContainerToTargetsMap Targets;
  Targets.add(CName, { "f1" }, FunctionKind);

  ContractGroup Contract1(RootKind, KE::Exact, 0, FunctionKind, 0);
  auto Res = Contract1.deduceRequirements(Targets, { CName });
  BOOST_TEST((&Res[CName][0].getKind() == &RootKind));
  BOOST_TEST((Res[CName][0].kindExactness() == KE::Exact));
  BOOST_TEST((Res[CName][0].getPathComponents().size() == 0));
}

BOOST_AUTO_TEST_CASE(InputOutputContractExactExactFineGrainedForward) {
  ContainerToTargetsMap Targets;
  Targets[CName].emplace_back(Target({}, RootKind));

  ContractGroup Contract1(RootKind, KE::Exact, 0, FunctionKind, 0);
  Contract1.deduceResults(Targets, { CName });
  BOOST_TEST((&Targets[CName][0].getKind() == &FunctionKind));
  BOOST_TEST((Targets[CName][0].kindExactness() == KE::Exact));
  BOOST_TEST((Targets[CName][0].getPathComponents().size() == 1));
  BOOST_TEST((Targets[CName][0].getPathComponents()[0].isAll()));
}

static void checkIfContains(auto &TargetRange, const Kind &K, KE Exact) {
  const auto ToFind = [&K, Exact](const Target &Target) {
    return &Target.getKind() == &K and Target.kindExactness() == Exact;
  };
  BOOST_TEST(llvm::find_if(TargetRange, ToFind) != TargetRange.end());
}

BOOST_AUTO_TEST_CASE(InputOutputContractMupltipleInputTest) {
  ContainerToTargetsMap Targets;
  Targets[CName].emplace_back(Target(RootKind2));
  Targets[CName].emplace_back(Target(RootKind));

  ContractGroup Contract1(RootKind, KE::DerivedFrom, 0, RootKind2, 0);
  auto Res = Contract1.deduceRequirements(Targets, { CName });

  const auto &ProducedResults = Res[CName];
  checkIfContains(ProducedResults, RootKind, KE::Exact);
}

BOOST_AUTO_TEST_CASE(InputOutputContractPreserved) {
  ContainerToTargetsMap Targets;
  Targets[CName].emplace_back(Target({}, RootKind2));
  Targets[CName].emplace_back(Target({}, RootKind));

  ContractGroup Contract1(RootKind,
                          KE::DerivedFrom,
                          0,
                          RootKind2,
                          0,
                          InputPreservation::Preserve);
  Contract1.deduceResults(Targets, { CName });
  const auto &ProducedResults = Targets[CName];
  checkIfContains(ProducedResults, RootKind2, KE::Exact);
  checkIfContains(ProducedResults, RootKind, KE::Exact);
  checkIfContains(ProducedResults, RootKind2, KE::Exact);
}

BOOST_AUTO_TEST_CASE(InputOutputContractPreservedBackwardMain) {
  ContainerToTargetsMap Targets;
  Targets[CName].emplace_back(Target({}, RootKind2));

  ContractGroup Contract1(RootKind,
                          KE::DerivedFrom,
                          0,
                          RootKind2,
                          0,
                          InputPreservation::Preserve);
  auto Res = Contract1.deduceRequirements(Targets, { CName });
  BOOST_TEST((&Res[CName][0].getKind() == &RootKind));
  BOOST_TEST((Res[CName][0].kindExactness() == KE::DerivedFrom));
}

BOOST_AUTO_TEST_CASE(InputOutputContractPreservedBackwardSecondary) {
  ContainerToTargetsMap Targets;
  Targets[CName].emplace_back(Target({}, RootKind));

  ContractGroup Contract1(RootKind,
                          KE::DerivedFrom,
                          0,
                          RootKind2,
                          0,
                          InputPreservation::Preserve);
  auto Res = Contract1.deduceRequirements(Targets, { CName });
  BOOST_TEST((&Res[CName][0].getKind() == &RootKind));
  BOOST_TEST((Res[CName][0].kindExactness() == KE::Exact));
}

BOOST_AUTO_TEST_CASE(StepCanCloneAndRun) {
  Context Ctx;
  Runner Pip(Ctx);

  ContainerSet Containers;
  auto Factory = getMapFactoryContainer();
  Containers.add(CName, Factory, Factory("dont_care"));
  Step Step("first_step",
            move(Containers),
            PipeWrapper::bind<TestPipe>(CName, CName));

  ContainerToTargetsMap Targets;
  Targets[CName].emplace_back(RootKind2);

  Containers = ContainerSet();
  auto Factory2 = getMapFactoryContainer();
  Containers.add(CName, Factory, Factory("dont_care"));
  cast<MapContainer>(Containers[CName]).get(Target({}, RootKind)) = 1;
  auto Result = Step.cloneAndRun(Ctx, std::move(Containers));

  auto &Cont = cast<MapContainer>(Result.at(CName));
  BOOST_TEST(Cont.get(Target({}, RootKind2)) == 1);
}

BOOST_AUTO_TEST_CASE(PipelineCanBeManuallyExectued) {
  ContainerFactorySet Registry;
  Registry.registerDefaultConstructibleFactory<MapContainer>(CName);

  Context Ctx;
  Runner Pip(Ctx);
  Pip.addStep(Step("first_step",
                   Registry.createEmpty(),
                   PipeWrapper::bind<TestPipe>(CName, CName)));

  auto Containers = Registry.createEmpty();
  auto &C1 = Containers.getOrCreate<MapContainer>(CName);
  C1.get(Target(RootKind)) = 1;

  auto Res = Pip["first_step"].cloneAndRun(Ctx, std::move(Containers));
  BOOST_TEST(cast<MapContainer>(Res.at(CName)).get(Target(RootKind2)) == 1);
  const auto &StartingContainer = Pip["first_step"]
                                    .containers()
                                    .getOrCreate<MapContainer>(CName);
  auto Val = StartingContainer.get(Target(RootKind2));
  BOOST_TEST(Val == 1);
}

BOOST_AUTO_TEST_CASE(SingleElementPipelineCanBeRunned) {
  Context Ctx;
  Runner Pip(Ctx);

  ContainerSet Content;
  auto Factory = getMapFactoryContainer();
  Content.add(CName, Factory, Factory("dont_care"));
  auto &C1 = cast<MapContainer>(Content[CName]);
  C1.get(Target(RootKind)) = 1;

  Step StepToAdd("first_step", move(Content));
  Pip.addStep(std::move(StepToAdd));
  ContainerSet &BCI = Pip["first_step"].containers();
  BOOST_TEST(cast<MapContainer>(BCI.at(CName)).get(Target(RootKind)) == 1);

  ContainerSet Containers2;
  Containers2.add(CName, Factory, make_unique<MapContainer>("dont_care"));
  Pip.addStep(Step("End",
                   move(Containers2),
                   Pip["first_step"],
                   PipeWrapper::bind<TestPipe>(CName, CName)));

  ContainerToTargetsMap Targets;
  Targets[CName].emplace_back(Target(RootKind2));
  auto Error = Pip.run("End", Targets);
  BOOST_TEST(!Error);
  ContainerSet &BC = Pip["End"].containers();
  BOOST_TEST(cast<MapContainer>(BC.at(CName)).get(Target(RootKind2)) == 1);
}

class FineGranerPipe {

public:
  static constexpr auto Name = "FinedGranedPipe";
  std::vector<ContractGroup> getContract() const {
    return { ContractGroup(RootKind, KE::Exact, 0, FunctionKind, 1) };
  }

  void run(Context &, const MapContainer &Source, MapContainer &Target) {
    for (const auto &Element : Source.getMap()) {

      if (&Element.first.getKind() != &RootKind)
        continue;

      auto PathComponents = Element.first.getPathComponents();
      PathComponents.emplace_back("f1");
      Target.get({ move(PathComponents), FunctionKind }) = Element.second;

      PathComponents = Element.first.getPathComponents();
      PathComponents.emplace_back("f2");
      Target.get({ move(PathComponents), FunctionKind }) = Element.second;
    }
  }
};

class CopyPipe {

public:
  static constexpr auto Name = "CopyPipe";
  std::vector<ContractGroup> getContract() const {
    return { ContractGroup(FunctionKind,
                           KE::Exact,
                           0,
                           FunctionKind,
                           1,
                           InputPreservation::Preserve) };
  }

  void run(Context &, const MapContainer &Source, MapContainer &Target) {
    for (const auto &Element : Source.getMap())
      Target.get(Element.first) = Element.second;
  }
};

BOOST_AUTO_TEST_CASE(SingleElementPipelineBackwardFinedGrained) {
  Context Ctx;
  Runner Pipeline(Ctx);
  Pipeline.addDefaultConstructibleFactory<MapContainer>(CName);

  const std::string Name = "first_step";
  Pipeline.emplaceStep("", Name);
  Pipeline.emplaceStep(Name,
                       "End",
                       PipeWrapper::bind<FineGranerPipe>(CName, CName));

  auto &Container(Pipeline[Name].containers().getOrCreate<MapContainer>(CName));
  Container.get(Target(RootKind)) = 1;

  ContainerToTargetsMap Targets;
  Targets.add(CName, { "f1" }, FunctionKind);

  auto Error = Pipeline.run("End", Targets);
  BOOST_TEST(!Error);
  auto &FinalContainer = Pipeline["End"].containers().get<MapContainer>(CName);
  Target FinalTarget({ "f1" }, FunctionKind);
  auto Val = FinalContainer.get(FinalTarget);

  BOOST_TEST(Val == 1);
}

BOOST_AUTO_TEST_CASE(DifferentNamesAreNotCompatible) {
  Target Target1({ "f1Wrong" }, FunctionKind);
  Target Target2({ "f1" }, FunctionKind);

  BOOST_TEST(not Target1.satisfies(Target2));
  BOOST_TEST(not Target2.satisfies(Target1));
}

BOOST_AUTO_TEST_CASE(DifferentNamesRootOnlyAreNotCompatibleSet) {
  Target Target1({ "f1Wrong" }, FunctionKind);
  Target Target2({ "f1" }, FunctionKind);
  TargetsList TargetList;
  TargetList.push_back(Target1);

  BOOST_TEST(not TargetList.contains(Target2));
  BOOST_TEST(TargetList.contains(Target1));
}

BOOST_AUTO_TEST_CASE(SingleElementPipelineFailure) {
  Context Ctx;
  Runner Pipeline(Ctx);
  Pipeline.addDefaultConstructibleFactory<MapContainer>(CName);

  const std::string Name = "first_step";
  Pipeline.emplaceStep("",
                       Name,
                       PipeWrapper::bind<FineGranerPipe>(CName, CName));
  Pipeline.emplaceStep(Name, "End");

  auto &Container(Pipeline[Name].containers().getOrCreate<MapContainer>(CName));
  Container.get(Target(RootKind)) = 1;

  ContainerToTargetsMap Targets;
  Targets.add(CName, {}, RootKind2);

  auto Error = Pipeline.run("End", Targets);
  BOOST_TEST(!!Error);
  consumeError(std::move(Error));
}

static void makeF(llvm::Module &M, llvm::StringRef FName) {

  auto VoidType = llvm::Type::getVoidTy(M.getContext());
  auto *FType = llvm::FunctionType::get(VoidType, {});
  auto F = M.getOrInsertFunction(FName, FType);
  auto *Fun = llvm::dyn_cast<llvm::Function>(F.getCallee());
  auto *BB = llvm::BasicBlock::Create(M.getContext(), "bb", Fun);
  llvm::IRBuilder<> Builder(BB);
  Builder.SetInsertPoint(BB);
  Builder.CreateRet(nullptr);
}

struct FunctionInserterPass : public llvm::ModulePass {
  static char ID;
  FunctionInserterPass() : llvm::ModulePass(ID) {}

  bool runOnModule(llvm::Module &M) override {
    makeF(M, "f1");
    return true;
  }
};
char FunctionInserterPass::ID = '_';

struct IdentityPass : public llvm::ModulePass {
  static char ID;

  IdentityPass() : llvm::ModulePass(ID) {}

  bool runOnModule(llvm::Module &M) override { return true; }
};

char IdentityPass::ID = '_';

static llvm::RegisterPass<IdentityPass> X2("IdentityPass", "IdentityPass");

struct LLVMPassFunctionCreator {
  static constexpr auto Name = "Function Creator";

  std::vector<ContractGroup> getContract() const {
    return { ContractGroup(RootKind, KE::Exact, 0, FunctionKind) };
  }

  void registerPasses(llvm::legacy::PassManager &Manager) {
    Manager.add(new FunctionInserterPass());
  }
};

struct LLVMPassFunctionIdentity {
  static constexpr auto Name = "Identity";

  std::vector<ContractGroup> getContract() const {
    return { ContractGroup(FunctionKind, KE::Exact) };
  }

  void registerPasses(llvm::legacy::PassManager &Manager) {
    Manager.add(new IdentityPass());
  }
};

BOOST_AUTO_TEST_CASE(SingleElementLLVMPipelineBackwardFinedGrained) {
  llvm::LLVMContext C;

  Context Ctx;
  Runner Pipeline(Ctx);
  Pipeline.addContainerFactory(CName, makeDefaultLLVMContainerFactory(Ctx, C));

  const std::string Name = "first_step";
  Pipeline.emplaceStep("", Name);
  Pipeline
    .emplaceStep(Name,
                 "End",
                 LLVMContainer::wrapLLVMPasses(CName,
                                               LLVMPassFunctionCreator(),
                                               LLVMPassFunctionIdentity()));

  auto &C1(Pipeline[Name].containers().getOrCreate<LLVMContainer>(CName));
  makeF(C1.getModule(), "root");

  ContainerToTargetsMap Targets;
  Targets.add(CName, Target({ PathComponent("f1") }, FunctionKind));

  auto Error = Pipeline.run("End", Targets);
  BOOST_TEST(!Error);

  const auto &Final = Pipeline["End"].containers().get<LLVMContainer>(CName);
  const auto *F = Final.getModule().getFunction("f1");

  BOOST_TEST(F != nullptr);
}

BOOST_AUTO_TEST_CASE(LLVMPurePipe) {
  llvm::LLVMContext C;

  Context Ctx;
  Runner Pipeline(Ctx);
  Pipeline.addContainerFactory(CName, makeDefaultLLVMContainerFactory(Ctx, C));

  const std::string Name = "first_step";
  PureLLVMPassWrapper IdentityPass("IdentityPass");
  Pipeline.emplaceStep("", Name);
  Pipeline.emplaceStep(Name,
                       "End",
                       LLVMContainer::wrapLLVMPasses(CName,
                                                     LLVMPassFunctionCreator(),
                                                     IdentityPass));

  auto &C1 = Pipeline[Name].containers().getOrCreate<LLVMContainer>(CName);
  makeF(C1.getModule(), "root");

  ContainerToTargetsMap Targets;
  Targets.add(CName, Target({ PathComponent("f1") }, FunctionKind));

  auto Error = Pipeline.run("End", Targets);
  BOOST_TEST(!Error);

  const auto &Final = Pipeline["End"].containers().get<LLVMContainer>(CName);
  const auto *F = Final.getModule().getFunction("f1");

  BOOST_TEST(F != nullptr);
}

BOOST_AUTO_TEST_CASE(SingleElementPipelineForwardFinedGrained) {
  Context Ctx;
  Runner Pipeline(Ctx);
  Pipeline.addDefaultConstructibleFactory<MapContainer>(CName);

  const std::string Name = "first_step";
  Pipeline.emplaceStep("", Name);
  Pipeline.emplaceStep(Name,
                       "End",
                       PipeWrapper::bind<FineGranerPipe>(CName, CName));

  auto &C1 = Pipeline[Name].containers().getOrCreate<MapContainer>(CName);
  C1.get(Target({}, RootKind)) = 1;
  auto &C2 = Pipeline["End"].containers().getOrCreate<MapContainer>(CName);

  const auto T = Target({ PathComponent::all() }, FunctionKind);
  C2.get(T) = 1;

  llvm::StringMap<ContainerToTargetsMap> Invalidations;
  Invalidations[Name].add(CName, {}, RootKind);

  auto Error = Pipeline.getInvalidations(Invalidations);
  BOOST_TEST(!Error);

  const auto &EndContainerInvalidations = Invalidations["End"][CName];
  BOOST_TEST(not EndContainerInvalidations.empty());
  const auto &QuantifOfInvalidated = EndContainerInvalidations.front()
                                       .getPathComponents();
  BOOST_TEST((QuantifOfInvalidated.back().isAll()));
}

BOOST_AUTO_TEST_CASE(SingleElementPipelineInvalidation) {
  Context Ctx;
  Runner Pipeline(Ctx);
  Pipeline.addDefaultConstructibleFactory<MapContainer>(CName);

  const std::string Name = "first_step";
  Pipeline.emplaceStep("", Name);
  Pipeline.emplaceStep(Name,
                       "End",
                       PipeWrapper::bind<FineGranerPipe>(CName, CName));

  auto &C1 = Pipeline[Name].containers().getOrCreate<MapContainer>(CName);
  C1.get(Target({}, RootKind)) = 1;
  auto &C2 = Pipeline["End"].containers().getOrCreate<MapContainer>(CName);

  const auto T = Target({ PathComponent::all() }, FunctionKind);
  C2.get(T) = 1;

  Target ToKill({}, RootKind);

  llvm::StringMap<ContainerToTargetsMap> Invalidations;
  auto Error = Pipeline.getInvalidations(ToKill, Invalidations);
  BOOST_TEST(!Error);
  const auto &QuantifOfInvalidated = Invalidations["End"][CName]
                                       .front()
                                       .getPathComponents();
  BOOST_TEST((QuantifOfInvalidated.front().isAll()));
}

BOOST_AUTO_TEST_CASE(SingleElementPipelineWithRemove) {
  Context Ctx;
  Runner Pipeline(Ctx);
  Pipeline.addDefaultConstructibleFactory<MapContainer>(CName);

  const std::string Name = "first_step";
  Pipeline.emplaceStep("",
                       Name,
                       PipeWrapper::bind<FineGranerPipe>(CName, CName));
  Pipeline.emplaceStep(Name, "End");

  auto &C1 = Pipeline[Name].containers().getOrCreate<MapContainer>(CName);
  C1.get(Target(RootKind)) = 1;

  Target ToKill(RootKind);
  auto Error = Pipeline.invalidate(ToKill);
  BOOST_TEST(!Error);

  auto &C2 = Pipeline[Name].containers().getOrCreate<MapContainer>(CName);
  auto IsIn = C2.contains(ToKill);
  BOOST_TEST(IsIn == false);
}

BOOST_AUTO_TEST_CASE(LoaderTest) {
  StepDeclaration SDeclaration{ "FirstStep",
                                { { "FineGranerPipe", { CName, CName } } } };
  PipelineDeclaration PDeclaration{ "",
                                    { { CName, "MapContainer" } },
                                    { move(SDeclaration) } };

  auto Ctx = Context::fromRegistry(Registry::registerAllKinds());
  Loader Loader(Ctx);
  Loader.addDefaultConstructibleContainer<MapContainer>("MapContainer");
  Loader.registerPipe<FineGranerPipe>("FineGranerPipe");

  auto MaybePipeline = Loader.load(PDeclaration);
  BOOST_TEST(!!MaybePipeline);
  auto &Pipeline = *MaybePipeline;
  const std::string Name = "FirstStep";
  BOOST_TEST((Pipeline[Name].getName() == Name));
  BOOST_TEST((Pipeline["begin"].getName() == "begin"));

  ContainerToTargetsMap Targets;
  Targets.add(CName, { "f1" }, FunctionKind);
  auto &C1 = Pipeline["begin"].containers().getOrCreate<MapContainer>(CName);
  C1.get(Target(RootKind)) = 1;

  auto Error = Pipeline.run(Name, Targets);
  BOOST_TEST(!Error);
  auto &FinalContainer = Pipeline[Name].containers().get<MapContainer>(CName);
  Target FinalTarget({ "f1" }, FunctionKind);
  auto Val = FinalContainer.get(FinalTarget);

  BOOST_TEST(Val == 1);
}

static const std::string Pipeline(R"(---
                       Containers:
                         - Name:            ContainerName
                           Type:            MapContainer
                       Steps:
                         - Name:            FirstStep
                           Pipes:
                             - Type:            FineGranerPipe
                               UsedContainers:
                                 - ContainerName
                                 - ContainerName
                       )");

BOOST_AUTO_TEST_CASE(LoaderTestFromYaml) {
  Context Ctx;
  Loader Loader(Ctx);
  Loader.addDefaultConstructibleContainer<MapContainer>("MapContainer");
  Loader.registerPipe<FineGranerPipe>("FineGranerPipe");
  auto MaybePipeline = Loader.load(Pipeline);
  BOOST_TEST(!!MaybePipeline);
}

static const std::string PipelineTree(R"(---
                       Containers:
                         - Name:            ContainerName
                           Type:            MapContainer
                       Steps:
                         - Name:            FirstStep
                           Pipes:
                             - Type:            FineGranerPipe
                               UsedContainers:
                                 - ContainerName
                                 - ContainerName
                       )");

static const std::string PipelineTree2(R"(---
                       From:      FirstStep
                       Containers:
                       Steps:
                         - Name:            SecondStep
                       )");

static const std::string PipelineTree3(R"(---
                       From:      FirstStep
                       Containers:
                       Steps:
                         - Name:            ThirdStep
                       )");

BOOST_AUTO_TEST_CASE(LoaderTestFromYamlTree) {
  Context Ctx;
  Loader Loader(Ctx);
  Loader.addDefaultConstructibleContainer<MapContainer>("MapContainer");
  Loader.registerPipe<FineGranerPipe>("FineGranerPipe");
  vector ToLoad({ PipelineTree2, PipelineTree3, PipelineTree });
  auto MaybePipeline = Loader.load(ToLoad);
  BOOST_TEST(!!MaybePipeline);
}

BOOST_AUTO_TEST_CASE(LoaderTestFromYamlLLVM) {
  llvm::LLVMContext C;
  Context Ctx;
  Loader Loader(Ctx);
  Loader.addContainerFactory("LLVMContainer",
                             makeDefaultLLVMContainerFactory(Ctx, C));
  auto *Name = "CreateFunctionPass";
  Loader.registerLLVMPass<LLVMPassFunctionCreator>(Name);
  Name = "IdentityPass";
  Loader.registerLLVMPass<LLVMPassFunctionIdentity>(Name);

  std::string LLVMPipeline(R"(---
                       Containers:
                         - Name:            CustomName
                           Type:            LLVMContainer
                       Steps:
                         - Name:            FirstStep
                           Pipes:
                             - Type:             LLVMPipe
                               UsedContainers:
                                 - CustomName
                               Passes:
                                 - CreateFunctionPass
                                 - IdentityPass
                       )");

  auto MaybePipeline = Loader.load(LLVMPipeline);

  BOOST_TEST(!!MaybePipeline);
}

static std::string getCurrentPath() {
  llvm::SmallVector<char, 3> ToReturn;
  llvm::sys::fs::current_path(ToReturn);
  return std::string(ToReturn.begin(), ToReturn.end());
}

BOOST_AUTO_TEST_CASE(SingleElementPipelineStoreToDisk) {
  Context Ctx;
  Runner Pipeline(Ctx);
  Pipeline.addDefaultConstructibleFactory<MapContainer>(CName);

  const std::string Name = "first_step";
  Pipeline.emplaceStep("",
                       Name,
                       PipeWrapper::bind<FineGranerPipe>(CName, CName));
  Pipeline.emplaceStep(Name, "End");

  auto &C1 = Pipeline[Name].containers().getOrCreate<MapContainer>(CName);
  C1.get(Target({}, RootKind)) = 1;

  BOOST_TEST((!Pipeline.storeToDisk(getCurrentPath())));

  auto &Container(Pipeline[Name].containers().getOrCreate<MapContainer>(CName));

  BOOST_TEST((Container.get(Target({}, RootKind)) == 1));
  Container.get(Target({}, RootKind)) = 2;
  BOOST_TEST((Container.get(Target({}, RootKind)) == 2));
  BOOST_TEST((!Pipeline.loadFromDisk(getCurrentPath())));
  BOOST_TEST(Pipeline[Name].containers().contains(CName));
}

BOOST_AUTO_TEST_CASE(SingleElementPipelineStoreToDiskWithOverrides) {
  Context Ctx;
  Loader Loader(Ctx);
  Loader.addDefaultConstructibleContainer<MapContainer>("MapContainer");
  Loader.registerPipe<FineGranerPipe>("FineGranerPipe");
  auto MaybePipeline = Loader.load(Pipeline);
  BOOST_TEST(!!MaybePipeline);
  auto &Pipeline = *MaybePipeline;
  auto *Mapping = "FirstStep:ContainerName:DontCareSourceFile";
  auto MaybeMapping = PipelineFileMapping::parse(Mapping);
  BOOST_TEST(!!MaybeMapping);

  const std::string Name = "FirstStep";
  auto &Container(Pipeline[Name].containers().getOrCreate<MapContainer>(CName));

  Container.get(Target({}, RootKind)) = 1;
  BOOST_TEST((!MaybeMapping->storeToDisk(Pipeline)));
  Container.get(Target({}, RootKind)) = 2;
  BOOST_TEST((Container.get(Target({}, RootKind)) == 2));
  BOOST_TEST((!MaybeMapping->loadFromDisk(Pipeline)));
  BOOST_TEST((Container.get(Target({}, RootKind)) == 1));
}

class EnumerableContainerExample
  : public EnumerableContainer<EnumerableContainerExample> {
public:
  static char ID;

  EnumerableContainerExample(Context &Ctx,
                             llvm::StringRef Name,
                             llvm::StringRef MIMEType) :
    EnumerableContainer<EnumerableContainerExample>(Ctx, Name, MIMEType) {}

  unique_ptr<ContainerBase>
  cloneFiltered(const TargetsList &Container) const final {
    return make_unique<EnumerableContainerExample>(*this);
  }

  ~EnumerableContainerExample() override = default;

  llvm::Error storeToDisk(llvm::StringRef Path) const override {

    return llvm::Error::success();
  }

  llvm::Error loadFromDisk(llvm::StringRef Path) override {

    return llvm::Error::success();
  }

  llvm::Error serialize(llvm::raw_ostream &OS) const final {

    return llvm::Error::success();
  }

  llvm::Error deserialize(const llvm::MemoryBuffer &Buffer) final {

    return llvm::Error::success();
  }

  llvm::Error
  extractOne(llvm::raw_ostream &OS, const Target &Target) const override {
    revng_abort();
    return llvm::Error::success();
  }

  /// Must reset the state of the container to the just built state
  void clear() final {}

  std::set<Target> Targets;

private:
  void mergeBackImpl(EnumerableContainerExample &&Container) override {}
};

char EnumerableContainerExample::ID;

class ExampleContainerInpsector
  : public ContainerEnumerator<EnumerableContainerExample> {
public:
  ExampleContainerInpsector() :
    ContainerEnumerator<EnumerableContainerExample>(RootKind) {}
  TargetsList
  enumerate(const Context &Ctx,
            const EnumerableContainerExample &Container) const final {
    TargetsList ToReturn;

    llvm::copy(Container.Targets, back_inserter(ToReturn));
    return ToReturn;
  }

  bool contains(const Target &Target,
                const EnumerableContainerExample &Container) const {
    return Container.Targets.count(Target);
  }

  bool remove(const Context &Ctx,
              const TargetsList &Targets,
              EnumerableContainerExample &Container) const {

    bool ErasedAll = true;
    for (const Target &Target : Targets)
      ErasedAll = remove(Target, Container) and ErasedAll;

    return ErasedAll;
  }

  bool
  remove(const Target &Target, EnumerableContainerExample &Container) const {

    if (not contains(Target, Container))
      return false;

    Container.Targets.erase(Target);
    return true;
  }
};

static ExampleContainerInpsector Example;

BOOST_AUTO_TEST_CASE(EnumerableContainersTest) {
  Context Ctx;
  EnumerableContainerExample Example(Ctx,
                                     "dont_care",
                                     "application/x.dont.care");
  Target T({}, RootKind, Exactness::DerivedFrom);
  Example.Targets.insert(T);
  BOOST_TEST(Example.contains(T));
  BOOST_TEST(Example.remove(TargetsList({ T })));
  BOOST_TEST(not Example.contains(T));
}

class LLVMInspectorExample
  : public LLVMGlobalKindBase<ExampleLLVMInspectalbeContainer> {
public:
  using LLVMGlobalKindBase<ExampleLLVMInspectalbeContainer>::LLVMGlobalKindBase;
  std::optional<Target>
  symbolToTarget(const llvm::Function &Symbol) const override {
    return Target({ Symbol.getName() }, FunctionKind);
  }

  TargetsList compactTargets(const Context &Ctx,
                             TargetsList::List &Targets) const override {
    Target F1({ "f1" }, FunctionKind);
    Target F2({ "f2" }, FunctionKind);
    Target All({ PathComponent::all() }, FunctionKind);
    if (find(Targets, F1) != Targets.end()
        and find(Targets, F2) != Targets.end())

      return TargetsList({ All });
    return TargetsList(move(Targets));
  }
};

class LLVMRootInspectorExample
  : public LLVMGlobalKindBase<ExampleLLVMInspectalbeContainer> {
public:
  using LLVMGlobalKindBase<ExampleLLVMInspectalbeContainer>::LLVMGlobalKindBase;
  std::optional<Target>
  symbolToTarget(const llvm::Function &Symbol) const override {
    return Target({}, RootKind);
  }

  TargetsList compactTargets(const Context &Ctx,
                             TargetsList::List &Targets) const override {
    return TargetsList(move(Targets));
  }
};

static LLVMInspectorExample ExampleLLVMInspector("dc", &FunctionRank);
static LLVMRootInspectorExample ExampleLLVMRootInspector("dc2", &Root);

BOOST_AUTO_TEST_CASE(LLVMKindTest) {
  llvm::LLVMContext C;
  using Cont = ExampleLLVMInspectalbeContainer;

  Context Ctx;
  Runner Pipeline(Ctx);
  Pipeline.addContainerFactory(CName, makeLLVMContainerFactory<Cont>(Ctx, C));

  Pipeline.emplaceStep("", "first_step");
  Pipeline.emplaceStep("first_step",
                       "End",
                       Cont::wrapLLVMPasses(CName, LLVMPassFunctionCreator()));

  makeF(Pipeline["first_step"]
          .containers()
          .getOrCreate<Cont>(CName)
          .getModule(),
        "root");

  ContainerToTargetsMap Targets;
  Targets.add(CName, Target({ "f1" }, FunctionKind));

  auto Error = Pipeline.run("End", Targets);
  BOOST_TEST(!Error);

  const auto &Final = Pipeline["End"].containers().get<Cont>(CName);
  const auto *F = Final.getModule().getFunction("f1");

  BOOST_TEST(F != nullptr);
}

class InspectorKindExample
  : public LLVMGlobalKindBase<ExampleLLVMInspectalbeContainer> {
public:
  InspectorKindExample() :
    LLVMGlobalKindBase<ExampleLLVMInspectalbeContainer>("ExampleName",
                                                        &FunctionRank) {}

  std::optional<Target>
  symbolToTarget(const llvm::Function &Symbol) const final {
    return Target({ Symbol.getName() }, *this);
  }

  TargetsList
  compactTargets(const Context &Ctx, TargetsList::List &Targets) const final {
    Target F1({ "f1" }, *this);
    Target F2({ "f2" }, *this);
    Target All({ PathComponent::all() }, *this);
    if (find(Targets, F1) != Targets.end()
        and find(Targets, F2) != Targets.end())

      return TargetsList({ All });
    return TargetsList(move(Targets));
  }

  ~InspectorKindExample() override {}
};

static InspectorKindExample InspKindExample;

BOOST_AUTO_TEST_CASE(InspectorKindTest) {
  Context Ctx;
  llvm::LLVMContext C;

  using Cont = ExampleLLVMInspectalbeContainer;
  auto Factory = makeLLVMContainerFactory<Cont>(Ctx, C);

  auto Container = Factory("dont_care");

  makeF(cast<Cont>(*Container).getModule(), "root");

  Target RootF({ "root" }, InspKindExample);
  BOOST_TEST(Container->enumerate().contains(RootF));
}

BOOST_AUTO_TEST_CASE(MultiStepInvalidationTest) {
  Context Ctx;
  Runner Pipeline(Ctx);
  auto CName2 = CName + "2";
  Pipeline.addDefaultConstructibleFactory<MapContainer>(CName);
  Pipeline.addDefaultConstructibleFactory<MapContainer>(CName2);

  const std::string Name = "first_step";
  const std::string SecondName = "second_step";
  Pipeline.emplaceStep("", Name);
  Pipeline.emplaceStep(Name,
                       SecondName,
                       PipeWrapper::bind<FineGranerPipe>(CName, CName));
  Pipeline.emplaceStep(SecondName,
                       "End",
                       PipeWrapper::bind<CopyPipe>(CName, CName2));

  auto &C1 = Pipeline[Name].containers().getOrCreate<MapContainer>(CName);
  auto &C1End = Pipeline["End"].containers().getOrCreate<MapContainer>(CName);
  auto &C2End = Pipeline["End"].containers().getOrCreate<MapContainer>(CName2);

  const auto T = Target({}, RootKind);
  C1.get(T) = 1;

  const auto ToProduce = Target({ PathComponent("f1") }, FunctionKind);
  ContainerToTargetsMap Map;
  Map[CName2].emplace_back(ToProduce);
  cantFail(Pipeline.run("End", Map));

  BOOST_TEST(C1.get(T) == 1);
  BOOST_TEST(C1End.get(T) == 1);

  BOOST_TEST(C2End.get(ToProduce) == 1);

  pipeline::InvalidationMap Invalidations;
  Invalidations[Name][CName].push_back(T);

  auto Error = Pipeline.getInvalidations(Invalidations);
  BOOST_TEST(!Error);

  Error = Pipeline.invalidate(Invalidations);

  BOOST_TEST(!Error);

  BOOST_TEST(C1.get(T) == 0);
  BOOST_TEST(C1End.get(T) == 0);

  BOOST_TEST(C2End.get(ToProduce) == 0);
}

BOOST_AUTO_TEST_SUITE_END()
