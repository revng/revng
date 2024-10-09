/// \file Pipeline.cpp

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
#include "revng/Pipeline/ContainerFactory.h"
#include "revng/Pipeline/ContainerFactorySet.h"
#include "revng/Pipeline/ContainerSet.h"
#include "revng/Pipeline/Context.h"
#include "revng/Pipeline/Contract.h"
#include "revng/Pipeline/Errors.h"
#include "revng/Pipeline/ExecutionContext.h"
#include "revng/Pipeline/GenericLLVMPipe.h"
#include "revng/Pipeline/Invokable.h"
#include "revng/Pipeline/Kind.h"
#include "revng/Pipeline/LLVMContainer.h"
#include "revng/Pipeline/LLVMContainerFactory.h"
#include "revng/Pipeline/LLVMKind.h"
#include "revng/Pipeline/Loader.h"
#include "revng/Pipeline/Runner.h"
#include "revng/Pipeline/Target.h"
#include "revng/Support/Assert.h"

#define BOOST_TEST_MODULE Pipeline
bool init_unit_test();
#include "boost/test/unit_test.hpp"

#include "revng/UnitTestHelpers/UnitTestHelpers.h"

using namespace pipeline;
using namespace std;
using namespace llvm;

static auto Root = defineRootRank<"root">();
static auto FunctionRank = defineRank<"function", std::string>(Root);
class RootKindType : public LLVMKind {
public:
  RootKindType() : LLVMKind("root-kind", Root) { revng_assert(depth() == 0); };

  std::optional<Target>
  symbolToTarget(const llvm::Function &Symbol) const override {
    if (Symbol.getName() == "root")
      return Target(*this);
    return std::nullopt;
  }

  void appendAllTargets(const pipeline::Context &Context,
                        pipeline::TargetsList &Out) const override {
    Out.push_back(Target(*this));
  }

  ~RootKindType() override {}
};

static RootKindType RootKind;
static SingleElementKind RootKind2("root-kind-2", RootKind, Root, {}, {});
static SingleElementKind RootKind3("root-kind-3", Root, {}, {});

class SingleFunctionKind : public LLVMKind {
public:
  using LLVMKind::LLVMKind;

  std::optional<Target>
  symbolToTarget(const llvm::Function &Symbol) const override {
    if (Symbol.getName() == "f1")
      return Target("f1", *this);
    if (Symbol.getName() == "f2")
      return Target("f2", *this);
    return std::nullopt;
  }

  void appendAllTargets(const pipeline::Context &Context,
                        pipeline::TargetsList &Out) const override {
    Out.push_back(Target("f1", *this));
    Out.push_back(Target("f2", *this));
  }

  ~SingleFunctionKind() override {}
};
static SingleFunctionKind FunctionKind("function-kind", FunctionRank);

static std::string CName = "container-name";

class MapContainer : public Container<MapContainer> {
public:
  static inline const llvm::StringRef MIMEType = "application/x.test.map";
  static inline const char *Name = "Name";
  MapContainer(std::map<Target, int> Map, llvm::StringRef Name) :
    Container<MapContainer>(Name), Map(std::move(Map)) {}
  MapContainer(llvm::StringRef Name) : Container<MapContainer>(Name), Map() {}

  ~MapContainer() override = default;

  unique_ptr<ContainerBase>
  cloneFiltered(const TargetsList &Container) const final {
    return make_unique<MapContainer>(Map, this->name());
  }
  static std::vector<pipeline::Kind *> possibleKinds() {
    return { &FunctionKind, &RootKind };
  }

  bool contains(const Target &T) const { return Map.contains(T); }

  TargetsList enumerate() const final {
    TargetsList ToReturn;

    for (const auto &Target : Map)
      ToReturn.push_back(Target.first);

    return ToReturn;
  }

  bool remove(const TargetsList &Targets) override {
    bool RemovedAll = true;
    for (const auto &Target : Targets)
      RemovedAll = remove(Target) && RemovedAll;

    return RemovedAll;
  }

  bool remove(const Target &Target) {
    if (!Map.contains(Target))
      return false;

    Map.erase(Target);
    return true;
  }

  llvm::Error extractOne(llvm::raw_ostream &OS,
                         const Target &Target) const override {
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

  llvm::Error store(const revng::FilePath &Path) const override {
    SavedData = Map;
    return llvm::Error::success();
  }

  llvm::Error load(const revng::FilePath &Path) override {
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

  auto Ptr = make_unique<MapContainer>(std::move(Map), "dont-care");
  ContainerBase *BasePtr = Ptr.get();
  BOOST_TEST(llvm::isa<MapContainer>(BasePtr));
  BOOST_TEST(llvm::cast<MapContainer>(BasePtr) != nullptr);
  BOOST_TEST(Ptr->get(ExampleTarget) == 1);
}

static ContainerFactory getMapFactoryContainer() {
  return ContainerFactory::create<MapContainer>();
}

BOOST_AUTO_TEST_CASE(ContainersCanBeCreated) {
  ContainerSet Containers;
  auto Factory = getMapFactoryContainer();
  Containers.add(CName, Factory, Factory("random-name"));
  BOOST_TEST(Containers.contains(CName));
  auto &Container = llvm::cast<MapContainer>(Containers[CName]);
  Container.get(ExampleTarget) = 1;
  BOOST_TEST(cast<MapContainer>(Containers.at(CName)).get(ExampleTarget) == 1);
}

class TestPipe {

public:
  static constexpr auto Name = "test";

  std::vector<ContractGroup> getContract() const {
    return {
      ContractGroup(RootKind, 0, RootKind2, 0, InputPreservation::Preserve)
    };
  }

  void run(const ExecutionContext &,
           const MapContainer &Source,
           MapContainer &Target) {
    Source.enumerate().dump();
    auto SrcCopy = Source;
    for (const auto &Element : SrcCopy.getMap())
      if (&Element.first.getKind() == &RootKind) {
        pipeline::Target NewTar(RootKind2);
        NewTar.dump();
        Target.get(NewTar) = Element.second;
        Target.enumerate().dump();
      }
    Target.enumerate().dump();
  }
};

BOOST_AUTO_TEST_CASE(PipeCanBeWrapper) {
  Context Context;
  ExecutionContext EC(Context, nullptr);
  MapContainer Map("random-name");
  Map.get({ {}, RootKind }) = 1;
  TestPipe Enf;
  Enf.run(EC, Map, Map);
  BOOST_TEST(Map.get({ {}, RootKind2 }) == 1);
}

BOOST_AUTO_TEST_CASE(InputOutputContractExactPassForward) {
  ContainerToTargetsMap Targets;
  Targets[CName].emplace_back(Target({}, RootKind));

  ContractGroup Contract1(RootKind);
  Contract1.deduceResults(Context(), Targets, { CName });
  BOOST_TEST((&Targets[CName][0].getKind() == &RootKind));
}

BOOST_AUTO_TEST_CASE(InputOutputContractExactExactForward) {
  ContainerToTargetsMap Targets;
  Targets[CName].emplace_back(Target({}, RootKind));
  ContractGroup Contract1(RootKind, 0, RootKind2, 0);
  Contract1.deduceResults(Context(), Targets, { CName });
  BOOST_TEST((&Targets[CName][0].getKind() == &RootKind2));
}

BOOST_AUTO_TEST_CASE(InputOutputContractExactKillForward) {
  ContainerToTargetsMap Targets;
  Targets[CName].emplace_back(Target({}, RootKind));
  ContractGroup Contract1(RootKind, 0, RootKind3, 0, InputPreservation::Erase);
  Contract1.deduceResults(Context(), Targets, { CName });
  BOOST_TEST((Targets[CName].size() == 1));
  BOOST_TEST((&Targets[CName][0].getKind() == &RootKind3));
}

BOOST_AUTO_TEST_CASE(InputOutputContractMultiLine) {
  ContainerToTargetsMap Targets;
  Targets["third"].emplace_back(Target({}, RootKind));

  Contract FirstPart(RootKind2, 0, RootKind, 2, InputPreservation::Preserve);
  Contract SecondPart(RootKind3, 1, RootKind, 2, InputPreservation::Preserve);
  ContractGroup Contract({ FirstPart, SecondPart });

  auto Res = Contract.deduceRequirements(Context(),
                                         Targets,
                                         { "first", "second", "third" });

  BOOST_TEST(Res["first"].size() == 1U);
  BOOST_TEST(Res["second"].size() == 1U);
}

BOOST_AUTO_TEST_CASE(InputOutputContractDerivedPassForward) {
  ContainerToTargetsMap Targets;
  Targets[CName].emplace_back(Target({}, RootKind2));

  ContractGroup Contract1(RootKind, 0, RootKind2, 0);
  Contract1.deduceResults(Context(), Targets, { CName });
  BOOST_TEST((&Targets[CName][0].getKind() == &RootKind2));
}

BOOST_AUTO_TEST_CASE(InputOutputContractDerivedExactForward) {
  ContainerToTargetsMap Targets;
  Targets[CName].emplace_back(Target({}, RootKind));

  ContractGroup Contract1(RootKind, 0, RootKind, 0);
  Contract1.deduceResults(Context(), Targets, { CName });
  BOOST_TEST((&Targets[CName][0].getKind() == &RootKind));
  BOOST_TEST((Targets[CName].size() == 1));
}

BOOST_AUTO_TEST_CASE(InputOutputContractExactPassBackward) {
  ContainerToTargetsMap Targets;
  Targets[CName].emplace_back(Target({}, RootKind));

  ContractGroup Contract1(RootKind);
  auto Res = Contract1.deduceRequirements(Context(), Targets, { CName });
  BOOST_TEST((&Res[CName][0].getKind() == &RootKind));
}

BOOST_AUTO_TEST_CASE(InputOutputContractExactExactBackward) {
  ContainerToTargetsMap Targets;
  Targets[CName].emplace_back(Target({}, RootKind2));

  ContractGroup Contract1(RootKind, 0, RootKind2, 0);
  auto Res = Contract1.deduceRequirements(Context(), Targets, { CName });
  BOOST_TEST((&Res[CName][0].getKind() == &RootKind));
}

BOOST_AUTO_TEST_CASE(InputOutputContractDerivedPassBackward) {
  ContainerToTargetsMap Targets;
  Targets[CName].emplace_back(Target({}, RootKind2));

  ContractGroup Contract1(RootKind);
  auto Res = Contract1.deduceRequirements(Context(), Targets, { CName });
  BOOST_TEST((&Res[CName][0].getKind() == &RootKind2));
}

BOOST_AUTO_TEST_CASE(InputOutputContractDerivedExactBackward) {
  ContainerToTargetsMap Targets;
  Targets[CName].emplace_back(Target({}, RootKind2));

  ContractGroup Contract1(RootKind, 0, RootKind2, 0);
  auto Res = Contract1.deduceRequirements(Context(), Targets, { CName });
  BOOST_TEST((&Res[CName][0].getKind() == &RootKind));
}

BOOST_AUTO_TEST_CASE(InputOutputContractExactExactFineGrainedBackward) {
  ContainerToTargetsMap Targets;
  Targets.add(CName, { "f1" }, FunctionKind);

  ContractGroup Contract1(RootKind, 0, FunctionKind, 0);
  auto Res = Contract1.deduceRequirements(Context(), Targets, { CName });
  BOOST_TEST((&Res[CName][0].getKind() == &RootKind));
  BOOST_TEST((Res[CName][0].getPathComponents().size() == 0));
}

BOOST_AUTO_TEST_CASE(InputOutputContractExactExactFineGrainedForward) {
  ContainerToTargetsMap Targets;
  Targets[CName].emplace_back(Target({}, RootKind));

  ContractGroup Contract1(RootKind, 0, FunctionKind, 0);
  Contract1.deduceResults(Context(), Targets, { CName });
  BOOST_TEST((&Targets[CName][0].getKind() == &FunctionKind));
  BOOST_TEST((Targets[CName][0].getPathComponents().size() == 1));
}

static void checkIfContains(auto &TargetRange, const Kind &K) {
  const auto ToFind = [&K](const Target &Target) {
    return &Target.getKind() == &K;
  };
  BOOST_TEST(llvm::find_if(TargetRange, ToFind) != TargetRange.end());
}

BOOST_AUTO_TEST_CASE(InputOutputContractMupltipleInputTest) {
  ContainerToTargetsMap Targets;
  Targets[CName].emplace_back(Target(RootKind2));
  Targets[CName].emplace_back(Target(RootKind));

  ContractGroup Contract1(RootKind, 0, RootKind2, 0);
  auto Res = Contract1.deduceRequirements(Context(), Targets, { CName });

  const auto &ProducedResults = Res[CName];
  checkIfContains(ProducedResults, RootKind);
}

BOOST_AUTO_TEST_CASE(InputOutputContractPreserved) {
  ContainerToTargetsMap Targets;
  Targets[CName].emplace_back(Target({}, RootKind2));
  Targets[CName].emplace_back(Target({}, RootKind));

  ContractGroup Contract1(RootKind,
                          0,
                          RootKind2,
                          0,
                          InputPreservation::Preserve);
  Contract1.deduceResults(Context(), Targets, { CName });
  const auto &ProducedResults = Targets[CName];
  checkIfContains(ProducedResults, RootKind2);
  checkIfContains(ProducedResults, RootKind);
  checkIfContains(ProducedResults, RootKind2);
}

BOOST_AUTO_TEST_CASE(InputOutputContractPreservedBackwardMain) {
  ContainerToTargetsMap Targets;
  Targets[CName].emplace_back(Target({}, RootKind2));

  ContractGroup Contract1(RootKind,
                          0,
                          RootKind2,
                          0,
                          InputPreservation::Preserve);
  auto Res = Contract1.deduceRequirements(Context(), Targets, { CName });
  BOOST_TEST((&Res[CName][0].getKind() == &RootKind));
}

BOOST_AUTO_TEST_CASE(InputOutputContractPreservedBackwardSecondary) {
  ContainerToTargetsMap Targets;
  Targets[CName].emplace_back(Target({}, RootKind));

  ContractGroup Contract1(RootKind,
                          0,
                          RootKind2,
                          0,
                          InputPreservation::Preserve);
  auto Res = Contract1.deduceRequirements(Context(), Targets, { CName });
  BOOST_TEST((&Res[CName][0].getKind() == &RootKind));
}

BOOST_AUTO_TEST_CASE(StepCanCloneAndRun) {
  Context Context;
  Runner Pip(Context);

  ContainerSet Containers;
  auto Factory = getMapFactoryContainer();
  Containers.add(CName, Factory, Factory("dont-care"));
  Step Step(Context,
            "first-step",
            "",
            std::move(Containers),
            PipeWrapper::bind<TestPipe>(CName, CName));

  ContainerToTargetsMap Targets;
  Targets[CName].emplace_back(RootKind2);

  Containers = ContainerSet();
  auto Factory2 = getMapFactoryContainer();
  Containers.add(CName, Factory, Factory("dont-care"));
  cast<MapContainer>(Containers[CName]).get(Target({}, RootKind)) = 1;
  auto Result = Step.run(std::move(Containers),
                         std::vector({ PipeExecutionEntry({}, {}) }));

  auto &Cont = cast<MapContainer>(Result.at(CName));
  BOOST_TEST(Cont.get(Target({}, RootKind2)) == 1);
}

BOOST_AUTO_TEST_CASE(PipelineCanBeManuallyExectued) {
  ContainerFactorySet Registry;
  Registry.registerDefaultConstructibleFactory<MapContainer>(CName);

  Context Context;
  Runner Pip(Context);
  Pip.addStep(Step(Context,
                   "first-step",
                   "",
                   Registry.createEmpty(),
                   PipeWrapper::bind<TestPipe>(CName, CName)));

  auto Containers = Registry.createEmpty();
  auto &C1 = Containers.getOrCreate<MapContainer>(CName);
  C1.get(Target(RootKind)) = 1;

  auto
    Res = Pip["first-step"]
            .run(std::move(Containers),
                 std::vector<PipeExecutionEntry>({ PipeExecutionEntry({},
                                                                      {}) }));
  BOOST_TEST(cast<MapContainer>(Res.at(CName)).get(Target(RootKind2)) == 1);
  const auto &StartingContainer = Pip["first-step"]
                                    .containers()
                                    .getOrCreate<MapContainer>(CName);
  auto Val = StartingContainer.get(Target(RootKind2));
  BOOST_TEST(Val == 1);
}

BOOST_AUTO_TEST_CASE(SingleElementPipelineCanBeRunned) {
  Context Context;
  Runner Pip(Context);

  ContainerSet Content;
  auto Factory = getMapFactoryContainer();
  Content.add(CName, Factory, Factory("dont-care"));
  auto &C1 = cast<MapContainer>(Content[CName]);
  C1.get(Target(RootKind)) = 1;

  Step StepToAdd(Context, "first-step", "", std::move(Content));
  Pip.addStep(std::move(StepToAdd));
  ContainerSet &BCI = Pip["first-step"].containers();
  BOOST_TEST(cast<MapContainer>(BCI.at(CName)).get(Target(RootKind)) == 1);

  ContainerSet Containers2;
  Containers2.add(CName, Factory, make_unique<MapContainer>("dont-care"));
  Pip.addStep(Step(Context,
                   "end",
                   "",
                   std::move(Containers2),
                   Pip["first-step"],
                   PipeWrapper::bind<TestPipe>(CName, CName)));

  ContainerToTargetsMap Targets;
  Targets[CName].emplace_back(Target(RootKind2));
  auto Error = Pip.run("end", Targets);
  BOOST_TEST(!Error);
  ContainerSet &BC = Pip["end"].containers();
  BOOST_TEST(cast<MapContainer>(BC.at(CName)).get(Target(RootKind2)) == 1);
}

class FineGrainPipe {

public:
  static constexpr auto Name = "fine-grain";

  std::vector<ContractGroup> getContract() const {
    return {
      ContractGroup(RootKind, 0, FunctionKind, 1, InputPreservation::Preserve)
    };
  }

  void
  run(ExecutionContext &EC, const MapContainer &Source, MapContainer &Target) {
    for (const auto &Element : Source.getMap()) {

      if (&Element.first.getKind() != &RootKind)
        continue;

      auto PathComponents = Element.first.getPathComponents();
      PathComponents.emplace_back("f1");
      Target.get({ std::move(PathComponents), FunctionKind }) = Element.second;

      PathComponents = Element.first.getPathComponents();
      PathComponents.emplace_back("f2");
      Target.get({ std::move(PathComponents), FunctionKind }) = Element.second;

      EC.commitAllFor(Target);
    }
  }
};

class CopyPipe {

public:
  static constexpr auto Name = "copy";

  std::vector<ContractGroup> getContract() const {
    return { ContractGroup(FunctionKind,
                           0,
                           FunctionKind,
                           1,
                           InputPreservation::Preserve) };
  }

  void
  run(ExecutionContext &EC, const MapContainer &Source, MapContainer &Target) {
    for (const auto &Element : Source.getMap())
      if (&Element.first.getKind() == &FunctionKind)
        Target.get(Element.first) = Element.second;

    EC.commitAllFor(Target);
  }
};

BOOST_AUTO_TEST_CASE(SingleElementPipelineBackwardFinedGrained) {
  Context Context;
  Runner Pipeline(Context);
  Pipeline.addDefaultConstructibleFactory<MapContainer>(CName);

  const std::string Name = "first-step";
  Pipeline.emplaceStep("", Name, "");
  Pipeline.emplaceStep(Name,
                       "end",
                       "",
                       PipeWrapper::bind<FineGrainPipe>(CName, CName));

  auto &Container(Pipeline[Name].containers().getOrCreate<MapContainer>(CName));
  Container.get(Target(RootKind)) = 1;

  ContainerToTargetsMap Targets;
  Targets.add(CName, { "f1" }, FunctionKind);

  auto Error = Pipeline.run("end", Targets);
  BOOST_TEST(!Error);
  auto &FinalContainer = Pipeline["end"].containers().get<MapContainer>(CName);
  Target FinalTarget({ "f1" }, FunctionKind);
  auto Val = FinalContainer.get(FinalTarget);

  BOOST_TEST(Val == 1);
}

BOOST_AUTO_TEST_CASE(DifferentNamesAreNotCompatible) {
  Target Target1({ "f1-wrong" }, FunctionKind);
  Target Target2({ "f1" }, FunctionKind);

  BOOST_TEST((Target1 != Target2));
  BOOST_TEST((Target2 != Target1));
}

BOOST_AUTO_TEST_CASE(DifferentNamesRootOnlyAreNotCompatibleSet) {
  Target Target1({ "f1-wrong" }, FunctionKind);
  Target Target2({ "f1" }, FunctionKind);
  TargetsList TargetList;
  TargetList.push_back(Target1);

  BOOST_TEST(not TargetList.contains(Target2));
  BOOST_TEST(TargetList.contains(Target1));
}

BOOST_AUTO_TEST_CASE(SingleElementPipelineFailure) {
  Context Context;
  Runner Pipeline(Context);
  Pipeline.addDefaultConstructibleFactory<MapContainer>(CName);

  const std::string Name = "first-step";
  Pipeline.emplaceStep("",
                       Name,
                       "",
                       PipeWrapper::bind<FineGrainPipe>(CName, CName));
  Pipeline.emplaceStep(Name, "end", "");

  auto &Container(Pipeline[Name].containers().getOrCreate<MapContainer>(CName));
  Container.get(Target(RootKind)) = 1;

  ContainerToTargetsMap Targets;
  Targets.add(CName, {}, RootKind2);

  auto Error = Pipeline.run("end", Targets);
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

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
    AU.addRequired<LoadExecutionContextPass>();
  }

  bool runOnModule(llvm::Module &M) override {
    M.getFunction("root")->eraseFromParent();
    makeF(M, "f1");

    auto &LECP = getAnalysis<LoadExecutionContextPass>();
    LECP.get()->commitAllFor(LECP.getContainerName());

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

static llvm::RegisterPass<IdentityPass> X2("identity-pass", "identity-pass");

struct LLVMPassFunctionCreator {
  static constexpr auto Name = "function-creator";

  std::vector<ContractGroup> getContract() const {
    return { ContractGroup(RootKind, 0, FunctionKind) };
  }

  void registerPasses(llvm::legacy::PassManager &Manager) {
    Manager.add(new FunctionInserterPass());
  }
};

struct LLVMPassFunctionIdentity {
  static constexpr auto Name = "identity";

  std::vector<ContractGroup> getContract() const {
    return { ContractGroup(FunctionKind) };
  }

  void registerPasses(llvm::legacy::PassManager &Manager) {
    Manager.add(new IdentityPass());
  }
};

BOOST_AUTO_TEST_CASE(SingleElementLLVMPipelineBackwardFinedGrained) {
  llvm::LLVMContext C;

  Context Context;
  Runner Pipeline(Context);
  Pipeline
    .addContainerFactory(CName,
                         ContainerFactory::fromGlobal<LLVMContainer>(&Context,
                                                                     &C));

  const std::string Name = "first-step";
  Pipeline.emplaceStep("", Name, "");
  Pipeline
    .emplaceStep(Name,
                 "end",
                 "",
                 LLVMContainer::wrapLLVMPasses(CName,
                                               LLVMPassFunctionCreator(),
                                               LLVMPassFunctionIdentity()));

  auto &C1(Pipeline[Name].containers().getOrCreate<LLVMContainer>(CName));
  makeF(C1.getModule(), "root");

  ContainerToTargetsMap Targets;
  Targets.add(CName, Target({ "f1" }, FunctionKind));

  auto Error = Pipeline.run("end", Targets);
  BOOST_TEST(!Error);

  const auto &Final = Pipeline["end"].containers().get<LLVMContainer>(CName);
  const auto *F = Final.getModule().getFunction("f1");

  BOOST_TEST(F != nullptr);
}

BOOST_AUTO_TEST_CASE(LLVMPurePipe) {
  llvm::LLVMContext C;

  Context Context;
  Runner Pipeline(Context);
  Pipeline
    .addContainerFactory(CName,
                         ContainerFactory::fromGlobal<LLVMContainer>(&Context,
                                                                     &C));

  const std::string Name = "first-_step";
  PureLLVMPassWrapper IdentityPass("identity-pass");
  Pipeline.emplaceStep("", Name, "");
  Pipeline.emplaceStep(Name,
                       "end",
                       "",
                       LLVMContainer::wrapLLVMPasses(CName,
                                                     LLVMPassFunctionCreator(),
                                                     IdentityPass));

  auto &C1 = Pipeline[Name].containers().getOrCreate<LLVMContainer>(CName);
  makeF(C1.getModule(), "root");

  ContainerToTargetsMap Targets;
  Targets.add(CName, Target({ "f1" }, FunctionKind));

  auto Error = Pipeline.run("end", Targets);
  BOOST_TEST(!Error);

  const auto &Final = Pipeline["end"].containers().get<LLVMContainer>(CName);
  const auto *F = Final.getModule().getFunction("f1");

  BOOST_TEST(F != nullptr);
}

BOOST_AUTO_TEST_CASE(SingleElementPipelineForwardFinedGrained) {
  Context Context;
  Runner Pipeline(Context);
  Pipeline.addDefaultConstructibleFactory<MapContainer>(CName);

  const std::string Name = "first-step";
  Pipeline.emplaceStep("", Name, "");
  Pipeline.emplaceStep(Name,
                       "end",
                       "",
                       PipeWrapper::bind<FineGrainPipe>(CName, CName));

  auto &C1 = Pipeline[Name].containers().getOrCreate<MapContainer>(CName);
  C1.get(Target({}, RootKind)) = 1;
  auto &C2 = Pipeline["end"].containers().getOrCreate<MapContainer>(CName);

  const auto T = Target({ "f1" }, FunctionKind);
  const auto T2 = Target({ "f2" }, FunctionKind);
  C2.get(T) = 1;
  C2.get(T2) = 1;

  llvm::StringMap<ContainerToTargetsMap> Invalidations;
  Invalidations[Name].add(CName, {}, RootKind);

  auto Error = Pipeline.getInvalidations(Invalidations);
  BOOST_TEST(!Error);

  const auto &EndContainerInvalidations = Invalidations["end"][CName];
  BOOST_TEST(not EndContainerInvalidations.empty());
  BOOST_TEST((EndContainerInvalidations == TargetsList({ T, T2 })));
}

BOOST_AUTO_TEST_CASE(SingleElementPipelineInvalidation) {
  Context Context;
  Runner Pipeline(Context);
  Pipeline.addDefaultConstructibleFactory<MapContainer>(CName);

  const std::string Name = "first-step";
  Pipeline.emplaceStep("", Name, "");
  Pipeline.emplaceStep(Name,
                       "end",
                       "",
                       PipeWrapper::bind<FineGrainPipe>(CName, CName));

  auto &C1 = Pipeline[Name].containers().getOrCreate<MapContainer>(CName);
  C1.get(Target({}, RootKind)) = 1;
  auto &C2 = Pipeline["end"].containers().getOrCreate<MapContainer>(CName);

  const auto T = Target({ "f1" }, FunctionKind);
  const auto T2 = Target({ "f2" }, FunctionKind);
  C2.get(T) = 1;
  C2.get(T2) = 1;

  Target ToKill({}, RootKind);

  llvm::StringMap<ContainerToTargetsMap> Invalidations;
  auto Error = Pipeline.getInvalidations(ToKill, Invalidations);
  BOOST_TEST(!Error);
  const auto &QuantifOfInvalidated = Invalidations["end"][CName]
                                       .front()
                                       .getPathComponents();
  BOOST_TEST((QuantifOfInvalidated.front() == "f1"));
}

BOOST_AUTO_TEST_CASE(SingleElementPipelineWithRemove) {
  Context Context;
  Runner Pipeline(Context);
  Pipeline.addDefaultConstructibleFactory<MapContainer>(CName);

  const std::string Name = "first-step";
  Pipeline.emplaceStep("",
                       Name,
                       "",
                       PipeWrapper::bind<FineGrainPipe>(CName, CName));
  Pipeline.emplaceStep(Name, "end", "");

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
  StepDeclaration SDeclaration{ "first-step",
                                { { "fine-grain", { CName, CName } } } };
  BranchDeclaration BDeclaration{ "", { std::move(SDeclaration) } };
  PipelineDeclaration PDeclaration{ "revng-test",
                                    { { CName, "map-container" } },
                                    { std::move(BDeclaration) } };

  auto Context = Context::fromRegistry(Registry::registerAllKinds());
  Loader Loader(Context);
  Loader.addDefaultConstructibleContainer<MapContainer>("map-container");
  Loader.registerPipe<FineGrainPipe>("fine-grain");

  auto MaybePipeline = Loader.load(PDeclaration);
  BOOST_TEST(!!MaybePipeline);
  auto &Pipeline = *MaybePipeline;
  const std::string Name = "first-step";
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
                       Component: revng-test
                       Containers:
                         - Name:            container-name
                           Type:            map-container
                       Branches:
                         - Steps:
                             - Name:            first-step
                               Pipes:
                                 - Type:            fine-grain
                                   UsedContainers:
                                     - container-name
                                     - container-name
                       )");

BOOST_AUTO_TEST_CASE(LoaderTestFromYaml) {
  Context Context;
  Loader Loader(Context);
  Loader.addDefaultConstructibleContainer<MapContainer>("map-container");
  Loader.registerPipe<FineGrainPipe>("fine-grain");
  auto MaybePipeline = Loader.load(Pipeline);
  BOOST_TEST(!!MaybePipeline);
}

static const std::string PipelineTree(R"(---
                       Component: revng-test
                       Containers:
                         - Name:            container-name
                           Type:            map-container
                       Branches:
                         - Steps:
                             - Name:            first-step
                               Pipes:
                                 - Type:            fine-grain
                                   UsedContainers:
                                     - container-name
                                     - container-name
                       )");

static const std::string PipelineTree2(R"(---
                       Component: revng-test
                       Containers:
                       Branches:
                         - From:      first-step
                           Steps:
                             - Name:            second-step
                       )");

static const std::string PipelineTree3(R"(---
                       Component: revng-test
                       Containers:
                       Branches:
                         - From:      first-step
                           Steps:
                             - Name:            third-step
                       )");

BOOST_AUTO_TEST_CASE(LoaderTestFromYamlTree) {
  Context Ctx;
  Loader Loader(Ctx);
  Loader.addDefaultConstructibleContainer<MapContainer>("map-container");
  Loader.registerPipe<FineGrainPipe>("fine-grain");
  vector ToLoad({ PipelineTree2, PipelineTree3, PipelineTree });
  auto MaybePipeline = Loader.load(ToLoad);
  BOOST_TEST(!!MaybePipeline);
}

BOOST_AUTO_TEST_CASE(LoaderTestFromYamlLLVM) {
  llvm::LLVMContext C;
  Context Ctx;
  Loader Loader(Ctx);

  Loader.addContainerFactory("llvm-container",
                             ContainerFactory::fromGlobal<LLVMContainer>(&Ctx,
                                                                         &C));
  auto *Name = "create-function-pass";
  Loader.registerLLVMPass<LLVMPassFunctionCreator>(Name);
  Name = "identity-pass";
  Loader.registerLLVMPass<LLVMPassFunctionIdentity>(Name);

  std::string LLVMPipeline(R"(---
                       Component: revng-test
                       Containers:
                         - Name:            custom-name
                           Type:            llvm-container
                       Branches:
                         - Steps:
                             - Name:            first-step
                               Pipes:
                                 - Type:             llvm-pipe
                                   UsedContainers:
                                     - custom-name
                                   Passes:
                                     - create-function-pass
                                     - identity-pass
                       )");

  auto MaybePipeline = Loader.load(LLVMPipeline);

  BOOST_TEST(!!MaybePipeline);
}

static revng::DirectoryPath getCurrentPath() {
  using revng::StorageClient;
  llvm::SmallString<128> ToReturn;
  llvm::sys::fs::current_path(ToReturn);
  return revng::DirectoryPath::fromLocalStorage(ToReturn.str());
}

BOOST_AUTO_TEST_CASE(SingleElementPipelinestore) {
  Context Ctx;
  Runner Pipeline(Ctx);
  revng::DirectoryPath Path = getCurrentPath();
  Pipeline.addDefaultConstructibleFactory<MapContainer>(CName);

  const std::string Name = "first-step";
  Pipeline.emplaceStep("",
                       Name,
                       "",
                       PipeWrapper::bind<FineGrainPipe>(CName, CName));
  Pipeline.emplaceStep(Name, "end", "");

  auto &C1 = Pipeline[Name].containers().getOrCreate<MapContainer>(CName);
  C1.get(Target({}, RootKind)) = 1;

  BOOST_TEST((!Pipeline.store(Path)));

  auto &Container(Pipeline[Name].containers().getOrCreate<MapContainer>(CName));

  BOOST_TEST((Container.get(Target({}, RootKind)) == 1));
  Container.get(Target({}, RootKind)) = 2;
  BOOST_TEST((Container.get(Target({}, RootKind)) == 2));
  BOOST_TEST((!Pipeline.load(Path)));
  BOOST_TEST(Pipeline[Name].containers().containsOrCanCreate(CName));
  BOOST_TEST(not Pipeline[Name].containers().contains(CName));
}

BOOST_AUTO_TEST_CASE(SingleElementPipelinestoreWithOverrides) {
  Context Ctx;
  Loader Loader(Ctx);
  Loader.addDefaultConstructibleContainer<MapContainer>("map-container");
  Loader.registerPipe<FineGrainPipe>("fine-grain");
  auto MaybePipeline = Loader.load(Pipeline);
  BOOST_TEST(!!MaybePipeline);
  auto &Pipeline = *MaybePipeline;
  auto *Mapping = "DontCareSourceFile:first-step/container-name";
  auto MaybeMapping = PipelineFileMapping::parse(Mapping);
  BOOST_TEST(!!MaybeMapping);

  const std::string Name = "first-step";
  auto &Container(Pipeline[Name].containers().getOrCreate<MapContainer>(CName));

  Container.get(Target({}, RootKind)) = 1;
  BOOST_TEST((!MaybeMapping->store(Pipeline)));
  Container.get(Target({}, RootKind)) = 2;
  BOOST_TEST((Container.get(Target({}, RootKind)) == 2));
  BOOST_TEST((!MaybeMapping->load(Pipeline)));
  BOOST_TEST((Container.get(Target({}, RootKind)) == 1));
}

class EnumerableContainerExample
  : public EnumerableContainer<EnumerableContainerExample> {
public:
  static char ID;

  static inline const llvm::StringRef MIMEType = "";
  EnumerableContainerExample(Context &Context, llvm::StringRef Name) :
    EnumerableContainer<EnumerableContainerExample>(Context, Name) {}

  unique_ptr<ContainerBase>
  cloneFiltered(const TargetsList &Container) const final {
    return make_unique<EnumerableContainerExample>(*this);
  }

  ~EnumerableContainerExample() override = default;

  llvm::Error store(const revng::FilePath &Path) const override {
    return llvm::Error::success();
  }

  llvm::Error load(const revng::FilePath &Path) override {
    return llvm::Error::success();
  }

  llvm::Error serialize(llvm::raw_ostream &OS) const final {
    return llvm::Error::success();
  }

  llvm::Error deserialize(const llvm::MemoryBuffer &Buffer) final {
    return llvm::Error::success();
  }

  llvm::Error extractOne(llvm::raw_ostream &OS,
                         const Target &Target) const override {
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
  enumerate(const EnumerableContainerExample &Container) const final {
    TargetsList ToReturn;

    llvm::copy(Container.Targets, back_inserter(ToReturn));
    return ToReturn;
  }

  bool contains(const Target &Target,
                const EnumerableContainerExample &Container) const {
    return Container.Targets.contains(Target);
  }

  bool remove(const TargetsList &Targets,
              EnumerableContainerExample &Container) const {

    bool ErasedAll = true;
    for (const Target &Target : Targets)
      ErasedAll = remove(Target, Container) and ErasedAll;

    return ErasedAll;
  }

  bool remove(const Target &Target,
              EnumerableContainerExample &Container) const {

    if (not contains(Target, Container))
      return false;

    Container.Targets.erase(Target);
    return true;
  }
};

static ExampleContainerInpsector Example;

BOOST_AUTO_TEST_CASE(EnumerableContainersTest) {
  Context Ctx;
  EnumerableContainerExample Example(Ctx, "dont-care");
  Target T({}, RootKind);
  Example.Targets.insert(T);
  BOOST_TEST(Example.contains(T));
  BOOST_TEST(Example.remove(TargetsList({ T })));
  BOOST_TEST(not Example.contains(T));
}

class LLVMInspectorExample : public LLVMKind {
public:
  using LLVMKind::LLVMKind;

  std::optional<Target>
  symbolToTarget(const llvm::Function &Symbol) const override {
    return Target({ Symbol.getName() }, FunctionKind);
  }

  void appendAllTargets(const pipeline::Context &Context,
                        pipeline::TargetsList &Out) const override {
    Out.push_back(Target("f1", *this));
    Out.push_back(Target("f2", *this));
  }
};

class LLVMRootInspectorExample : public LLVMKind {
public:
  using LLVMKind::LLVMKind;

  std::optional<Target>
  symbolToTarget(const llvm::Function &Symbol) const override {
    return Target({}, RootKind);
  }

  void appendAllTargets(const pipeline::Context &Context,
                        pipeline::TargetsList &Out) const override {
    Out.push_back(Target(*this));
  }
};

static LLVMInspectorExample ExampleLLVMInspector("dc", FunctionRank);
static LLVMRootInspectorExample ExampleLLVMRootInspector("dc2", Root);

BOOST_AUTO_TEST_CASE(LLVMKindTest) {
  llvm::LLVMContext C;
  using Cont = LLVMContainer;

  Context Ctx;
  Runner Pipeline(Ctx);
  Pipeline.addContainerFactory(CName,
                               ContainerFactory::fromGlobal<Cont>(&Ctx, &C));

  Pipeline.emplaceStep("", "first-step", "");
  Pipeline.emplaceStep("first-step",
                       "end",
                       "",
                       Cont::wrapLLVMPasses(CName, LLVMPassFunctionCreator()));

  makeF(Pipeline["first-step"]
          .containers()
          .getOrCreate<Cont>(CName)
          .getModule(),
        "root");

  ContainerToTargetsMap Targets;
  Targets.add(CName, Target({ "f1" }, FunctionKind));

  auto Error = Pipeline.run("end", Targets);
  BOOST_TEST(!Error);

  const auto &Final = Pipeline["end"].containers().get<Cont>(CName);
  const auto *F = Final.getModule().getFunction("f1");

  BOOST_TEST(F != nullptr);
}

class InspectorKindExample : public LLVMKind {
public:
  InspectorKindExample() : LLVMKind("example-name", FunctionRank) {}

  std::optional<Target>
  symbolToTarget(const llvm::Function &Symbol) const final {
    return Target({ Symbol.getName() }, *this);
  }
  void appendAllTargets(const pipeline::Context &Context,
                        pipeline::TargetsList &Out) const override {
    Out.push_back(Target("f1", *this));
    Out.push_back(Target("f2", *this));
  }

  ~InspectorKindExample() override {}
};

static InspectorKindExample InspKindExample;

BOOST_AUTO_TEST_CASE(InspectorKindTest) {
  Context Ctx;
  llvm::LLVMContext C;

  using Cont = LLVMContainer;
  auto Factory = ContainerFactory::fromGlobal<Cont>(&Ctx, &C);

  auto Container = Factory("dont-care");

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

  const std::string Name = "first-step";
  const std::string SecondName = "second-step";
  Pipeline.emplaceStep("", Name, "");
  Pipeline.emplaceStep(Name,
                       SecondName,
                       "",
                       PipeWrapper::bind<FineGrainPipe>(CName, CName));
  Pipeline.emplaceStep(SecondName,
                       "end",
                       "",
                       PipeWrapper::bind<CopyPipe>(CName, CName2));

  auto &C1 = Pipeline[Name].containers().getOrCreate<MapContainer>(CName);
  auto &C1End = Pipeline["end"].containers().getOrCreate<MapContainer>(CName);
  auto &C2End = Pipeline["end"].containers().getOrCreate<MapContainer>(CName2);

  const auto T = Target(RootKind);
  C1.get(T) = 1;

  const auto ToProduce = Target({ "f1" }, FunctionKind);
  ContainerToTargetsMap Map;
  Map[CName2].emplace_back(ToProduce);
  cantFail(Pipeline.run("end", Map));

  BOOST_TEST(C1.get(T) == 1);
  BOOST_TEST(C1End.get(T) == 1);

  BOOST_TEST(C2End.get(ToProduce) == 1);

  pipeline::TargetInStepSet Invalidations;
  Invalidations[Name][CName].push_back(T);

  auto Error = Pipeline.getInvalidations(Invalidations);
  BOOST_TEST(!Error);

  Error = Pipeline.invalidate(Invalidations);

  BOOST_TEST(!Error);

  BOOST_TEST(C1.get(T) == 0);
  BOOST_TEST(C1End.get(T) == 0);

  BOOST_TEST(C2End.get(ToProduce) == 0);
}

class ArgumentTestAnalysis {
public:
  constexpr static const char *Name = "dont-care";

  constexpr static std::tuple Options = {
    pipeline::Option("first", 10),
    pipeline::Option("second", "something"),
    pipeline::Option("third", static_cast<uint64_t>(32))
  };

  std::vector<std::vector<pipeline::Kind *>> AcceptedKinds = { { &RootKind } };

  void run(const ExecutionContext &EC,
           const MapContainer &Cont,
           int First,
           std::string Second,
           uint64_t Third) {
    BOOST_TEST(First == 10);
    BOOST_TEST(Second == "something");
    BOOST_TEST(Third == 32UL);
  }
};

BOOST_AUTO_TEST_CASE(PipeOptions) {
  RegisterAnalysis<ArgumentTestAnalysis> Dummy;
  pipeline::AnalysisWrapperImpl W(ArgumentTestAnalysis(), { "container-name" });
  Context Ctx;
  ExecutionContext ExecutionCtx(Ctx, nullptr);
  ContainerSet Set;
  auto Factory = ContainerFactory::create<MapContainer>();
  Set.add("container-name", Factory);
  Set["container-name"];
  if (auto Error = W.run(ExecutionCtx, Set, {}))
    BOOST_FAIL("unreachable");
}

BOOST_AUTO_TEST_SUITE_END()
