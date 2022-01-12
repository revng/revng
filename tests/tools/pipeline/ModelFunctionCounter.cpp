#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/Pipeline/ContainerSet.h"
#include "revng/Pipeline/Context.h"
#include "revng/Pipeline/Registry.h"
#include "revng/Pipeline/Target.h"
#include "revng/Pipes/Kinds.h"
#include "revng/Pipes/ModelGlobal.h"

using namespace pipeline;
using namespace revng::pipes;

// We need a kind to name the thing we are trying to produce, that is counting
// the number of function in the model.
static Kind FunctionCounter("FunctionCounter", &RootRank);

static pipeline::Target FTarget("FunctionsCount", FunctionCounter);

class FunctionContainer : public Container<FunctionContainer> {
private:
  std::optional<int> CountedFunctions = std::nullopt;

public:
  using Container<FunctionContainer>::Container;
  static const char ID;

  virtual std::unique_ptr<ContainerBase>
  cloneFiltered(const TargetsList &Targets) const override {
    return std::make_unique<FunctionContainer>(*this);
  }
  TargetsList enumerate() const final {
    if (CountedFunctions.has_value()) {
      pipeline::Target Target(PathComponent::all(), FunctionCounter);
      return TargetsList({ Target });
    }
    return {};
  }

  bool remove(const TargetsList &Targets) final {
    if (not Targets.contains({ FTarget }))
      return false;

    CountedFunctions = std::nullopt;
    return true;
  }

  llvm::Error loadFromDisk(llvm::StringRef Path) final {
    if (not llvm::sys::fs::exists(Path)) {
      CountedFunctions = std::nullopt;
      return llvm::Error::success();
    }

    std::error_code EC;
    auto MaybeBuffer = llvm::MemoryBuffer::getFile(Path);
    if (not MaybeBuffer)
      return llvm::make_error<llvm::StringError>(MaybeBuffer.getError()
                                                   .message(),
                                                 MaybeBuffer.getError());

    auto Str = MaybeBuffer->get()->getBuffer();
    Str.consumeInteger(10, *CountedFunctions);

    return llvm::Error::success();
  }
  llvm::Error storeToDisk(llvm::StringRef Path) const final {
    if (not CountedFunctions.has_value()) {
      auto EC = llvm::sys::fs::remove(Path, true);
      if (!!EC)
        return llvm::make_error<llvm::StringError>(EC.message(), EC);
      return llvm::Error::success();
    }

    std::error_code EC;
    llvm::raw_fd_ostream Out(Path, EC, llvm::sys::fs::CD_CreateAlways);
    if (!!EC)
      return llvm::make_error<llvm::StringError>(EC.message(), EC);

    Out << *CountedFunctions;

    return llvm::Error::success();
  }

  ~FunctionContainer() override{};

  void setFunctionCount(int FCount) { CountedFunctions = FCount; }

private:
  void mergeBackImpl(FunctionContainer &&Container) final {
    CountedFunctions = Container.CountedFunctions;
  };
};

const char FunctionContainer::ID = '0';

class FunctionCounterPipe {
public:
  static constexpr auto Name = "Function Counter";

  std::array<ContractGroup, 1> getContract() const {
    return { ContractGroup(FunctionCounter, 0) };
  }

  void run(const Context &Ctx, FunctionContainer &Out) {
    Out.setFunctionCount(getModelFromContext(Ctx).Functions.size());
  }
};

class FunctionCounterRegistry : public Registry {
public:
  void registerContainersAndPipes(Loader &Loader) override {
    std::string Name("CountFunctionContainer");
    Loader.addDefaultConstructibleContainer<FunctionContainer>(Name);
    Loader.registerPipe("CountFunctionsPipe", FunctionCounterPipe());
  }

  void registerKinds(KindsRegistry &KindDictionary) override {
    KindDictionary.registerKind(FunctionCounter);
  }

  void libraryInitialization() override {}

  ~FunctionCounterRegistry() override = default;
};

static FunctionCounterRegistry Registry;
