#include "revng/Pipeline/CopyPipe.h"
#include "revng/Pipeline/Registry.h"
#include "revng/Pipes/ModelGlobal.h"
#include "revng/Pipes/RootKind.h"

using std::string;
using namespace llvm;
using namespace llvm::cl;
using namespace pipeline;
using namespace revng::pipes;

class StringContainer : public Container<StringContainer> {
public:
  using Container<StringContainer>::Container;
  ~StringContainer() override = default;

  static char ID;

  std::unique_ptr<ContainerBase>
  cloneFiltered(const TargetsList &Container) const final {
    auto ToReturn = std::make_unique<StringContainer>(this->name());
    for (const auto &Target : Container)
      ToReturn->insert(Target);
    return ToReturn;
  }

  void insert(const Target &Target) {
    ContainedStrings.insert(toString(Target));
  }

  bool contains(const Target &Target) const {
    return ContainedStrings.count(Target.getPathComponents().back().getName());
  }

  bool remove(const TargetsList &Targets) override {
    bool RemovedAll = true;
    for (const auto &Target : Targets)
      RemovedAll = remove(Target) && RemovedAll;

    return RemovedAll;
  }

  bool remove(const Target &Target) {
    if (contains(Target))
      return false;

    ContainedStrings.erase(toString(Target));
    return true;
  }

  TargetsList enumerate() const final {
    TargetsList ToReturn;

    for (const auto &Names : ContainedStrings)
      ToReturn.emplace_back(Names, Root);

    return ToReturn;
  }

  llvm::Error storeToDisk(llvm::StringRef Path) const override {
    std::error_code EC;
    llvm::raw_fd_ostream OS(Path, EC, llvm::sys::fs::CD_CreateAlways);
    if (EC)
      return llvm::createStringError(EC,
                                     "Could not store to file %s",
                                     Path.str().c_str());

    for (const auto &S : ContainedStrings)
      OS << S << "\n";
    return llvm::Error::success();
  }

  llvm::Error loadFromDisk(llvm::StringRef Path) override {
    if (not llvm::sys::fs::exists(Path)) {
      ContainedStrings.clear();
      return llvm::Error::success();
    }

    std::ifstream OS;
    OS.open(Path.str(), std::ios::in);
    if (not OS.is_open())
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "Could not load file to file %s",
                                     Path.str().c_str());

    std::string S;
    while (getline(OS, S))
      ContainedStrings.insert(S);
    return llvm::Error::success();
  }

  const std::set<std::string> &getStrings() const { return ContainedStrings; }

private:
  static std::string toString(const Target &Target) {
    std::stringstream S;
    Target::dumpPathComponents(S, Target.getPathComponents());
    std::string ToInsert = S.str();
    return ToInsert;
  }

  void mergeBackImpl(StringContainer &&Container) override {
    for (auto &S : Container.ContainedStrings)
      ContainedStrings.insert(S);
  }

  std::set<std::string> ContainedStrings;
};

struct ExamplePass : public llvm::ModulePass {
  static char ID;

  ExamplePass() : llvm::ModulePass(ID) {}

  bool runOnModule(llvm::Module &M) override { return true; }
};

char ExamplePass::ID = '_';

static llvm::RegisterPass<ExamplePass> X2("example-pass", "ExamplePass");

char StringContainer::ID;

static RegisterPipe<CopyPipe<StringContainer>> E1(Root);
static RegisterDefaultConstructibleContainer<StringContainer> C("StringContaine"
                                                                "r");
