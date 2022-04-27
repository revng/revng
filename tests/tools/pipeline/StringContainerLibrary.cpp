/// \file StringContainerLibrary.cpp
/// \brief the kind associated to non isolated root.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Pipeline/AllRegistries.h"
#include "revng/Pipeline/CopyPipe.h"
#include "revng/Pipes/Kinds.h"
#include "revng/Pipes/ModelGlobal.h"
#include "revng/Support/Assert.h"

using std::string;
using namespace llvm;
using namespace llvm::cl;
using namespace pipeline;
using namespace ::revng::pipes;

static Kind StringKind("StringKind", &FunctionsRank);

class StringContainer : public Container<StringContainer> {
public:
  StringContainer(llvm::StringRef Name) :
    Container<StringContainer>(Name, "String") {}
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
      ToReturn.emplace_back(Names, StringKind);

    return ToReturn;
  }

  void clear() final { ContainedStrings.clear(); }

  llvm::Error serialize(llvm::raw_ostream &OS) const final {
    for (const auto &S : ContainedStrings) {
      OS << S << "\n";
    }
    return llvm::Error::success();
  }

  llvm::Error deserialize(const llvm::MemoryBuffer &Buffer) final {
    clear();
    SmallVector<llvm::StringRef, 0> Strings;
    Buffer.getBuffer().split(Strings, '\n');
    for (llvm::StringRef S : Strings) {
      if (S.empty())
        continue;
      ContainedStrings.insert(S.str());
    }
    return llvm::Error::success();
  }

  llvm::Error extractOne(llvm::raw_ostream &OS,
                         const pipeline::Target &Target) const override {
    revng_abort();
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

static RegisterPipe<CopyPipe<StringContainer>> E1(StringKind);
static const std::string Name = "StringContainer";
static RegisterDefaultConstructibleContainer<StringContainer> C(Name);
