/// \file StringContainerLibrary.cpp
/// The kind associated with non isolated root.

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

#include "revng/Pipeline/AllRegistries.h"
#include "revng/Pipeline/CopyPipe.h"
#include "revng/Pipeline/RegisterContainerFactory.h"
#include "revng/Pipeline/Target.h"
#include "revng/Pipes/Kinds.h"
#include "revng/Pipes/ModelGlobal.h"
#include "revng/Support/Assert.h"

using std::string;
using namespace llvm;
using namespace llvm::cl;
using namespace pipeline;
using namespace ::revng::kinds;

class StringKindType : public Kind {
  using Kind::Kind;
  void appendAllTargets(const pipeline::Context &Ctx,
                        pipeline::TargetsList &Out) const override {
    Out.push_back(Target("f1", *this));
  }
};

static StringKindType StringKind("string-kind", revng::ranks::Function, {}, {});

class StringContainer : public Container<StringContainer> {
public:
  StringContainer(llvm::StringRef Name) : Container<StringContainer>(Name) {}
  ~StringContainer() override = default;

  inline static const llvm::StringRef MIMEType = "string";
  inline static const char *Name = "string-container";
  static char ID;

  std::unique_ptr<ContainerBase>
  cloneFiltered(const TargetsList &Container) const final {
    auto ToReturn = std::make_unique<StringContainer>(this->name());
    for (const auto &Target : Container)
      ToReturn->insert(Target);
    return ToReturn;
  }

  static std::vector<pipeline::Kind *> possibleKinds() {
    return { &StringKind };
  }

  void insert(const Target &Target) {
    ContainedStrings.insert(toString(Target));
  }

  bool contains(const Target &Target) const {
    return ContainedStrings.contains(Target.getPathComponents().back());
  }

  bool remove(const TargetsList &Targets) override {
    bool RemovedAll = true;
    for (const auto &Target : Targets)
      RemovedAll = remove(Target) && RemovedAll;

    return RemovedAll;
  }

  bool remove(const Target &Target) {
    if (not contains(Target))
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
    return Target.getPathComponents().front();
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

static RegisterPipe<CopyPipe<StringKindType, &StringKind, StringContainer>> E1;
static const std::string Name = "string-container";
static RegisterDefaultConstructibleContainer<StringContainer> C;
static RegisterRole R(Name, "string-role");
