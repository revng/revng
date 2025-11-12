#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/LegacyPassManager.h"
#include "llvm/PassInfo.h"
#include "llvm/Support/Progress.h"

#include "revng/PipeboxCommon/Common.h"
#include "revng/PipeboxCommon/LLVMContainer.h"
#include "revng/PipeboxCommon/Model.h"

namespace revng::pypeline::pipes {

namespace detail {

/// Base class for both PureLLVMPassesRootPipe and PureLLVMPassesPipe, which
/// factors out some common logic. Both pipes run pure LLVM passes and do not
/// read the model, hence the fact that they are plain pipes and not piperuns.
class PureLLVMPassesPipeBase {
public:
  struct Configuration {
    std::vector<std::string> Passes;
    static Configuration parse(llvm::StringRef Input);
  };

private:
  std::vector<const llvm::PassInfo *> PassInfos;

protected:
  std::string TaskName;

public:
  const std::string StaticConfiguration;
  PureLLVMPassesPipeBase(llvm::StringRef StaticConfiguration);

protected:
  llvm::legacy::PassManager makePassManager() {
    llvm::legacy::PassManager Manager;
    for (const llvm::PassInfo *PassInfo : PassInfos)
      Manager.add(PassInfo->createPass());
    return Manager;
  }
};

} // namespace detail

class PureLLVMPassesRootPipe : public detail::PureLLVMPassesPipeBase {
public:
  static constexpr llvm::StringRef Name = "PureLLVMPassesRootPipe";
  using Arguments = TypeList<
    PipeArgument<"Module", "LLVM Module to apply the LLVM passes to">>;

public:
  revng::pypeline::ObjectDependencies
  run(const Model &TheModel,
      const revng::pypeline::Request &Incoming,
      const revng::pypeline::Request &Outgoing,
      llvm::StringRef Configuration,
      LLVMRootContainer &Container) {
    if (Outgoing[0].size() == 0)
      return {};

    revng_assert(Outgoing[0].size() == 1);
    revng_assert(Outgoing[0][0]->kind() == Kinds::Binary);

    llvm::Task T(1, TaskName);
    T.advance("run on LLVMRootContainer", true);
    llvm::legacy::PassManager Manager = makePassManager();
    Manager.run(Container.getModule());

    return {};
  }
};

class PureLLVMPassesPipe : public detail::PureLLVMPassesPipeBase {
public:
  static constexpr llvm::StringRef Name = "PureLLVMPassesPipe";
  using Arguments = TypeList<
    PipeArgument<"Module", "LLVM Modules to apply the LLVM passes to">>;

public:
  revng::pypeline::ObjectDependencies
  run(const Model &TheModel,
      const revng::pypeline::Request &Incoming,
      const revng::pypeline::Request &Outgoing,
      llvm::StringRef Configuration,
      LLVMFunctionContainer &Container) {

    llvm::Task T(Outgoing[0].size(), TaskName);
    llvm::legacy::PassManager Manager = makePassManager();
    for (const ObjectID *Object : Outgoing[0]) {
      revng_assert(Object->kind() == Kinds::Function);
      const MetaAddress &Key = std::get<MetaAddress>(Object->key());
      T.advance(Key.toString(), true);
      Manager.run(Container.getModule(*Object));
    }

    return {};
  }
};

} // namespace revng::pypeline::pipes
