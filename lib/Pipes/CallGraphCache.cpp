#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"

#include "revng/Pipeline/Context.h"
#include "revng/Pipeline/Contract.h"
#include "revng/Pipeline/RegisterLLVMPass.h"
#include "revng/Pipeline/RegisterPipe.h"
#include "revng/Pipes/CallGraphCache.h"

using namespace revng::pipes;
using namespace pipeline;

template<>
const char pipeline::YamlizableGlobal<revng::pipes::CallGraphCache>::ID = '0';

class CreateCallGraphCachePass : public llvm::ModulePass {
public:
  static char ID;
  bool runOnModule(llvm::Module &Module) override {
    ToWrite.Name = "ASD";
    return true;
  }

  CreateCallGraphCachePass(CallGraphCache &ToWrite) :
    llvm::ModulePass(ID), ToWrite(ToWrite) {}

private:
  CallGraphCache &ToWrite;
};

char CreateCallGraphCachePass::ID = '0';

CallGraphCache &CallGraphCache::fromContext(pipeline::Context &Ctx) {
  using namespace llvm;
  const auto &Name = CallGraphCache::GlobalName;
  return *cantFail(Ctx.getGlobal<CallGraphGlobal>(Name))->get();
}
const CallGraphCache &
CallGraphCache::fromContext(const pipeline::Context &Ctx) {
  const auto &Name = CallGraphCache::GlobalName;
  using namespace llvm;
  return *cantFail(Ctx.getGlobal<CallGraphGlobal>(Name))->get();
}

class CreateCallGraphPipe {
public:
  static constexpr auto Name = "cache-call-graph";
  std::vector<ContractGroup> getContract() const { return {}; }

  void registerPasses(Context &Ctx, llvm::legacy::PassManager &Manager) {
    Manager.add(new CreateCallGraphCachePass(CallGraphCache::fromContext(Ctx)));
  }
};

static RegisterLLVMPass<CreateCallGraphPipe> X;
