/// \file SlimDownHelpersModule.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/Constants.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"

using namespace llvm;

class SlimDownHelpersModule : public llvm::ModulePass {
public:
  static char ID;

public:
  SlimDownHelpersModule() : llvm::ModulePass(ID) {}

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {}

  bool runOnModule(llvm::Module &M) override {
    LLVMContext &Context = M.getContext();

    auto DbgKindID = M.getContext().getMDKindID("dbg");
    for (Function &F : M) {
      // Remove the body of all the functions *not* tagged with `revng_inline`.
      // Also preserve helper_initialize_env, which needs to survive but it's
      // not to be inlined.
      if (F.getSection() != "revng_inline"
          and F.getName() != "helper_initialize_env") {
        F.eraseMetadata(DbgKindID);
        SmallVector<std::pair<unsigned, MDNode *>, 8> MDs;
        F.getAllMetadata(MDs);
        F.deleteBody();
        for (auto [ID, Node] : MDs)
          F.setMetadata(ID, Node);
      }
    }

    // Mark all global variables as internal, so we can purge those that are now
    // unused due to the removal of most of helpers' bodies.
    // Also preserve the `arch_cpu_type_beacon` and all the variables with
    // revng.tags metadata, since they are CSVs produced by VariableManager.
    for (GlobalVariable &GV : M.globals()) {
      if (not GV.isDeclaration() and GV.getName() != "cpu_loop_exiting"
          and GV.getName() != "arch_cpu_type_beacon"
          and not GV.hasMetadata("revng.tags")) {
        GV.setLinkage(llvm::GlobalValue::InternalLinkage);
      }
    }

    // Assume the module M already contains the functions.
    // Collect all functions whose name starts with "helper_" to prevent DCE
    llvm::DenseSet<Function *> Helpers;
    for (Function &F : M)
      if (F.getName().startswith("helper_"))
        Helpers.insert(&F);

    PointerType *PointerType = PointerType::get(Context, 0);

    // Create array type: [N x <function pointer type>]
    ArrayType *FunctionArray = ArrayType::get(PointerType, Helpers.size());

    // Create constant array initializer with function pointers
    std::vector<Constant *> FunctionPointers;
    for (Function *F : Helpers)
      FunctionPointers.push_back(ConstantExpr::getBitCast(F, PointerType));

    Constant *Initializer = ConstantArray::get(FunctionArray, FunctionPointers);

    new GlobalVariable(M,
                       FunctionArray,
                       true,
                       GlobalValue::ExternalLinkage,
                       Initializer,
                       "helpers_list");

    return true;
  }
};

char SlimDownHelpersModule::ID = 0;
using Register = RegisterPass<SlimDownHelpersModule>;
static Register X("slim-down-helpers-module",
                  "Purge non-inline helper functions but keep their "
                  "declarations alive in a array of pointers.",
                  true,
                  true);
