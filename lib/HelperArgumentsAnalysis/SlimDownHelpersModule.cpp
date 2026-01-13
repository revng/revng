//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Transforms/IPO/GlobalDCE.h"

#include "revng/Model/FunctionTags.h"
#include "revng/Support/IRHelpers.h"
#include "revng/Support/SimplePassManager.h"

#include "PostProcessingHelpers.h"

using namespace llvm;

class SlimDownHelpersModule : public llvm::ModulePass {
public:
  static char ID;

public:
  SlimDownHelpersModule() : llvm::ModulePass(ID) {}

  bool runOnModule(llvm::Module &M) override {
    LLVMContext &Context = M.getContext();

    for (Function &F : M) {
      if (F.isDeclaration() or F.isIntrinsic())
        continue;

      // Remove the body of all the functions *not* tagged with `revng_inline`
      if (F.getSection() != InlineHelpersSection) {
        deleteOnlyBody(F);
      } else {
        llvm::stripDebugInfo(F);

        F.removeFnAttr(Attribute::NoInline);
        F.addFnAttr(Attribute::AlwaysInline);

        F.setDSOLocal(false);
        F.setVisibility(GlobalValue::DefaultVisibility);
      }
    }

    // Now that all of the non-`revng_inline` functions have gone, we want to
    // reduce the number of global variables with some help from GlobalDCE.
    // We have three type of variables:
    // * Special variables (`arch_cpu_type_beacon`, `cpu_loop_exiting`) which
    //   should not be touched
    // * CSVs we want to keep only the ones used by the `revng_inline`
    //   functions, so we set their linkage temporarily to internal, run
    //   `globaldce` and then restore them to their original linkage
    // * The rest (QEMU globals) do not need their initializer, so it's removed
    //   and their linkage is set to external. `globaldce` will also remove
    //   them if they are unused
    StringMap<GlobalValue::LinkageTypes> GlobalLinkageBackup;
    for (GlobalVariable &GV : M.globals()) {
      GV.eraseMetadata(LLVMContext::MD_dbg);
      if (GV.isDeclaration() or isSpecialGV(GV))
        continue;

      if (FunctionTags::CSV.isTagOf(&GV)) {
        // Save linkage to be restored after we run globaldce
        GlobalLinkageBackup[GV.getName()] = GV.getLinkage();
        GV.setLinkage(GlobalValue::InternalLinkage);
      } else {
        // Blank out initializers for any non-CSV GlobalVariable
        GV.setInitializer(nullptr);
        GV.setLinkage(GlobalValue::ExternalLinkage);
      }
    }

    // Run always-inline and globaldce.
    // It's important to run `always-inline` now so that when the helper
    // functions are inlined into the main module it can be done in a single
    // pass and not in a fixed-point fashion.
    SimplePassManager PM;
    PM.addPass(llvm::AlwaysInlinerPass());
    PM.addPass(llvm::GlobalDCEPass());
    PM.run(M);

    // Restore the linkage to CSVs after running globaldce
    for (GlobalVariable &GV : M.globals()) {
      if (GlobalLinkageBackup.count(GV.getName()) != 0)
        GV.setLinkage(GlobalLinkageBackup[GV.getName()]);
    }

    // Helper functions, down the pipeline, will be given the `NoInline`
    // attribute, to avoid the module failing verification remove the
    // `AlwaysInline` attribute.
    for (llvm::Function &F : M) {
      if (F.getSection() == InlineHelpersSection)
        F.removeFnAttr(Attribute::AlwaysInline);
    }

    // Tag all globals
    tagQEMUGlobalsAndFunctions(M);

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
