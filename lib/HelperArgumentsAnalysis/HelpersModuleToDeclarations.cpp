//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/Module.h"
#include "llvm/Pass.h"

#include "revng/Model/FunctionTags.h"

#include "PostProcessingHelpers.h"

using namespace llvm;

class HelpersModuleToDeclarations : public llvm::ModulePass {
public:
  static char ID;

public:
  HelpersModuleToDeclarations() : llvm::ModulePass(ID) {}

  bool runOnModule(llvm::Module &M) override {
    for (Function &F : M) {
      // Only the full module requires debugging information. Remove the !dbg
      // tag and the body of the functions. Later `-globaldce` and
      // `-strip-dead-debug-info` will take care of dropping all the unused
      // data.
      F.eraseMetadata(LLVMContext::MD_dbg);
      deleteOnlyBody(F);
    }

    // The global variables are not needed in this module. Drop their debug info
    // and their initializer. GlobalDCE will be run later to remove them.
    // Note that we preserve CSVs and some other special variables. In
    // particular, we need CSVs to avoid having new CSVs popping up
    // post-inlining, which is problematic since sometimes we handle
    // non-existing CSVs in a special way.
    for (GlobalVariable &GV : M.globals()) {
      GV.eraseMetadata(LLVMContext::MD_dbg);
      if (not isSpecialGV(GV) and not FunctionTags::CSV.isTagOf(&GV))
        GV.setInitializer(nullptr);
    }

    // Assume the module M already contains the functions.
    // Collect all functions whose name starts with "helper_" to prevent DCE
    llvm::DenseSet<Function *> Helpers;
    for (Function &F : M)
      if (isHelper(F))
        Helpers.insert(&F);

    PointerType *PointerType = PointerType::get(M.getContext(), 0);

    // Create constant array initializer with function pointers to the helpers
    // functions. Since all the helper functions are declarations they might be
    // removed by a globaldce, to prevent that store their pointer in a global
    // variable with external linkage.
    std::vector<Constant *> FunctionPointers;
    for (Function *F : Helpers)
      FunctionPointers.push_back(ConstantExpr::getBitCast(F, PointerType));

    // Create array type: [N x <function pointer type>]
    ArrayType *FunctionArray = ArrayType::get(PointerType,
                                              FunctionPointers.size());
    Constant *Initializer = ConstantArray::get(FunctionArray, FunctionPointers);

    new GlobalVariable(M,
                       FunctionArray,
                       true,
                       GlobalValue::ExternalLinkage,
                       Initializer,
                       HelpersListName);

    // Tag all globals
    tagQEMUGlobalsAndFunctions(M);

    return true;
  }
};

char HelpersModuleToDeclarations::ID = 0;
using Register = RegisterPass<HelpersModuleToDeclarations>;
static Register X("helpers-module-to-declarations",
                  "Purge all helper functions but keep their "
                  "declarations alive in a array of pointers.",
                  true,
                  true);
