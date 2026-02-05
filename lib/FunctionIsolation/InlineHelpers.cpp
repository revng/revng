//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <iterator>

#include "llvm/ADT/SCCIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include "revng/ADT/Queue.h"
#include "revng/BasicAnalyses/GeneratedCodeBasicInfo.h"
#include "revng/FunctionIsolation/InlineHelpers.h"
#include "revng/Support/IRHelpers.h"
#include "revng/Support/OpaqueFunctionsPool.h"
#include "revng/Support/ResourceFinder.h"

using namespace llvm;

char InlineHelpersPass::ID = 0;

using Register = RegisterPass<InlineHelpersPass>;
static Register X("inline-helpers", "Inline Helpers Pass", true, true);

class InlineHelpers {
private:
  LLVMContext &C;
  SmallDenseSet<Function *, 8> Recursive;

public:
  InlineHelpers(Module &M) : C(M.getContext()) {
    using namespace llvm;
    llvm::CallGraph CG(M);
    for (auto It = scc_begin(&CG), End = scc_end(&CG); It != End; ++It)
      if (It.hasCycle())
        for (auto &Node : *It)
          if (Function *F = Node->getFunction())
            Recursive.insert(F);
  }

  void run(Function *F);

private:
  void doInline(CallInst *Call) const;
  bool doInline(Function *F) const;
  CallInst *getCallToInline(Instruction *I) const;
  bool shouldInline(Function *F) const;
};

bool InlineHelpers::shouldInline(Function *F) const {
  if (F == nullptr)
    return false;

  if (Recursive.count(F) != 0)
    return false;

  return F->getSection() == InlineHelpersSection;
}

CallInst *InlineHelpers::getCallToInline(Instruction *I) const {
  if (auto *Call = dyn_cast<CallInst>(I)) {
    if (shouldInline(getCalledFunction(Call))) {
      return Call;
    }
  }

  return nullptr;
}

void InlineHelpers::doInline(CallInst *Call) const {
  InlineFunctionInfo IFI;
  auto Result = InlineFunction(*Call, IFI, false, nullptr, false);
  revng_assert(Result.isSuccess(), Result.getFailureReason());
}

bool InlineHelpers::doInline(Function *F) const {
  SmallVector<CallInst *, 8> ToInline;

  for (BasicBlock &BB : *F)
    for (Instruction &I : BB)
      if (auto *Call = getCallToInline(&I))
        ToInline.push_back(Call);

  for (CallInst *Call : ToInline)
    doInline(Call);

  return ToInline.size() > 0;
}

static void dropDebugOrPseudoInst(Function *F) {
  SmallVector<Instruction *, 16> ToErase;
  for (BasicBlock &BB : *F) {
    for (Instruction &I : BB) {
      if (I.isDebugOrPseudoInst()) {
        ToErase.push_back(&I);
      }
    }
  }

  for (Instruction *I : ToErase) {
    I->eraseFromParent();
  }
}

void InlineHelpers::run(Function *F) {
  // Since all the helpers have been already inlined the `doInline` function can
  // be called only once to inline all the helper functions.
  doInline(F);
  dropDebugOrPseudoInst(F);
}

void InlineHelpersPass::linkRequiredHelpers(llvm::Module &M) {
  NamedMDNode *Node = M.getNamedMetadata(QemuArchitectureMD);
  revng_assert(Node != nullptr);
  revng_assert(Node->getNumOperands() > 0);

  // Multiple linkings could have lead to this node having more than one
  // operand, if so check that they are all the same.
  llvm::StringSet Strings;
  for (size_t I = 0; I < Node->getNumOperands(); I++) {
    revng_assert(Node->getOperand(0)->getNumOperands() == 1);
    const MDOperand &StringNode = Node->getOperand(0)->getOperand(0);
    Strings.insert(cast<MDString>(StringNode)->getString());
  }

  revng_assert(Strings.size() == 1,
               "QEMU helpers of multiple architectures were linked inside the "
               "module");

  // Given the architecture MD, retrieve the correct libtcg module (if not
  // already in the `HelpersModules` map) and link it into the main module
  StringRef ArchName = Strings.begin()->first();
  if (HelpersModules.count(ArchName) == 0) {
    const std::string LibHelpersName = ("/share/revng/"
                                        "libtcg-helpers-to-inline-"
                                        + ArchName + ".bc")
                                         .str();
    auto OptionalHelpers = revng::ResourceFinder.findFile(LibHelpersName);
    revng_assert(OptionalHelpers.has_value(), "Cannot find libtcg helpers");
    HelpersModules[ArchName] = parseIR(M.getContext(), OptionalHelpers.value());
  }

  llvm::Module &HelpersModule = *HelpersModules[ArchName];

  std::set<const llvm::Function *> ToClone;
  for (llvm::Function &F : M) {
    if (not F.isDeclaration() or F.getSection() != InlineHelpersSection)
      continue;

    llvm::Function *HelperFunction = HelpersModule.getFunction(F.getName());
    revng_assert(HelperFunction != nullptr);
    ToClone.insert(HelperFunction);
  }

  auto ClonedHelpersModule = ::cloneFiltered(HelpersModule, ToClone);
  linkModules(std::move(ClonedHelpersModule), M);
}

bool InlineHelpersPass::runOnModule(llvm::Module &M) {
  linkRequiredHelpers(M);

  SmallVector<Function *, 32> Isolated;
  for (Function &F : M)
    if (FunctionTags::Isolated.isTagOf(&F))
      Isolated.push_back(&F);

  llvm::Task T(Isolated.size(), "Inline helpers");
  for (Function *F : Isolated) {
    T.advance(F->getName());
    InlineHelpers IH(*F->getParent());
    IH.run(F);
  }

  // Since we're done inlining helpers they are no longer needed, delete their
  // body
  for (Function &F : M) {
    if (F.getSection() == InlineHelpersSection)
      deleteOnlyBody(F);
  }

  return true;
}
