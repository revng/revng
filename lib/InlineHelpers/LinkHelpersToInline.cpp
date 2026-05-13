//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <memory>
#include <set>
#include <string>

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"

#include "revng/InlineHelpers/LinkHelpersToInline.h"
#include "revng/Support/Assert.h"
#include "revng/Support/IRHelpers.h"
#include "revng/Support/ResourceFinder.h"

using namespace llvm;

char LinkHelpersToInlinePass::ID = 0;

using Register = RegisterPass<LinkHelpersToInlinePass>;
static Register X("link-helpers-to-inline",
                  "Link Required Helpers To Inline Pass",
                  true,
                  true);

bool LinkHelpersToInlinePass::runOnModule(llvm::Module &M) {
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

  // Collect the names of revng_inline helpers that lack a body in `M` and
  // therefore need to be cloned from the per-architecture to-inline bitcode.
  SmallVector<StringRef, 32> MissingBodies;
  for (llvm::Function &F : M) {
    if (F.isDeclaration() and F.getSection() == InlineHelpersSection) {
      MissingBodies.push_back(F.getName());
    }
  }

  // Nothing to link — every referenced revng_inline helper already has a
  // body in `M`. Skip the expensive bitcode parse + clone + link in this case.
  if (MissingBodies.empty()) {
    return true;
  }

  // Given the architecture MD, retrieve the correct libtcg module and link
  // it into the main module.
  StringRef ArchName = Strings.begin()->first();
  const std::string LibHelpersName = ("/share/revng/"
                                      "libtcg-helpers-to-inline-"
                                      + ArchName + ".bc")
                                       .str();
  auto OptionalHelpers = revng::ResourceFinder.findFile(LibHelpersName);
  revng_assert(OptionalHelpers.has_value(), "Cannot find libtcg helpers");
  std::unique_ptr<llvm::Module>
    HelpersModule = parseIR(M.getContext(), OptionalHelpers.value());

  std::set<const llvm::Function *> ToClone;
  for (StringRef Name : MissingBodies) {
    llvm::Function *HelperFunction = HelpersModule->getFunction(Name);
    revng_assert(HelperFunction != nullptr
                   and not HelperFunction->isDeclaration(),
                 "revng_inline helper missing body in to-inline.bc; the "
                 "static-policy / slim-down chain produced an "
                 "inconsistent state");
    ToClone.insert(HelperFunction);
  }

  auto ClonedHelpersModule = ::cloneFiltered(*HelpersModule, ToClone);
  linkModules(std::move(ClonedHelpersModule), M);

  return true;
}
