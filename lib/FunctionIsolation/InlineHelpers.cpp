//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <iterator>
#include <system_error>

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include "revng/ADT/Queue.h"
#include "revng/BasicAnalyses/GeneratedCodeBasicInfo.h"
#include "revng/FunctionIsolation/InlineHelpers.h"
#include "revng/Support/CommandLine.h"
#include "revng/Support/Debug.h"
#include "revng/Support/IRHelpers.h"
#include "revng/Support/OpaqueFunctionsPool.h"
#include "revng/Support/ResourceFinder.h"

using namespace llvm;

static Logger Log("inline-helpers");

static cl::opt<std::string> DumpBeforeDir("inline-helpers-dump-before-dir",
                                          cl::desc("Directory in which to "
                                                   "write one `<fn>.ll` file "
                                                   "per Isolated function, "
                                                   "containing the body of "
                                                   "the function right "
                                                   "before inline-helpers "
                                                   "touches it. Created if "
                                                   "absent."),
                                          cl::init(""),
                                          cl::cat(MainCategory));

static cl::opt<std::string> DumpAfterDir("inline-helpers-dump-after-dir",
                                         cl::desc("Directory in which to "
                                                  "write one `<fn>.ll` file "
                                                  "per Isolated function, "
                                                  "containing the body of "
                                                  "the function right after "
                                                  "inline-helpers has "
                                                  "finished with it. "
                                                  "Created if absent."),
                                         cl::init(""),
                                         cl::cat(MainCategory));

// Dump one `<F.name>.ll` file to `Dir`, containing just the body of `F`.
// Called once per Isolated function per pass invocation, from
// `InlineHelpers::run(F)`; the user then gets exactly one file per
// Isolated function, named by it.
static void dumpFunctionToDir(const Function &F, StringRef Dir) {
  std::error_code EC = sys::fs::create_directories(Dir);
  revng_assert(not EC,
               ("inline-helpers: cannot create dump directory: " + EC.message())
                 .c_str());

  SmallString<256> Path(Dir);
  sys::path::append(Path, (F.getName() + ".ll").str());

  raw_fd_ostream OS(Path, EC, sys::fs::OF_Text);
  revng_assert(not EC,
               ("inline-helpers: cannot open dump file: " + EC.message())
                 .c_str());
  F.print(OS, nullptr);
}

char InlineHelpersPass::ID = 0;

using Register = RegisterPass<InlineHelpersPass>;
static Register X("inline-helpers", "Inline Helpers Pass", true, true);

class InlineHelpers {
public:
  InlineHelpers() = default;

  void run(Function *F);

private:
  void doInline(CallInst *Call) const;
  bool doInline(Function *F) const;
  CallInst *getCallToInline(Instruction *I) const;
  bool shouldInline(const CallInst *Call) const;
};

bool InlineHelpers::shouldInline(const CallInst *Call) const {
  const Function *Callee = getCalledFunction(Call);
  if (Callee == nullptr)
    return false;

  if (Callee->getSection() != InlineHelpersSection)
    return false;

  InliningPolicy P = deserializeInliningPolicy(*Callee);
  const BitVector &Critical = P.CriticalArguments;

  if (Critical.none()) {
    revng_log(Log, "inline " << Callee->getName() << ": no critical arguments");
    return true;
  }

  revng_log(Log, "Analyzing arguments of " << Callee->getName());
  LoggerIndent Indent(Log);

  bool AllConstant = true;
  for (int Index = Critical.find_first(); Index >= 0;
       Index = Critical.find_next(Index)) {
    const bool IsConst = isa<Constant>(Call->getArgOperand(Index));
    if (not IsConst)
      AllConstant = false;
    revng_log(Log,
              "argument " << Index << ": "
                          << (IsConst ? "const" : "NON-CONST"));
  }

  revng_log(Log, (AllConstant ? "inline" : "skip"));
  return AllConstant;
}

CallInst *InlineHelpers::getCallToInline(Instruction *I) const {
  if (auto *Call = dyn_cast<CallInst>(I)) {
    if (shouldInline(Call)) {
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

  if (F->isDeclaration())
    return;

  revng_log(Log, "Analyzing " << F->getName());

  // Dump the IR of the function before the inlining takes place
  if (not DumpBeforeDir.empty())
    dumpFunctionToDir(*F, DumpBeforeDir);

  // `slim-down-helpers-module` already fully inlines helpers, but we might
  // still have wrappers created by `promote-csvs` at runtime that we need to
  // traverse and inline as well.
  while (doInline(F)) {
  }
  dropDebugOrPseudoInst(F);

  // Dump the IR of the function after inlining has taken place
  if (not DumpAfterDir.empty())
    dumpFunctionToDir(*F, DumpAfterDir);
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

  // Collect the names of every `revng_inline` helper referenced in `M`;
  // they all lack a body in `M` and need to be cloned from the
  // per-architecture to-inline bitcode.
  SmallVector<StringRef, 32> HelpersToInline;
  for (llvm::Function &F : M) {
    if (F.isDeclaration() and F.getSection() == InlineHelpersSection)
      HelpersToInline.push_back(F.getName());
  }

  // Skip the expensive bitcode parse + clone + link if there's nothing to do.
  if (HelpersToInline.empty())
    return;

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
  for (StringRef Name : HelpersToInline) {
    llvm::Function *HelperFunction = HelpersModule.getFunction(Name);
    revng_assert(HelperFunction != nullptr
                   and not HelperFunction->isDeclaration(),
                 "revng_inline helper missing body in to-inline.bc");
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
    InlineHelpers IH;
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
