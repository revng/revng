//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/Path.h"

#include "revng/ADT/Queue.h"
#include "revng/Support/Debug.h"
#include "revng/Support/IRHelpers.h"

using namespace llvm;

static Logger Log("mark-inline-helpers-up-to");

/// Cap on the body size of a single helper that we are willing to mark as
/// `revng_inline`. Above this threshold the helper is left alone — inlining
/// it would explode the size of every caller, with no offsetting benefit in
/// the lifted IR.
static constexpr unsigned MaxBodyInstructions = 100;

/// Directory-suffix list. A function is treated as a "leaf" of the runtime
/// library we want to inline up to whenever its `DISubprogram`'s file path,
/// after stripping the file name, ends in one of these suffixes.
static constexpr StringLiteral RuntimeLibraryDirectories[] = { "fpu" };

/// True if the directory containing `F`'s definition ends in one of
/// `DirectorySuffixes`.
static bool isDefinedUnderDirectory(const Function &F,
                                    ArrayRef<StringLiteral> DirectorySuffixes) {
  const DISubprogram *Subprogram = F.getSubprogram();
  if (Subprogram == nullptr)
    return false;
  const DIFile *File = Subprogram->getFile();
  if (File == nullptr)
    return false;

  // The match operates on the parsed directory path: the file basename is
  // stripped, then we check that the resulting directory path's *last
  // component* equals one of the supplied suffixes.
  SmallString<128> Path(File->getFilename());
  StringRef Directory = sys::path::parent_path(Path);
  StringRef LastComponent = sys::path::filename(Directory);

  for (StringRef Suffix : DirectorySuffixes)
    if (LastComponent == Suffix)
      return true;
  return false;
}

static unsigned instructionCount(const Function &F) {
  unsigned Count = 0;
  for (const BasicBlock &BB : F)
    Count += BB.size();
  return Count;
}

/// Marks every QEMU helper that transitively reaches a function defined in
/// one of the configured runtime-library directories (currently `fpu/` only)
/// as `revng_inline`. The intent is to make the lift pipeline surface
/// "leaf-level" runtime-library calls (e.g. softfloat ops) directly in the
/// lifted IR by inlining away every wrapper helper that sits between QEMU's
/// `helper_*` boundary and the leaf.
///
/// The pass bails out if a candidate is found, whose body exceeds
/// `MaxBodyInstructions`.
class MarkInlineHelpersUpTo : public ModulePass {
public:
  static char ID;

  MarkInlineHelpersUpTo() : ModulePass(ID) {}

  bool runOnModule(Module &M) override {

    // Identify every function defined under one of the configured runtime-
    // library directories — these are the "leaves" we want to inline up to.
    SmallSet<const Function *, 4> Leaves;
    {
      revng_log(Log, "Collecting runtime-library leaf functions:");
      LoggerIndent Indent(Log);
      for (Function &F : M) {
        if (F.isDeclaration())
          continue;
        if (isDefinedUnderDirectory(F, RuntimeLibraryDirectories)) {
          Leaves.insert(&F);
          revng_log(Log, F.getName());
        }
      }
    }

    if (Leaves.empty()) {
      revng_log(Log, "No runtime-library leaf functions found, nothing to do.");
      return false;
    }

    // Build the reverse call graph (callee -> callers) so we can do the
    // transitive backward walk in a single pass over the module.
    DenseMap<const Function *, SmallVector<const Function *>> Callers;
    {
      revng_log(Log, "Building reverse call graph:");
      LoggerIndent Indent(Log);
      for (const Function &F : M) {
        if (F.isDeclaration())
          continue;
        for (const BasicBlock &BB : F) {
          for (const Instruction &I : BB) {
            const auto *Call = dyn_cast<CallInst>(&I);
            if (Call == nullptr)
              continue;
            const Function *Callee = getCalledFunction(Call);
            if (Callee == nullptr or Callee->isIntrinsic())
              continue;
            Callers[Callee].push_back(&F);
          }
        }
      }
    }

    // Transitive-callers backward walk from each leaf.
    OnceQueue<const Function *> Worklist;
    for (const Function *Leaf : Leaves)
      Worklist.insert(Leaf);

    while (not Worklist.empty()) {
      const Function *Callee = Worklist.pop();
      auto It = Callers.find(Callee);
      if (It == Callers.end())
        continue;
      for (const Function *Caller : It->second)
        Worklist.insert(Caller);
    }
    std::set<const Function *> Reachable = Worklist.visited();

    // Tag every transitive caller that is *not* itself a leaf and whose body is
    // under the size cap.
    revng_log(Log, "Tagging transitive callers as revng_inline:");
    LoggerIndent Indent(Log);
    bool Changed = false;
    for (Function &F : M) {
      if (Leaves.contains(&F) or not Reachable.contains(&F))
        continue;

      unsigned Size = instructionCount(F);
      if (Size > MaxBodyInstructions) {
        revng_log(Log,
                  "skip " << F.getName() << " body=" << Size
                          << " > cap=" << MaxBodyInstructions);
        continue;
      }

      F.setSection(InlineHelpersSection);
      Changed = true;
      revng_log(Log, "tag " << F.getName() << " body=" << Size);
    }

    return Changed;
  }
};

char MarkInlineHelpersUpTo::ID = 0;
using Register = RegisterPass<MarkInlineHelpersUpTo>;
static Register X("mark-inline-helpers-up-to",
                  "Mark every helper transitively reaching a configured "
                  "runtime-library "
                  "directory (currently `fpu/`) as `revng_inline`",
                  true,
                  true);
