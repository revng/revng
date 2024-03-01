/// \file AttachDebugInfo.cpp
///
/// A simple pass to attach debug metadata.
///
/// This pass visits each function into reverse post order and, each time it
/// finds a call to newpc, updates the "current location". While doing the visit
/// we attach the "current location" to each instruction we meet.
///
/// The debug location we attach refers to a program specific to that program
/// counter which has been virtually inlined into another subprogram that
/// represents the current function.
///
/// Note however, that subprogram representing the current function is not
/// attached the function itself, since otherwise that would trigger a rather
/// strict debug info verification logic, which we currently do not handle.
/// Specifically, if a function as debug information, then all the inlinable
/// call sites targeting it need to have debug information too.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"

#include "revng/EarlyFunctionAnalysis/AttachDebugInfo.h"
#include "revng/Pipeline/Location.h"
#include "revng/Pipes/Ranks.h"
#include "revng/Support/FunctionTags.h"

using namespace llvm;

char AttachDebugInfo::ID = 0;

using Register = RegisterPass<AttachDebugInfo>;
static Register X("attach-debug-info", "Attach debug info metadata.");
static Logger<> Log("attach-debug-info");

static bool isTrue(const llvm::Value *V) {
  return getLimitedValue(V) != 0;
}

static void handleFunction(DIBuilder &DIB,
                           llvm::Function &F,
                           DISubprogram *TheSubprogram,
                           efa::FunctionMetadata &FM,
                           GeneratedCodeBasicInfo &GCBI) {
  namespace ranks = revng::ranks;

  LLVMContext &Context = F.getParent()->getContext();
  DILocation *CurrentDebugLocation = nullptr;
  BasicBlockID LastJumpTarget;
  for (auto *BB : ReversePostOrderTraversal(&F)) {
    if (not GCBI.isTranslated(BB))
      continue;

    for (auto &I : *BB) {
      if (auto *Call = getCallTo(&I, "newpc")) {
        BasicBlockID Address = blockIDFromNewPC(Call);

        if (isTrue(Call->getArgOperand(NewPCArguments::IsJumpTarget))) {
          const auto &CFG = FM.ControlFlowGraph();
          if (CFG.contains(Address)) {
            LastJumpTarget = Address;
          } else {
            revng_assert(CFG.at(LastJumpTarget).contains(Address));
          }
        }

        revng_assert(LastJumpTarget.isValid());
        revng_assert(Address.inliningIndex() == LastJumpTarget.inliningIndex());

        auto SPFlags = DISubprogram::toSPFlags(false, /* isLocalToUnit */
                                               true, /* isDefinition*/
                                               false /* isOptimized */);
        auto Type = DIB.createSubroutineType(DIB.getOrCreateTypeArray({}));

        // Let's make the debug location that points back to the binary.
        std::string NewDebugLocation = serializedLocation(ranks::Instruction,
                                                          FM.Entry(),
                                                          LastJumpTarget,
                                                          Address.start());
        auto Subprogram = DIB.createFunction(TheSubprogram->getFile(), // Scope
                                             NewDebugLocation, // Name
                                             StringRef(), // LinkageName
                                             TheSubprogram->getFile(), // File
                                             1, // LineNo
                                             Type, // Ty (subroutine type)
                                             1, // ScopeLine
                                             DINode::FlagPrototyped, // Flags
                                             SPFlags);
        DIB.finalizeSubprogram(Subprogram);

        auto InlineLocationForMetaAddress = DILocation::get(Context,
                                                            0,
                                                            0,
                                                            TheSubprogram,
                                                            nullptr);

        // Represent debug info for all the isolated functions as if they were
        // inlined in the root.
        auto InlineLoc = DILocation::get(Context,
                                         0,
                                         0,
                                         Subprogram,
                                         InlineLocationForMetaAddress);

        CurrentDebugLocation = InlineLoc;
      }

      I.setDebugLoc(CurrentDebugLocation);
    }
  }
}

bool AttachDebugInfo::runOnModule(llvm::Module &M) {
  DIBuilder DIB(M);
  FunctionMetadataCache *Cache = &getAnalysis<FunctionMetadataCachePass>()
                                    .get();
  auto &GCBI = getAnalysis<GeneratedCodeBasicInfoWrapperPass>().getGCBI();

  // This will be used for attaching the !dbg to instructions.
  // TODO: Document how are we going to abuse DILocation fields.
  DIFile *File = DIB.createFile(M.getSourceFileName(), "./");
  // Also add dummy CU.
  DICompileUnit *CU = DIB.createCompileUnit(dwarf::DW_LANG_C,
                                            File,
                                            "revng", // Producer
                                            true, // isOptimized
                                            "", // Flags
                                            0 // RV
  );

  for (auto &F : M) {
    // Skip non-isolated functions (e.g. helpers from QEMU).
    if (not FunctionTags::Isolated.isTagOf(&F))
      continue;

    // Skip functions with debug-info.
    if (F.getSubprogram())
      continue;

    // Skip declarations
    if (F.isDeclaration())
      continue;

    auto FM = Cache->getFunctionMetadata(&F);
    revng_log(Log,
              "Metadata for Function " << F.getName() << ":"
                                       << FM.Entry().toString());

    // Create debug info for the function.
    auto SPFlags = DISubprogram::toSPFlags(false, // isLocalToUnit
                                           true, // isDefinition
                                           false // isOptimized
    );
    auto SPType = DIB.createSubroutineType(DIB.getOrCreateTypeArray({}));

    DISubprogram
      *TheSubprogram = DIB.createFunction(CU->getFile(), // Scope
                                          F.getName(), // Name
                                          StringRef(), // LinkageName
                                          CU->getFile(), // File
                                          1, // LineNo
                                          SPType, // Ty (subroutine type)
                                          1, // ScopeLine
                                          DINode::FlagPrototyped, // Flags
                                          SPFlags);
    DIB.finalizeSubprogram(TheSubprogram);

    handleFunction(DIB, F, TheSubprogram, FM, GCBI);
  }

  return true;
}
