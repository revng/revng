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
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"

#include "revng/BasicAnalyses/GeneratedCodeBasicInfo.h"
#include "revng/EarlyFunctionAnalysis/AttachDebugInfo.h"
#include "revng/EarlyFunctionAnalysis/ControlFlowGraphCache.h"
#include "revng/Model/FunctionTags.h"
#include "revng/Ranks/Location.h"
#include "revng/Ranks/Ranks.h"
#include "revng/Support/BasicBlockID.h"
#include "revng/Support/MetaAddress.h"

using namespace llvm;

static Logger Log("attach-debug-info");

static bool isTrue(const llvm::Value *V) {
  return getLimitedValue(V) != 0;
}

class DebugInformationBuilder {
private:
  DIBuilder &DIB;
  LLVMContext &Context;
  DIFile *File = nullptr;

  DISubprogram::DISPFlags SubprogramFlags;
  DISubroutineType *SubprogramType = nullptr;
  DISubprogram *FunctionSubprogram = nullptr;

public:
  DebugInformationBuilder(DIBuilder &DIB,
                          LLVMContext &Context,
                          DIFile *File,
                          llvm::StringRef Name) :
    DIB(DIB), Context(Context), File(File) {
    SubprogramFlags = DISubprogram::toSPFlags(false, /* isLocalToUnit */
                                              true, /* isDefinition*/
                                              false /* isOptimized */);
    SubprogramType = DIB.createSubroutineType(DIB.getOrCreateTypeArray({}));
    FunctionSubprogram = makeSubprogram(Name);
  }

private:
  DISubprogram *makeSubprogram(llvm::StringRef Name) {
    DISubprogram
      *Result = DIB.createFunction(/* Scope = */ File,
                                   /* Name = */ Name,
                                   /* LinkageName = */ StringRef(),
                                   /* File = */ File,
                                   /* LineNo = */ 1,
                                   /* Ty = */ SubprogramType,
                                   /* ScopeLine = */ 1,
                                   /* DIFlags = */ DINode::FlagPrototyped,
                                   /* SPFlags = */ SubprogramFlags);
    DIB.finalizeSubprogram(Result);
    return Result;
  }

  DILocation *buildDI(MetaAddress FunctionAddress,
                      BasicBlockID BasicBlockAddress,
                      MetaAddress InstructionAddress) {
    std::string NewDebugLocation = locationString(revng::ranks::Instruction,
                                                  FunctionAddress,
                                                  BasicBlockAddress,
                                                  InstructionAddress);
    DISubprogram *Subprogram = makeSubprogram(NewDebugLocation);

    auto InlineLocationForMetaAddress = DILocation::get(Context,
                                                        0,
                                                        0,
                                                        Subprogram,
                                                        nullptr);

    // Represent debug info for all the isolated functions as if they were
    // inlined in the root.
    return DILocation::get(Context,
                           0,
                           0,
                           Subprogram,
                           InlineLocationForMetaAddress);
  }

public:
  void handleFunction(llvm::Function &F,
                      const efa::ControlFlowGraph &FM,
                      GeneratedCodeBasicInfo &GCBI) {
    BasicBlockID CurrentBB = BasicBlockID(FM.Entry());
    DILocation *DefaultDI = buildDI(FM.Entry(), CurrentBB, FM.Entry());
    DILocation *CurrentDI = DefaultDI;

    for (auto *BB : ReversePostOrderTraversal(&F)) {
      // There are two options of how we can handle basic block addresses:
      //
      // - in the general case, addresses provided by the `newpc` helper calls
      //   are used to fill in the metadata of all the instructions in-between
      //   (note that the reverse post order traversal is important).
      //
      // - when that is not possible (which most likely means that the block
      //   in question is artificial), the fallback is to use the most basic
      //   possible location:
      //   ```
      //   /instruction/<function-entry>/<function-entry>/<function-entry>
      //   ```
      //
      // The following flag decides which approach is used for this basic block:
      bool UseFallbackDebugLocation = !GCBI.isTranslated(BB);

      if (getType(BB) == BlockType::IndirectBranchDispatcherHelperBlock) {
        // These helper blocks are introduced to handle indirect jumps (for
        // example, `jmp rax`). But, since CFG around them is reasonable AND
        // because we're traversing them in the reverse post order, we can let
        // normal `newpc`-based address setter do its job for them too.
        UseFallbackDebugLocation = false;
      }

      // TODO: keep a close eye on this, especially if we ever add more basic
      //       block types, as using the default location is pretty much
      //       the worst option available.
      if (UseFallbackDebugLocation) {
        for (auto &I : *BB)
          I.setDebugLoc(DefaultDI);

        continue;
      }

      for (auto &I : *BB) {
        if (auto *Call = getCallTo(&I, "newpc")) {
          BasicBlockID Address = blockIDFromNewPC(Call);

          if (isTrue(Call->getArgOperand(NewPCArguments::IsJumpTarget))) {
            const auto &CFG = FM.Blocks();
            if (CFG.contains(Address)) {
              CurrentBB = Address;
              revng_assert(CurrentBB.isValid());

            } else {
              revng_assert(CFG.at(CurrentBB).contains(Address));
            }
          }

          revng_assert(Address.inliningIndex() == CurrentBB.inliningIndex());
          CurrentDI = buildDI(FM.Entry(), CurrentBB, Address.start());

          if (llvm::Error Error = isDebugLocationInvalid(CurrentDI))
            revng_abort(revng::unwrapError(std::move(Error)).c_str());
        }

        I.setDebugLoc(CurrentDI);
      }
    }
  }
};

namespace revng::pypeline::piperuns {

void AttachDebugInfo::runOnLLVMFunction(const model::Function &Function,
                                        llvm::Function &LLVMFunction) {
  llvm::Module &Module = *LLVMFunction.getParent();
  GeneratedCodeBasicInfo GCBI(Binary, Module);

  DIBuilder DIB(Module);
  // This will be used for attaching the !dbg to instructions.
  // TODO: Document how are we going to abuse DILocation fields.
  DIFile *File = DIB.createFile(Module.getSourceFileName(), "./");
  // Also add dummy CU.
  DICompileUnit *CU = DIB.createCompileUnit(dwarf::DW_LANG_C,
                                            File,
                                            "revng", // Producer
                                            true, // isOptimized
                                            "", // Flags
                                            0 // RV
  );

  // Skip functions with debug-info.
  if (LLVMFunction.getSubprogram())
    return;

  // Skip declarations
  revng_assert(not LLVMFunction.isDeclaration());

  ObjectID Object(Function.Entry());
  const efa::ControlFlowGraph &FM = CFG.getElement(Object)->MainFunction();
  revng_log(Log,
            "Metadata for Function " << LLVMFunction.getName() << ":"
                                     << FM.Entry().toString());

  LLVMContext &Context = LLVMFunction.getParent()->getContext();
  DebugInformationBuilder Builder(DIB,
                                  Context,
                                  CU->getFile(),
                                  LLVMFunction.getName());
  Builder.handleFunction(LLVMFunction, FM, GCBI);
}

} // namespace revng::pypeline::piperuns
