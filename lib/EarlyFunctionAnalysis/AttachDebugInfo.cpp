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

#include "revng/EarlyFunctionAnalysis/AttachDebugInfo.h"
#include "revng/EarlyFunctionAnalysis/ControlFlowGraphCache.h"
#include "revng/Model/FunctionTags.h"
#include "revng/Model/IRHelpers.h"
#include "revng/Ranks/Location.h"
#include "revng/Ranks/Ranks.h"
#include "revng/Support/BasicBlockID.h"
#include "revng/Support/MetaAddress.h"
#include "revng/Support/NewPC.h"

using namespace llvm;

static Logger Log("attach-debug-info");

/// \return the control-flow graph whose entry is \p Owner, if any
///
/// Looking the block up instead would be ambiguous: two functions can share
/// code, so the same block can appear in more than one control-flow graph.
static const efa::ControlFlowGraph *
findOwner(const efa::ControlFlowGraph &FM,
          llvm::ArrayRef<const efa::ControlFlowGraph *> Inlined,
          const MetaAddress &Owner) {
  if (FM.Entry() == Owner)
    return &FM;

  for (const efa::ControlFlowGraph *Candidate : Inlined)
    if (Candidate->Entry() == Owner)
      return Candidate;

  return nullptr;
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
  /// Attach the debug locations to \p F
  ///
  /// \p FM describes \p F itself, while \p Inlined describes the functions
  /// whose body might have been inlined into it. The code coming from those
  /// keeps referring to their own addresses.
  void handleFunction(llvm::Function &F,
                      const efa::ControlFlowGraph &FM,
                      llvm::ArrayRef<const efa::ControlFlowGraph *> Inlined) {
    const efa::ControlFlowGraph *CurrentCFG = &FM;
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
      bool UseFallbackDebugLocation = !isTranslated(BB);

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
        if (std::optional Call = NewPCHelper.getCall(&I)) {
          BasicBlockID Address = blockIDFromNewPC(*Call);

          if (startsBasicBlock(*Call)) {
            const efa::ControlFlowGraph
              *Owner = findOwner(FM, Inlined, ownerFromNewPC(*Call));
            revng_assert(Owner != nullptr,
                         "`newpc` refers to a function that is not part of "
                         "this bundle");

            if (Owner->Blocks().contains(Address)) {
              CurrentCFG = Owner;
              CurrentBB = Address;
              revng_assert(CurrentBB.isValid());

            } else {
              revng_assert(CurrentCFG->Blocks()
                             .at(CurrentBB)
                             .contains(Address));
            }
          }

          revng_assert(Address.inliningIndex() == CurrentBB.inliningIndex());
          CurrentDI = buildDI(CurrentCFG->Entry(), CurrentBB, Address.start());

          if (llvm::Error Error = isDebugLocationInvalid(CurrentDI))
            revng_abort(revng::unwrapError(std::move(Error)).c_str());
        }

        I.setDebugLoc(CurrentDI);
      }
    }
  }
};

/// Give \p LLVMFunction, described by \p FM, a debug location per instruction
///
/// \p Inlined describes the functions whose body might have been inlined into
/// it.
static void attachTo(DIBuilder &DIB,
                     DIFile *File,
                     llvm::Function &LLVMFunction,
                     const efa::ControlFlowGraph &FM,
                     llvm::ArrayRef<const efa::ControlFlowGraph *> Inlined) {
  // Skip functions with debug-info.
  if (LLVMFunction.getSubprogram())
    return;

  // Skip declarations
  revng_assert(not LLVMFunction.isDeclaration());

  revng_log(Log,
            "Metadata for Function " << LLVMFunction.getName() << ":"
                                     << FM.Entry().toString());

  DebugInformationBuilder Builder(DIB,
                                  LLVMFunction.getContext(),
                                  File,
                                  LLVMFunction.getName());
  Builder.handleFunction(LLVMFunction, FM, Inlined);
}

namespace revng::pypeline::piperuns {

void AttachDebugInfo::runOnLLVMFunction(const model::Function &Function,
                                        llvm::Function &LLVMFunction) {
  llvm::Module &Module = *LLVMFunction.getParent();

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

  const auto &Bundle = *CFG.getElement(ObjectID(Function.Entry()));

  llvm::SmallVector<const efa::ControlFlowGraph *> Inlined;
  for (const efa::ControlFlowGraph &Callee : Bundle.AlwaysInlineFunctions())
    Inlined.push_back(&Callee);

  attachTo(DIB, CU->getFile(), LLVMFunction, Bundle.MainFunction(), Inlined);

  // Until the inlining pipe runs, the module also carries the body of the
  // functions to inline into this one: they need locations of their own, or
  // the code resulting from inlining them would have none.
  for (llvm::Function &F : Module.functions()) {
    if (&F == &LLVMFunction or F.isDeclaration())
      continue;

    if (not FunctionTags::Isolated.isTagOf(&F))
      continue;

    MetaAddress Entry = getMetaAddressOfIsolatedFunction(F);
    attachTo(DIB,
             CU->getFile(),
             F,
             Bundle.AlwaysInlineFunctions().at(Entry),
             {});
  }
}

} // namespace revng::pypeline::piperuns
