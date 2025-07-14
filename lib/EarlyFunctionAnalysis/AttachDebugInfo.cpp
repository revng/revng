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
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"

#include "revng/BasicAnalyses/GeneratedCodeBasicInfo.h"
#include "revng/EarlyFunctionAnalysis/ControlFlowGraphCache.h"
#include "revng/Model/LoadModelPass.h"
#include "revng/Pipeline/Location.h"
#include "revng/Pipeline/RegisterPipe.h"
#include "revng/Pipes/Kinds.h"
#include "revng/Pipes/ModelGlobal.h"
#include "revng/Pipes/Ranks.h"
#include "revng/Support/BasicBlockID.h"
#include "revng/Support/FunctionTags.h"
#include "revng/Support/MetaAddress.h"

using namespace llvm;

static Logger<> Log("attach-debug-info");

class AttachDebugInfo : public pipeline::FunctionPassImpl {
private:
  llvm::Module &M;
  DIBuilder DIB;
  DICompileUnit *CU = nullptr;

public:
  AttachDebugInfo(llvm::ModulePass &Pass,
                  const model::Binary &Binary,
                  llvm::Module &M) :
    pipeline::FunctionPassImpl(Pass), M(M), DIB(M) {}

  static void getAnalysisUsage(llvm::AnalysisUsage &AU) {
    AU.setPreservesAll();
    AU.addRequired<ControlFlowGraphCachePass>();
    AU.addRequired<GeneratedCodeBasicInfoWrapperPass>();
  }

  bool prologue() final;

  bool runOnFunction(const model::Function &ModelFunction,
                     llvm::Function &Function) final;

  bool epilogue() final { return false; }
};

template<>
char pipeline::FunctionPass<AttachDebugInfo>::ID = 0;

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
                      efa::ControlFlowGraph &FM,
                      GeneratedCodeBasicInfo &GCBI) {
    BasicBlockID CurrentBB = BasicBlockID(FM.Entry());
    DILocation *DefaultDI = buildDI(FM.Entry(), CurrentBB, FM.Entry());
    DILocation *CurrentDI = DefaultDI;

    for (auto *BB : ReversePostOrderTraversal(&F)) {
      bool CanBeHandledNormally = GCBI.isTranslated(BB);

      if (getType(BB) == BlockType::IndirectBranchDispatcherHelperBlock) {
        // These helper blocks are introduced to handle indirect jumps (for
        // example, `jmp rax`). But, since CFG around them is reasonable AND
        // because we're traversing them in the reverse post order, we can let
        // normal `newpc`-based address setter do its job for them too.
        CanBeHandledNormally = true;
      }

      // TODO: keep a close eye on this, especially if we ever add more basic
      //       block types, as using the default location is pretty much
      //       the worst option available.
      if (not CanBeHandledNormally) {
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
        }

        I.setDebugLoc(CurrentDI);
      }
    }
  }
};

bool AttachDebugInfo::prologue() {
  // This will be used for attaching the !dbg to instructions.
  // TODO: Document how are we going to abuse DILocation fields.
  DIFile *File = DIB.createFile(M.getSourceFileName(), "./");
  // Also add dummy CU.
  CU = DIB.createCompileUnit(dwarf::DW_LANG_C,
                             File,
                             "revng", // Producer
                             true, // isOptimized
                             "", // Flags
                             0 // RV
  );

  return true;
}

bool AttachDebugInfo::runOnFunction(const model::Function &ModelFunction,
                                    llvm::Function &F) {
  ControlFlowGraphCache *Cache = &getAnalysis<ControlFlowGraphCachePass>()
                                    .get();
  auto &GCBI = getAnalysis<GeneratedCodeBasicInfoWrapperPass>().getGCBI();

  // Skip functions with debug-info.
  if (F.getSubprogram())
    return true;

  // Skip declarations
  revng_assert(not F.isDeclaration());

  auto FM = Cache->getControlFlowGraph(&F);
  revng_log(Log,
            "Metadata for Function " << F.getName() << ":"
                                     << FM.Entry().toString());

  LLVMContext &Context = F.getParent()->getContext();
  DebugInformationBuilder Builder(DIB, Context, CU->getFile(), F.getName());
  Builder.handleFunction(F, FM, GCBI);

  return true;
}

// Note: unfortunately, due to the presence of kinds, we need two distinct pipes

struct AttachDebugInfoToIsolatedPipe {
  static constexpr auto Name = "attach-debug-info-to-isolated";

  std::vector<pipeline::ContractGroup> getContract() const {
    using namespace revng;
    using namespace pipeline;
    return { ContractGroup({ Contract(kinds::CFG,
                                      0,
                                      kinds::Isolated,
                                      1,
                                      InputPreservation::Preserve),
                             Contract(kinds::Isolated,
                                      1,
                                      kinds::Isolated,
                                      1,
                                      InputPreservation::Preserve) }) };
  }

  void run(pipeline::ExecutionContext &EC,
           const revng::pipes::CFGMap &CFGMap,
           pipeline::LLVMContainer &ModuleContainer) {
    llvm::legacy::PassManager Manager;
    Manager.add(new pipeline::LoadExecutionContextPass(&EC,
                                                       ModuleContainer.name()));
    Manager.add(new LoadModelWrapperPass(revng::getModelFromContext(EC)));
    Manager.add(new ControlFlowGraphCachePass(CFGMap));
    Manager.add(new pipeline::FunctionPass<AttachDebugInfo>());
    Manager.run(ModuleContainer.getModule());
  }
};

static pipeline::RegisterPipe<AttachDebugInfoToIsolatedPipe> Y1;

struct AttachDebugInfoToABIEnforcedPipe {
  static constexpr auto Name = "attach-debug-info-to-abi-enforced";

  std::vector<pipeline::ContractGroup> getContract() const {
    using namespace revng;
    using namespace pipeline;
    return { ContractGroup({ Contract(kinds::ABIEnforced,
                                      1,
                                      kinds::ABIEnforced,
                                      1,
                                      InputPreservation::Preserve),
                             Contract(kinds::CFG,
                                      0,
                                      kinds::Isolated,
                                      1,
                                      InputPreservation::Preserve) }) };
  }

  void run(pipeline::ExecutionContext &EC,
           const revng::pipes::CFGMap &CFGMap,
           pipeline::LLVMContainer &ModuleContainer) {
    llvm::legacy::PassManager Manager;
    Manager.add(new pipeline::LoadExecutionContextPass(&EC,
                                                       ModuleContainer.name()));
    Manager.add(new LoadModelWrapperPass(revng::getModelFromContext(EC)));
    Manager.add(new ControlFlowGraphCachePass(CFGMap));
    Manager.add(new pipeline::FunctionPass<AttachDebugInfo>());
    Manager.run(ModuleContainer.getModule());
  }
};

static pipeline::RegisterPipe<AttachDebugInfoToABIEnforcedPipe> Y2;
