//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/ModuleSlotTracker.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"

#include "revng/LocalVariables/Statements.h"
#include "revng/Model/FunctionTags.h"
#include "revng/Support/Assert.h"
#include "revng/Support/CommandLine.h"
#include "revng/Support/Debug.h"
#include "revng/Support/IRBuilder.h"
#include "revng/Support/IRHelpers.h"

using namespace llvm;

static Logger Log{ "split-exponential-dataflow" };

static constexpr const char *SplitExponentialDataflowFlag = "split-exponential-"
                                                            "dataflow";

/// The largest expression, in nodes, the clifter is allowed to build.
///
/// Lowering it trades local variables for shorter expressions, raising it does
/// the opposite. 32 is the knee of the curve measured on real binaries.
static cl::opt<uint64_t> Threshold("split-exponential-dataflow-threshold",
                                   cl::desc("Maximum number of Clift "
                                            "expression nodes allowed in a "
                                            "single emitted C expression"),
                                   cl::init(32),
                                   cl::cat(MainCategory));

/// Stores into a local variable every value whose expression would grow past
/// `Threshold`.
struct SplitExponentialDataflow : public llvm::FunctionPass {
public:
  static char ID;

public:
  SplitExponentialDataflow() : FunctionPass(ID) {}

public:
  bool runOnFunction(llvm::Function &F) override;

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
  }

private:
  llvm::DenseMap<const llvm::Instruction *, uint64_t> Sizes;

  /// The instructions to move into a local variable. Acted upon only once
  llvm::SmallVector<llvm::Instruction *> ToSerialize;

  uint64_t sizeOf(const llvm::Value *V) const {
    if (revng::isExpressionLeaf(*V))
      return 1;

    auto It = Sizes.find(llvm::cast<llvm::Instruction>(V));
    revng_assert(It != Sizes.end(), "An instruction is used before its size");
    return It->second;
  }
};

/// Store \p I into a local variable and make all of its users read it back.
static void serialize(llvm::Function &F, llvm::Instruction &I) {
  revng::IRBuilder B(F.getContext());
  auto Location = I.getDebugLoc();

  B.SetInsertPointPastAllocas(&F, Location);
  auto *Alloca = B.createSimpleAlloca(I.getType());

  B.SetInsertPoint(&*std::next(I.getIterator()), Location);
  auto *Load = B.createLoadFromVariable(Alloca, I.getType());

  I.replaceAllUsesWith(Load);

  B.SetInsertPoint(Load, Location);
  B.createStoreToVariable(&I, Alloca);
}

bool SplitExponentialDataflow::runOnFunction(llvm::Function &F) {
  // The same pass object is reused for every function.
  Sizes.clear();
  ToSerialize.clear();

  llvm::ModuleSlotTracker MST(F.getParent(),
                              /* ShouldInitializeAllMetadata = */ false);
  if (Log.isEnabled())
    MST.incorporateFunction(F);

  // Reverse post order visits each instruction after the ones it uses, and
  // makes the result independent of how the blocks are laid out.
  for (llvm::BasicBlock *BB : llvm::ReversePostOrderTraversal(&F)) {
    for (llvm::Instruction &I : *BB) {
      revng_assert(not llvm::isa<llvm::PHINode>(&I),
                   "This pass runs after exit-ssa, there should be no PHIs");

      // Only the instructions the clifter inlines into their users can grow
      // an expression; a statement is already emitted exactly once.
      if (revng::isNotEmitted(I) or revng::isStatement(I)) {
        Sizes[&I] = 1;
        continue;
      }

      uint64_t Size = 1;
      for (const llvm::Value *Operand : I.operand_values())
        Size += sizeOf(Operand);

      if (isCallToTagged(&I, FunctionTags::StructInitializer)) {
        // StructInitializer already has a special case in the Clifter because
        // it's always emitted inline as `return (struct_X){ field0, field1};`
        // For this reason, we ignore the Threshold for it. This may cause
        // problems, but since StructInitializers are used only in Raw functions
        // returning register sets, the size shouldn't blow up anyway.
        // So we never do anything for them.
      } else if (Size > Threshold) {
        // For all the other cases use the threshold.
        revng_log(Log,
                  "Splitting an expression of " << Size << " nodes at "
                                                << dumpToString(&I, MST));
        ToSerialize.push_back(&I);

        // Readers now see a single node instead of the subtree. This is what
        // bounds the estimate: no operand is ever worth more than `Threshold`.
        Size = 1;
      }

      Sizes[&I] = Size;
    }
  }

  // Only now: serializing inserts new instructions.
  for (llvm::Instruction *I : ToSerialize)
    serialize(F, *I);

  return not ToSerialize.empty();
}

char SplitExponentialDataflow::ID = 0;

using Reg = llvm::RegisterPass<SplitExponentialDataflow>;
static Reg X{ SplitExponentialDataflowFlag,
              SplitExponentialDataflowFlag,
              false,
              false };
