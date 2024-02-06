//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/TypeFinder.h"
#include "llvm/Transforms/Utils.h"
#include "llvm/Transforms/Utils/Local.h"

#include "revng/EarlyFunctionAnalysis/BasicBlock.h"
#include "revng/EarlyFunctionAnalysis/FunctionMetadataCache.h"
#include "revng/Model/IRHelpers.h"
#include "revng/Model/LoadModelPass.h"
#include "revng/Model/RawFunctionType.h"
#include "revng/Pipeline/Location.h"
#include "revng/Pipeline/RegisterLLVMPass.h"
#include "revng/Pipes/Ranks.h"
#include "revng/Support/FunctionTags.h"

#include "revng-c/Pipes/Kinds.h"
#include "revng-c/Support/FunctionTags.h"
#include "revng-c/Support/IRHelpers.h"
#include "revng-c/TypeNames/ModelTypeNames.h"

using namespace llvm;
using pipeline::serializedLocation;
namespace ranks = revng::ranks;

static Logger<> Log{ "prepare-llvmir-for-mlir" };

static void saveFunctionEntryPointInDISubprogram(llvm::Function &F) {
  QuickMetadata QMD(getContext(F.getParent()));

  std::string FunctionEntryLocation;
  auto MaybeMetaAddress = getMetaAddressMetadata(&F, FunctionEntryMDNName);
  if (MaybeMetaAddress != MetaAddress::invalid()) {
    FunctionEntryLocation = serializedLocation(ranks::Function,
                                               MaybeMetaAddress);
    revng_log(Log, "Function entry: " << FunctionEntryLocation);
    // For the purpose of preserving `!revng.function.entry`, let's map it in
    // DISubprogram's `linkageName;` field.
    auto SP = F.getSubprogram();
    SP->replaceRawLinkageName(QMD.get(FunctionEntryLocation));
    revng_log(Log,
              "Saved !revng.function.entry in DISubprogram: "
                << SP->getRawLinkageName()->getString());
  } else {
    revng_log(Log,
              "WARNING: Function " << F.getName()
                                   << " with an invalid MetaAddress.");
  }
}

static void handleFunctionEntryPoint(llvm::Function &F) {
  if (F.getSubprogram()) {
    saveFunctionEntryPointInDISubprogram(F);
    return;
  }

  // If there is no DISubprogram attached to the function, use the one we find
  // attached to instructions from it.
  for (llvm::BasicBlock &BB : F) {
    for (llvm::Instruction &I : BB) {
      auto DebugLoc = I.getDebugLoc();
      if (not DebugLoc)
        continue;

      auto InlinedAt = DebugLoc->getInlinedAt();
      if (not InlinedAt)
        continue;

      if (auto SP = dyn_cast<DISubprogram>(InlinedAt->getScope())) {
        revng_log(Log, "Attaching !dbg to " << F.getName());
        F.setSubprogram(SP);
        saveFunctionEntryPointInDISubprogram(F);
        return;
      }
    }
  }
}

static void saveFunctionEntryPointInDISubprogram(Module &M) {
  for (llvm::Function &F : M)
    handleFunctionEntryPoint(F);
}

static void adjustRevngMetadata(Module &M) {
  for (llvm::Function &F : M) {
    for (llvm::BasicBlock &BB : F) {
      for (llvm::Instruction &I : BB) {
        // Clean up metadata we don't need. Also, we abused this metadata by
        // attaching some non standard register state metadata to stores and
        // loads, and we don't want it preserved in the LLVM Dialect.
        if (MDNode *Node = I.getMetadata(llvm::LLVMContext::MD_noalias))
          I.setMetadata(llvm::LLVMContext::MD_noalias, nullptr);

        if (auto *Call = dyn_cast<CallBase>(&I)) {
          // revng abuses debug info in order to keep mapping between addresses
          // of instructions and decompiled C code. revng generates LLVM IR that
          // is good enough, by attaching !dbg/DILocation attachments to
          // llvm::Instructions only, and by avoiding to create/attach different
          // !dbg/DISubprogram attachment to each llvm::Function. By avoiding
          // the later, we are 1) able to use just one DISubprogram from root
          // function (please note that DIlocation needs a DISubprogram); 2) we
          // have generated away less DISubprograms, since if we chose to attach
          // to each LLVM Function we need to create a new DISubprogram for each
          // of them, so it is away more debug info to carry along the pipeline;
          // 3) by avoiding the attachment on LLVM Function we avoid verifying
          // of debug info inside functions, such as the one we are fixing very
          // late - this could be annoying for LLVM Passes in the pipeline,
          // since some fixes like this one we are applying here very late, when
          // producing MLIR could be needed at several places/passes earlier,
          // e.g. if a Pass creates a call to a function that could be inlined,
          // it needs to have a !dbg/DILocation attachment (at least an
          // artificial one - DILocation(line: 0)), since calls to inlinable
          // functions must have a !dbg attachment.
          if (Call->getFunction()->getSubprogram()
              and Call->getCalledFunction()) {
            auto Location = DILocation::get(M.getContext(),
                                            0,
                                            0,
                                            F.getSubprogram(),
                                            nullptr);
            Call->setDebugLoc(Location);
          }
        }
      }
    }
  }
}

static void setStructNameIfNeeded(llvm::Type *Type,
                                  const std::string &StructName) {
  if (llvm::StructType *STy = llvm::dyn_cast<llvm::StructType>(Type)) {
    if (not STy->isLiteral() and not STy->hasName()) {
      STy->setName(StructName);
    }
  }
}

/// Give a name to all anonymous structs, because LLVM MLIR dialect does not
/// expect nameless structs. Only literals can be anonymous.
static void adjustAnonymousStructs(Module &M, const model::Binary &Model) {
  using PTMLCBuilder = ptml::PTMLCBuilder;
  PTMLCBuilder B(/*GeneratePlainC*/ true);
  FunctionMetadataCache Cache;

  for (llvm::Function &F : M) {
    auto FunctionTags = FunctionTags::TagsSet::from(&F);
    bool IsIsolated = FunctionTags.contains(FunctionTags::Isolated)
                      or FunctionTags.contains(FunctionTags::CSVsPromoted);

    if (not IsIsolated
        and not FunctionTags.contains(FunctionTags::DynamicFunction))
      continue;

    const model::Type *Prototype = nullptr;
    if (IsIsolated) {
      const model::Function *ModelFunc = llvmToModelFunction(Model, F);
      Prototype = ModelFunc->Prototype().getConst();
    } else if (FunctionTags.contains(FunctionTags::DynamicFunction)) {
      llvm::StringRef SymbolName = F.getName().drop_front(strlen("dynamic_"));
      auto It = Model.ImportedDynamicFunctions().find(SymbolName.str());
      revng_assert(It != Model.ImportedDynamicFunctions().end());
      const auto &TTR = It->prototype(Model);
      revng_assert(TTR.isValid());
      Prototype = TTR.getConst();
    }

    revng_assert(Prototype);

    for (Argument &Argument : F.args())
      revng_assert(not Argument.getType()->isStructTy());

    llvm::Type *ReturnType = F.getReturnType();
    std::string
      ReturnTypeName = getReturnTypeName(*Prototype, B, false).str().str();
    setStructNameIfNeeded(ReturnType, ReturnTypeName);

    // Look for all the call sites: they might return anonymous structs too
    for (Instruction &I : llvm::instructions(F)) {
      auto *T = I.getType();
      CallInst *Call = getCallToIsolatedFunction(&I);
      if (Call != nullptr and T->isStructTy()) {
        const auto &Prototype = *Cache.getCallSitePrototype(Model, Call).get();
        auto
          ReturnTypeName = getReturnTypeName(Prototype, B, false).str().str();
        setStructNameIfNeeded(T, ReturnTypeName);
      }
    }
  }

  // Make sure we have no nameless structs.
  llvm::TypeFinder StructTypes;
  StructTypes.run(M, /* OnlyNamed */ false);

  for (auto *STy : StructTypes) {
    if (not STy->isLiteral())
      revng_assert(STy->hasName());
  }
}

// For the purpose of preserving the `!revng.tags` metadata, we incorporate
// the tag within function name.
static void tagFunction(Function &F) {
  auto FunctionTags = FunctionTags::TagsSet::from(&F);

  // Those should never appear at this stage.
  static const FunctionTags::TagsSet UnexpectedTags = {
    &FunctionTags::ModelCast,
    &FunctionTags::Parentheses,
    &FunctionTags::LiteralPrintDecorator,
    &FunctionTags::HexInteger,
    &FunctionTags::CharInteger,
    &FunctionTags::BoolInteger,
    &FunctionTags::NullPtr,
    &FunctionTags::ReadsMemory,
    &FunctionTags::UnaryMinus,
    &FunctionTags::BinaryNot,
    &FunctionTags::Copy,
    &FunctionTags::Root,
    &FunctionTags::IsolatedRoot,
    &FunctionTags::Marker,
    &FunctionTags::FunctionDispatcher,
    &FunctionTags::StackOffsetMarker
  };

  static const FunctionTags::TagsSet IgnoredTags = {
    &FunctionTags::Assign,
    &FunctionTags::AllocatesLocalVariable,
    &FunctionTags::MallocLike,
    &FunctionTags::IsRef,
    &FunctionTags::ModelGEP,
    &FunctionTags::ModelGEPRef,
    &FunctionTags::WritesMemory,
    &FunctionTags::Exceptional,
    &FunctionTags::CSVsAsArgumentsWrapper
  };

  static const FunctionTags::TagsSet SuppressedByDebugInfo = {
    &FunctionTags::LiftingArtifactsRemoved,
    &FunctionTags::StackPointerPromoted,
    &FunctionTags::StackAccessesSegregated,
    &FunctionTags::DecompiledToYAML,
    &FunctionTags::Isolated,
    &FunctionTags::ABIEnforced,
    &FunctionTags::CSVsPromoted,
    &FunctionTags::DynamicFunction
  };

  for (const auto &Tag : FunctionTags) {

    revng_assert(not UnexpectedTags.contains(*Tag));

    if (IgnoredTags.contains(*Tag)) {
      revng_log(Log,
                "Ignoring Tag: " << Tag->name()
                                 << " for Function: " << F.getName());
      continue;
    }

    if (SuppressedByDebugInfo.contains(*Tag)) {
      revng_log(Log,
                "Tag: " << Tag->name()
                        << " was suppressed by !dbg for Function: "
                        << F.getName());
      continue;
    }

    // Save revng-c tags we care about.
    revng_log(Log,
              "Saving Tag: " << Tag->name()
                             << " for Function: " << F.getName());
    F.setName(F.getName() + "_" + Tag->name());
  }
}

static void saveFunctionTags(Module &M) {
  for (llvm::Function &F : M)
    tagFunction(F);
}

struct PrepareLLVMIRForMLIRPass : public ModulePass {
public:
  static char ID;

  PrepareLLVMIRForMLIRPass() : ModulePass(ID) {}

  bool runOnModule(Module &M) override {
    auto &Model = getAnalysis<LoadModelWrapperPass>().get().getReadOnlyModel();
    adjustAnonymousStructs(M, *Model);
    saveFunctionTags(M);
    saveFunctionEntryPointInDISubprogram(M);
    adjustRevngMetadata(M);

    return true;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    AU.addRequired<LoadModelWrapperPass>();
  }
};

char PrepareLLVMIRForMLIRPass::ID = 0;
static constexpr const char *Flag = "prepare-llvmir-for-mlir";
using Register = RegisterPass<PrepareLLVMIRForMLIRPass>;
static Register X(Flag,
                  "Pass that removes things that we do not need in "
                  "MLIR. ",
                  false,
                  false);
