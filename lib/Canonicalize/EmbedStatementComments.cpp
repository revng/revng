//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/IRBuilder.h"

#include "revng/PTML/CommentPlacementHelper.h"
#include "revng/Pipeline/LLVMContainer.h"
#include "revng/Pipeline/Location.h"
#include "revng/Pipeline/Pipe.h"
#include "revng/Pipeline/RegisterPipe.h"
#include "revng/Pipes/DebugInfoHelpers.h"
#include "revng/Pipes/Kinds.h"
#include "revng/Pipes/Ranks.h"
#include "revng/Support/DecompilationHelpers.h"

// This name is not present in the emitted C.
RegisterIRHelper CommentHelper("comment");

template<typename ResultSet>
static ResultSet gatherNonStatementAddresses(const llvm::Instruction &I,
                                             ResultSet &&Result = {}) {
  if (std::optional MaybeAddress = revng::tryExtractAddress(I))
    Result.emplace(*MaybeAddress);

  for (const llvm::Use &V : I.operands())
    if (const llvm::Instruction *Cast = llvm::dyn_cast<llvm::Instruction>(V))
      if (not isStatement(*Cast))
        gatherNonStatementAddresses(*Cast, Result);

  return Result;
}

template<>
struct yield::StatementTraits<llvm::BasicBlock *> {

  using StatementType = llvm::Instruction *;

  static RangeOf<StatementType> auto getStatements(llvm::BasicBlock *Node) {
    return *Node | std::views::filter(isStatement)
           | std::views::transform([](auto &&R) { return &R; });
  }

  static std::set<MetaAddress> getAddresses(StatementType Statement) {
    return gatherNonStatementAddresses<std::set<MetaAddress>>(*Statement);
  }
};

struct EmbedStatementComments {
public:
  static constexpr auto Name = "embed-statement-comments";

private:
  llvm::FunctionCallee makeIRComment(llvm::Module &M) {
    using StringLiteral = const uint8_t *;
    llvm::FunctionType &FT = *createFunctionType<void,
                                                 int64_t,
                                                 bool,
                                                 StringLiteral,
                                                 StringLiteral>(M.getContext());
    auto Result = getOrInsertIRHelper("comment", M, &FT);
    auto &Callee = *llvm::cast<llvm::Function>(Result.getCallee());
    Callee.addFnAttr(llvm::Attribute::NoUnwind);
    Callee.addFnAttr(llvm::Attribute::WillReturn);
    Callee.addFnAttr(llvm::Attribute::NoMerge);
    Callee.setDoesNotAccessMemory();
    FunctionTags::Comment.addTo(&Callee);

    return Result;
  }

public:
  inline std::array<pipeline::ContractGroup, 1> getContract() const {
    return { pipeline::ContractGroup(revng::kinds::StackAccessesSegregated,
                                     0,
                                     revng::kinds::StackAccessesSegregated) };
  }

  void run(pipeline::ExecutionContext &Context,
           pipeline::LLVMContainer &ModuleContainer) {
    llvm::Module &M = ModuleContainer.getModule();
    llvm::IRBuilder<> B(M.getContext());

    llvm::FunctionCallee IRComment = makeIRComment(M);

    using TaggedFunctionKind = revng::kinds::TaggedFunctionKind;
    llvm::StringRef ContainerName = ModuleContainer.name();

    for (auto &&[ModelFunction, LLVMFunction] :
         TaggedFunctionKind::getFunctionsAndCommit(Context, M, ContainerName)) {
      if (ModelFunction->Comments().empty())
        continue;

      using MapT = yield::CommentPlacementHelper<llvm::BasicBlock *>;
      MapT CM(*ModelFunction, *LLVMFunction);

      auto EmitAComment = [&](llvm::Instruction *Where,
                              const MapT::CommentAssignment &Comment,
                              llvm::StringRef EmittedLocation) {
        B.SetInsertPoint(Where);

        std::array<llvm::Value *, 4> Arguments = {
          llvm::ConstantInt::get(llvm::Type::getInt64Ty(B.getContext()),
                                 Comment.CommentIndex),
          llvm::ConstantInt::get(llvm::Type::getInt8Ty(B.getContext()),
                                 Comment.LocationMatchesExactly),
          getUniqueString(&M, addressesToString(*Comment.ExpectedLocation)),
          getUniqueString(&M, EmittedLocation)
        };

        auto *Call = B.CreateCall(IRComment, Arguments);
        Call->copyMetadata(*Where);
      };

      llvm::SmallVector<llvm::Value *, 8> Argumentss;
      llvm::BasicBlock &EntryBlock = LLVMFunction->getEntryBlock();
      for (llvm::BasicBlock *Node : llvm::depth_first(&EntryBlock)) {
        using Trait = yield::StatementTraits<llvm::BasicBlock *>;
        for (llvm::Instruction *I : Trait::getStatements(Node))
          for (const MapT::CommentAssignment &Comment : CM.getComments(I))
            EmitAComment(I, Comment, addressesToString(Trait::getAddresses(I)));
      }

      for (const MapT::CommentAssignment &Comment : CM.getHomelessComments()) {
        // For now emit homeless comments at the very top of the function.
        // TODO: find a better place for them.
        EmitAComment(&*LLVMFunction->begin()->begin(),
                     Comment,
                     "at the function entry point");
      }
    }
  }
};

static pipeline::RegisterPipe<EmbedStatementComments> ESCPipe;
