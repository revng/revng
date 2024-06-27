#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/StringRef.h"

#include "revng/Pipeline/Context.h"
#include "revng/Pipeline/LLVMContainer.h"
#include "revng/Pipes/ModelGlobal.h"
#include "revng/Support/FunctionTags.h"
#include "revng/Support/Generator.h"

namespace revng::kinds {

/// A tagged function kind is a kind associated to tagged global elements. When
/// enumerating a llvm::Module it will produce a target for each global object
/// with that tag.
class TaggedFunctionKind : public pipeline::LLVMKind {
private:
  const FunctionTags::Tag *Tag;

  // It is on the heap to avoid Initialization Order Fiasco
  std::unique_ptr<llvm::SmallVector<TaggedFunctionKind *>> Children = nullptr;

  // we have a redundant extra list of children here that would be a subset of
  // that present in kind because kinds are not upcastable, and so we would not
  // be able to know which one we care about without a second list.
  void registerChild(TaggedFunctionKind *Child) {
    if (Children == nullptr)
      Children = std::make_unique<llvm::SmallVector<TaggedFunctionKind *>>();

    Children->push_back(Child);
  }

public:
  template<typename BaseRank>
  TaggedFunctionKind(llvm::StringRef Name,
                     const BaseRank &Rank,
                     const FunctionTags::Tag &Tag) :
    pipeline::LLVMKind(Name, Rank), Tag(&Tag) {}

  template<typename BaseRank>
  TaggedFunctionKind(llvm::StringRef Name,
                     TaggedFunctionKind &Parent,
                     const BaseRank &Rank,
                     const FunctionTags::Tag &Tag) :
    pipeline::LLVMKind(Name, Parent, Rank), Tag(&Tag) {
    Parent.registerChild(this);
  }

  std::optional<pipeline::Target>
  symbolToTarget(const llvm::Function &Symbol) const override;

  void appendAllTargets(const pipeline::Context &Ctx,
                        pipeline::TargetsList &Out) const override;

  static cppcoro::generator<
    std::pair<const model::Function *, llvm::Function *>>
  getFunctionsAndCommit(pipeline::ExecutionContext &Context,
                        llvm::Module &Module,
                        llvm::StringRef ContainerName);

  static cppcoro::generator<const model::Function *>
  getFunctionsAndCommit(pipeline::ExecutionContext &Context,
                        const pipeline::ContainerBase &Container);
};

} // namespace revng::kinds
