#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/StringRef.h"

#include "revng/Pipeline/Context.h"
#include "revng/Pipeline/LLVMContainer.h"
#include "revng/Pipes/ModelGlobal.h"
#include "revng/Support/FunctionTags.h"

namespace revng::kinds {

/// A tagged function kind is a kind associated to tagged global elements. When
/// enumerating a llvm::Module it will produce a target for each global object
/// with that tag.
class TaggedFunctionKind : public pipeline::LLVMKind {
private:
  const FunctionTags::Tag *Tag;

  // It is on the heap to avoid Initialization Order Fiasco
  std::unique_ptr<llvm::SmallVector<TaggedFunctionKind *>> Children = nullptr;

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
    pipeline::LLVMKind(Name, Parent, Rank), Tag(&Tag) {}

  std::optional<pipeline::Target>
  symbolToTarget(const llvm::Function &Symbol) const override;

  void
  getInvalidations(const pipeline::Context &Ctx,
                   pipeline::TargetsList &ToRemove,
                   const pipeline::GlobalTupleTreeDiff &Diff) const override;

  void appendAllTargets(const pipeline::Context &Ctx,
                        pipeline::TargetsList &Out) const override;
};

} // namespace revng::kinds
