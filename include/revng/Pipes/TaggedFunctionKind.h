#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/StringRef.h"

#include "revng/Pipeline/LLVMContainer.h"
#include "revng/Pipes/AllFunctions.h"
#include "revng/Pipes/ModelGlobal.h"
#include "revng/Support/FunctionTags.h"

namespace revng::pipes {

/// A tagged function kind is a kind associated to tagged global elements. When
/// enumerating a llvm::Module it will produce a target for each global object
/// with that tag.
class TaggedFunctionKind : public pipeline::LLVMKind {
private:
  const FunctionTags::Tag *Tag;

public:
  TaggedFunctionKind(llvm::StringRef Name,
                     pipeline::Rank *Rank,
                     const FunctionTags::Tag &Tag) :
    pipeline::LLVMKind(Name, Rank), Tag(&Tag) {}

  TaggedFunctionKind(llvm::StringRef Name,
                     TaggedFunctionKind &Parent,
                     pipeline::Rank *Rank,
                     const FunctionTags::Tag &Tag) :
    pipeline::LLVMKind(Name, Parent, Rank), Tag(&Tag) {}

  pipeline::TargetsList
  compactTargets(const pipeline::Context &Ctx,
                 pipeline::TargetsList::List &Targets) const final {
    return compactFunctionTargets(getModelFromContext(Ctx), Targets, *this);
  }

  void expandTarget(const pipeline::Context &Ctx,
                    const pipeline::Target &Input,
                    pipeline::TargetsList &Output) const override;

  std::optional<pipeline::Target>
  symbolToTarget(const llvm::Function &Symbol) const override;
};

} // namespace revng::pipes
