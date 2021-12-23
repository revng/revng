#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//
#include <vector>

#include "llvm/ADT/STLExtras.h"

#include "revng/Model/IRHelpers.h"
#include "revng/Pipeline/Context.h"
#include "revng/Pipeline/GenericLLVMPipe.h"
#include "revng/Pipeline/Kind.h"
#include "revng/Pipeline/LLVMContainer.h"
#include "revng/Pipeline/LLVMGlobalKindBase.h"
#include "revng/Pipeline/PathComponent.h"
#include "revng/Pipeline/Target.h"
#include "revng/Pipes/ModelGlobal.h"

namespace revng::pipes {
extern pipeline::Rank FunctionsRank;

/**
 * A tagged function kind is a kind associated to tagged global elements
 * When enumerating a llvm module it will produce a target for each global
 * object with that tag.
 *
 **/
class TaggedFunctionKind : public pipeline::LLVMKind {
private:
  const FunctionTags::Tag *Tag;

public:
  TaggedFunctionKind(llvm::StringRef Name, const FunctionTags::Tag &Tag) :
    pipeline::LLVMKind(Name, &FunctionsRank), Tag(&Tag) {}

  TaggedFunctionKind(llvm::StringRef Name,
                     TaggedFunctionKind &Parent,
                     const FunctionTags::Tag &Tag) :
    pipeline::LLVMKind(Name, Parent, &FunctionsRank), Tag(&Tag) {}

  pipeline::TargetsList
  compactTargets(const pipeline::Context &Ctx,
                 pipeline::TargetsList::List &Targets) const final {

    auto &Model = getModelFromContext(Ctx);
    std::set<std::string> Set;
    const pipeline::Target AllFunctions({ pipeline::PathComponent("root"),
                                          pipeline::PathComponent::all() },
                                        *this);

    // enumerate the targets
    for (const pipeline::Target &Target : Targets) {
      // if we see a * then no need to check if we have all functions, just
      // return that.
      if (Target.getPathComponents().back().isAll())
        return pipeline::TargetsList({ AllFunctions });

      Set.insert(Target.getPathComponents().back().getName());
    }

    // check if all functions in the model are in the targets
    // if they are, return *
    if (llvm::all_of(Model.Functions, [&Set](const model::Function &F) {
          return Set.contains(F.Entry.toString());
        })) {

      return pipeline::TargetsList({ AllFunctions });
    }

    return Targets;
  }

  void expandTarget(const pipeline::Context &Ctx,
                    const pipeline::Target &Input,
                    pipeline::TargetsList &Output) const override;

  std::optional<pipeline::Target>
  symbolToTarget(const llvm::Function &Symbol) const override {
    if (not Tag->isTagOf(&Symbol) or Symbol.isDeclaration())
      return std::nullopt;

    auto Address = getMetaAddressOfIsolatedFunction(Symbol);
    revng_assert(Address.isValid());
    return pipeline::Target({ "root", Address.toString() }, *this);
  }
};

extern TaggedFunctionKind Isolated;

} // namespace revng::pipes
