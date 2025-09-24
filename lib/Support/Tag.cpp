/// \file Tags.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <map>

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Support/ManagedStatic.h"

#include "revng/Model/FunctionTags.h"
#include "revng/Support/Assert.h"
#include "revng/Support/IRHelpers.h"

using namespace llvm;

namespace FunctionTags {

// These are here because they are needed by Pipeline, LLVMContainer
// specifically
Tag UniquedByMetadata("uniqued-by-metadata");
Tag UniquedByPrototype("uniqued-by-prototype");
Tag Isolated("isolated");

Tag::Tag(llvm::StringRef Name) : DynamicHierarchy(Name) {
}
Tag::Tag(llvm::StringRef Name, Tag &Parent) : DynamicHierarchy(Name, Parent) {
}

llvm::MDNode *TagsSet::getMetadata(LLVMContext &C) const {
  SmallVector<Metadata *, 8> MDTags;
  for (const Tag *T : Tags)
    MDTags.push_back(MDString::get(C, T->name()));
  return MDTuple::get(C, MDTags);
}

TagsSet TagsSet::from(const MDNode *MD) {
  TagsSet Result;

  if (MD == nullptr)
    return Result;

  for (const MDOperand &Op : cast<MDTuple>(MD)->operands()) {
    StringRef Name = cast<MDString>(Op.get())->getString();
    Tag *T = Tag::findByName(Name);
    revng_assert(T != nullptr);
    Result.Tags.insert(T);
  }

  return Result;
}

} // namespace FunctionTags

const llvm::CallInst *getCallToTagged(const llvm::Value *V,
                                      const FunctionTags::Tag &T) {
  if (auto *Call = llvm::dyn_cast_or_null<llvm::CallInst>(V))
    if (auto *CalledFunc = getCalledFunction(Call))
      if (T.isTagOf(CalledFunc))
        return Call;

  return nullptr;
}

llvm::CallInst *getCallToTagged(llvm::Value *V, const FunctionTags::Tag &T) {
  if (auto *Call = llvm::dyn_cast_or_null<llvm::CallInst>(V))
    if (auto *CalledFunc = getCalledFunction(Call))
      if (T.isTagOf(CalledFunc))
        return Call;

  return nullptr;
}
