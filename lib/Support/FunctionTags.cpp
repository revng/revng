/// \file FunctionTags.cpp
/// \brief

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <map>

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Support/ManagedStatic.h"

#include "revng/Support/Assert.h"
#include "revng/Support/FunctionTags.h"
#include "revng/Support/IRHelpers.h"

using namespace llvm;

namespace FunctionTags {

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
    if (auto *CalledFunc = Call->getCalledFunction())
      if (T.isTagOf(CalledFunc))
        return Call;

  return nullptr;
}

llvm::CallInst *getCallToTagged(llvm::Value *V, const FunctionTags::Tag &T) {
  if (auto *Call = llvm::dyn_cast_or_null<llvm::CallInst>(V))
    if (auto *CalledFunc = Call->getCalledFunction())
      if (T.isTagOf(CalledFunc))
        return Call;

  return nullptr;
}

const llvm::CallInst *getCallToIsolatedFunction(const llvm::Value *V) {
  if (const llvm::CallInst *Call = getCallToTagged(V, FunctionTags::Isolated)) {
    // The callee is an isolated function
    return Call;
  } else if (const llvm::CallInst
               *Call = getCallToTagged(V, FunctionTags::DynamicFunction)) {
    // The callee is a dynamic function
    return Call;
  } else if (auto *Call = dyn_cast<llvm::CallInst>(V)) {
    // It's a call to an isolated function if it's indirect
    return Call->getCalledFunction() == nullptr ? Call : nullptr;
  } else {
    return nullptr;
  }
}

llvm::CallInst *getCallToIsolatedFunction(llvm::Value *V) {
  if (llvm::CallInst *Call = getCallToTagged(V, FunctionTags::Isolated)) {
    // The callee is an isolated function
    return Call;
  } else if (llvm::CallInst
               *Call = getCallToTagged(V, FunctionTags::DynamicFunction)) {
    // The callee is a dynamic function
    return Call;
  } else if (auto *Call = dyn_cast<llvm::CallInst>(V)) {
    // It's a call to an isolated function if it's indirect
    return Call->getCalledFunction() == nullptr ? Call : nullptr;
  } else {
    return nullptr;
  }
}
