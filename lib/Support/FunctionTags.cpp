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

Tag QEMU("QEMU");
Tag Helper("Helper");
Tag Lifted("Lifted");
Tag CallToLifted("CallToLifted");
Tag Exceptional("Exceptional");
Tag StructInitializer("StructInitializer");
Tag OpaqueCSVValue("OpaqueCSVValue");
Tag FunctionDispatcher("FunctionDispatcher");
Tag Root("Root");
Tag CSVsAsArgumentsWrapper("CSVsAsArgumentsWrapper");
Tag Marker("Marker");
Tag DynamicFunction("DynamicFunction");

static const char *TagsMetadataName = "revng.tags";

static ManagedStatic<std::map<StringRef, Tag *>> Registry;

template<typename T>
static llvm::MDNode *getMetadata(T *V) {
  return V->getMetadata(TagsMetadataName);
}

template<typename T>
static void append(T *V, StringRef Name) {
  LLVMContext &C = getContext(V);
  SmallVector<Metadata *, 8> Tags;

  if (auto *Tuple = getMetadata(V)) {
    bool Found = false;
    for (const MDOperand &Op : Tuple->operands()) {
      MDString *String = cast<MDString>(Op.get());
      if (String->getString() == Name)
        return;

      Tags.push_back(String);
    }
  }

  Tags.push_back(MDString::get(C, Name));

  V->setMetadata(TagsMetadataName, MDTuple::get(C, Tags));
}

Tag::Tag(StringRef Name) : Name(Name) {
  revng_check(Registry->count(Name) == 0,
              "Tag with the same name already registered");
  (*Registry)[Name] = this;
}

void Tag::addTo(Instruction *I) const {
  append(I, Name);
}

void Tag::addTo(GlobalObject *G) const {
  append(G, Name);
}

TagsSet TagsSet::from(const MDNode *MD) {
  TagsSet Result;

  if (MD == nullptr)
    return Result;

  for (const MDOperand &Op : cast<MDTuple>(MD)->operands()) {
    StringRef Name = cast<MDString>(Op.get())->getString();
    auto It = Registry->find(Name);
    if (It != Registry->end())
      Result.Tags.insert(It->second);
  }

  return Result;
}

TagsSet TagsSet::from(const Instruction *I) {
  return from(getMetadata(I));
}

TagsSet TagsSet::from(const GlobalObject *G) {
  return from(getMetadata(G));
}

} // namespace FunctionTags
