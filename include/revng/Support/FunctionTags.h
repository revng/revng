#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <set>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/GlobalObject.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Module.h"

#include "revng/Support/DynamicHierarchy.h"

namespace llvm {
class MDNode;
class User;
class Module;
} // namespace llvm

namespace FunctionTags {

class Tag;

template<typename T>
concept Taggable = (std::is_same_v<llvm::GlobalVariable, T>
                    or std::is_same_v<llvm::Function, T>);

/// Represents a set of Tag that can be attached to an
/// Instruction/GlobalVariable/Function
///
/// \note This class automatically deduplicates Tags and keeps the most specific
///       Tag available, in case two Tags that are in a parent-child
///       relationship are added to the set
class TagsSet {
private:
  static constexpr const char *TagsMetadataName = "revng.tags";

private:
  std::set<const Tag *> Tags;

public:
  TagsSet() {}
  TagsSet(std::initializer_list<const Tag *> I) : Tags{ I } {}

public:
  static TagsSet from(const Taggable auto *V) {
    return from(V->getMetadata(TagsMetadataName));
  }
  static TagsSet from(const llvm::MDNode *MD);

public:
  auto begin() const { return Tags.begin(); }
  auto end() const { return Tags.end(); }

public:
  bool containsExactly(const Tag &Target) const {
    return Tags.contains(&Target);
  }

  // TODO: This seems non-obvious to me. I feel like it would be more natural
  //       for this to be called `containsDescendants`, while `containsExactly`
  //       could either stay as is or be renamed into just `contains`.
  bool contains(const Tag &Target) const;

public:
  void set(Taggable auto *V) const {
    V->setMetadata(TagsMetadataName, getMetadata(V->getContext()));
  }

public:
  void insert(const Tag &Target);

private:
  llvm::MDNode *getMetadata(llvm::LLVMContext &C) const;
};

/// Represents a tag that can be attached to a
/// Instruction/GlobalVariable/Function
///
/// \note Tag can have a parent tag.
class Tag : public DynamicHierarchy<Tag> {
public:
  Tag(llvm::StringRef Name) : DynamicHierarchy(Name) {}
  Tag(llvm::StringRef Name, Tag &Parent) : DynamicHierarchy(Name, Parent) {}

public:
  void addTo(Taggable auto *I) const {
    auto Set = TagsSet::from(I);
    Set.insert(*this);
    Set.set(I);
  }

public:
  bool isTagOf(const Taggable auto *I) const {
    return TagsSet::from(I).contains(*this);
  }

  bool isExactTagOf(const Taggable auto *I) const {
    return TagsSet::from(I).containsExactly(*this);
  }

  auto functions(llvm::Module *M) const {
    using namespace llvm;
    auto Filter = [this](Function &F) { return isTagOf(&F); };
    return make_filter_range(M->functions(), Filter);
  }

  auto functions(const llvm::Module *M) const {
    using namespace llvm;
    auto Filter = [this](const Function &F) { return isTagOf(&F); };
    return make_filter_range(M->functions(), Filter);
  }

  auto exactFunctions(llvm::Module *M) const {
    using namespace llvm;
    auto Filter = [this](Function &F) { return isExactTagOf(&F); };
    return make_filter_range(M->functions(), Filter);
  }

  auto exactFunctions(const llvm::Module *M) const {
    using namespace llvm;
    auto Filter = [this](const Function &F) { return isExactTagOf(&F); };
    return make_filter_range(M->functions(), Filter);
  }

  auto globals(llvm::Module *M) const {
    using namespace llvm;
    auto Filter = [this](GlobalVariable &G) { return isTagOf(&G); };
    return make_filter_range(M->globals(), Filter);
  }

  auto globals(const llvm::Module *M) const {
    using namespace llvm;
    auto Filter = [this](const GlobalVariable &G) { return isTagOf(&G); };
    return make_filter_range(M->globals(), Filter);
  }

  auto exactGlobals(llvm::Module *M) const {
    using namespace llvm;
    auto Filter = [this](GlobalVariable &G) { return isExactTagOf(&G); };
    return make_filter_range(M->globals(), Filter);
  }

  auto exactGlobals(const llvm::Module *M) const {
    using namespace llvm;
    auto Filter = [this](const GlobalVariable &G) { return isExactTagOf(&G); };
    return make_filter_range(M->globals(), Filter);
  }
};

inline bool TagsSet::contains(const Tag &Target) const {
  for (const Tag *T : Tags)
    if (Target.ancestorOf(*T))
      return true;
  return false;
}

inline void TagsSet::insert(const Tag &Target) {
  for (auto It = Tags.begin(); It != Tags.end();) {
    if (Target.ancestorOf(**It)) {
      // No need to insert, we already have a Tag derived from Target in the set
      return;
    } else if ((*It)->ancestorOf(Target)) {
      // Target is more specific than (*It), delete (*It)
      It = Tags.erase(It);
    } else {
      ++It;
    }
  }

  Tags.insert(&Target);
}

inline Tag QEMU("QEMU");
inline Tag Helper("Helper");

inline Tag Isolated("Isolated");
inline Tag ABIEnforced("ABIEnforced", Isolated);
inline Tag CSVsPromoted("CSVsPromoted", ABIEnforced);

inline Tag Exceptional("Exceptional");
inline Tag StructInitializer("StructInitializer");
inline Tag OpaqueCSVValue("OpaqueCSVValue");
inline Tag FunctionDispatcher("FunctionDispatcher");
inline Tag Root("Root");
inline Tag IsolatedRoot("IsolatedRoot");
inline Tag CSVsAsArgumentsWrapper("CSVsAsArgumentsWrapper");
inline Tag Marker("Marker");
inline Tag DynamicFunction("DynamicFunction");
inline Tag ClobbererFunction("ClobbererFunction");
inline Tag ReaderFunction("ReaderFunction");

inline Tag CSV("CSV");

} // namespace FunctionTags

inline bool isRootOrLifted(const llvm::Function *F) {
  auto Tags = FunctionTags::TagsSet::from(F);
  return Tags.contains(FunctionTags::Root)
         or Tags.contains(FunctionTags::Isolated);
}

//
// {is,get}CallToTagged
//
const llvm::CallInst *getCallToTagged(const llvm::Value *V,
                                      const FunctionTags::Tag &T);

llvm::CallInst *getCallToTagged(llvm::Value *V, const FunctionTags::Tag &T);

inline bool isCallToTagged(const llvm::Value *V, const FunctionTags::Tag &T) {
  return getCallToTagged(V, T) != nullptr;
}

//
// {is,get}CallToIsolatedFunction
//
const llvm::CallInst *getCallToIsolatedFunction(const llvm::Value *V);

llvm::CallInst *getCallToIsolatedFunction(llvm::Value *V);

inline bool isCallToIsolatedFunction(const llvm::Value *V) {
  return getCallToIsolatedFunction(V) != nullptr;
}
