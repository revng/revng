/// \file DeduplicateCollidingNames.cpp
/// Implementation of deduplication of colliding names.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <unordered_set>

#include "revng/Model/Binary.h"
#include "revng/Model/Pass/DeduplicateCollidingNames.h"
#include "revng/Model/Pass/RegisterModelPass.h"

#include "../NamespaceBuilder.h"

static RegisterModelPass R("deduplicate-colliding-names",
                           "This looks for groups of names are duplicates of "
                           "each other and appends arbitrary suffixes "
                           "to prevent collisions.",
                           model::deduplicateCollidingNames);

static Logger Log("name-deduplication");

struct GlobalEntry {
  llvm::StringRef FirstModelPath;
  uint64_t NextIndex = 0;
};

static llvm::DenseMap<llvm::StringRef, llvm::StringRef>
impl(std::vector<model::NamespaceEntry<false>> &Collisions,
     const model::Namespace<false> &CurrentNamespace,
     const model::Namespace<false> &ParentNamespace,
     const llvm::DenseMap<llvm::StringRef, llvm::StringRef> &Renames) {
  revng_assert(not Collisions.empty());
  llvm::StringRef Name = *Collisions.front().Name;

  uint64_t SuffixIndex = 1;
  llvm::StringRef FirstModelPath = Collisions.front().ModelPath;
  if (auto It = ParentNamespace.find(Name); It != ParentNamespace.end()) {
    FirstModelPath = It->second[0].ModelPath;
    ++SuffixIndex;
  }

  llvm::DenseMap<llvm::StringRef, llvm::StringRef> NewRenames;
  for (auto &Collision : Collisions) {
    std::string Current = *Collision.Name;
    if (SuffixIndex != 1) {
      Current += "_" + std::to_string(SuffixIndex);

      while (CurrentNamespace.contains(Current)
             or ParentNamespace.contains(Current) or Renames.count(Current)) {
        Current = *Collision.Name + "_" + std::to_string(++SuffixIndex);
      }

      revng_log(Log,
                "Appending `_" + std::to_string(SuffixIndex) + "` suffix to `"
                  + *Collision.Name + "` (`" + Collision.ModelPath
                  + "`) because it collides with `" + FirstModelPath.str()
                  + "`");

      *Collision.Name += "_" + std::to_string(SuffixIndex);
      auto [_, Success] = NewRenames.try_emplace(*Collision.Name,
                                                 Collision.ModelPath);
      revng_assert(Success);
    }

    ++SuffixIndex;
  }

  return NewRenames;
}

void model::deduplicateCollidingNames(TupleTree<model::Binary> &Binary) {
  llvm::Expected Namespaces = collectNamespaces(*Binary);
  if (not Namespaces) {
    // TODO: This shouldn't probably be an abort.
    revng_abort(revng::unwrapError(Namespaces.takeError()).c_str());
  }

  // Global names first,
  llvm::DenseMap<llvm::StringRef, llvm::StringRef> GlobalRenames;
  for (auto &&[_, List] : Namespaces->Global) {
    auto NewRenames = impl(List, Namespaces->Global, {}, GlobalRenames);
    GlobalRenames.insert(NewRenames.begin(), NewRenames.end());
  }

  // Then local ones.
  for (auto &&CurrentNamespace : Namespaces->Local)
    for (auto &&[_, List] : CurrentNamespace)
      impl(List, CurrentNamespace, Namespaces->Global, GlobalRenames);
}
