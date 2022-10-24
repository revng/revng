#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <array>
#include <memory>
#include <optional>
#include <set>
#include <type_traits>
#include <variant>
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/ADT/KeyedObjectContainer.h"
#include "revng/ADT/UpcastablePointer.h"
#include "revng/Support/Assert.h"
#include "revng/Support/Debug.h"
#include "revng/Support/YAMLTraits.h"
#include "revng/TupleTree/TupleTreeCompatible.h"
#include "revng/TupleTree/TupleTreePath.h"
#include "revng/TupleTree/TupleTreeReference.h"
#include "revng/TupleTree/Visits.h"

template<TupleTreeCompatible T>
class TupleTree {
private:
  std::unique_ptr<T> Root;
  bool AllReferencesAreCached = false;

public:
  TupleTree() : Root(new T), AllReferencesAreCached(false) {}

  // Allow expensive copy
  TupleTree(const TupleTree &Other) : Root(std::make_unique<T>()) {
    *this = Other;
  }
  TupleTree &operator=(const TupleTree &Other) {
    if (Other.get() == nullptr) {
      Root = nullptr;
      AllReferencesAreCached = false;
      return *this;
    }

    if (this != &Other) {
      *Root = *Other.Root;
      AllReferencesAreCached = false;
      initializeUncachedReferences();
    }
    return *this;
  }

  // Moving is fine
  TupleTree(TupleTree &&Other) { *this = std::move(Other); }
  TupleTree &operator=(TupleTree &&Other) {
    if (Other.get() == nullptr) {
      Root = nullptr;
      AllReferencesAreCached = false;

      Other.Root.reset();
      Other.AllReferencesAreCached = false;

      return *this;
    }

    if (this != &Other) {
      Root = std::move(Other.Root);
      AllReferencesAreCached = Other.AllReferencesAreCached;

      Other.Root.reset();
      Other.AllReferencesAreCached = false;
    }
    return *this;
  }

  template<StrictSpecializationOf<TupleTreeReference> TTR>
  void replaceReferences(const std::map<TTR, TTR> &Map) {
    auto Visitor = [&Map](TTR &Reference) {
      auto It = Map.find(Reference);
      if (It != Map.end())
        Reference = It->second;
    };
    visitReferences(Visitor);
    evictCachedReferences();
  }

public:
  static llvm::ErrorOr<TupleTree> deserialize(llvm::StringRef YAMLString) {
    TupleTree Result{};

    llvm::yaml::Input YAMLInput(YAMLString);

    auto MaybeRoot = revng::detail::deserializeImpl<T>(YAMLString);
    if (not MaybeRoot)
      return llvm::errorToErrorCode(MaybeRoot.takeError());

    *Result.Root = std::move(*MaybeRoot);

    // Update references to root
    Result.initializeReferences();

    return Result;
  }

  static llvm::ErrorOr<TupleTree> fromFile(const llvm::StringRef &Path) {
    auto MaybeBuffer = llvm::MemoryBuffer::getFile(Path);
    if (not MaybeBuffer)
      return MaybeBuffer.getError();

    return deserialize((*MaybeBuffer)->getBuffer());
  }

  llvm::Error toFile(const llvm::StringRef &Path) const {
    return ::serializeToFile(*Root, Path);
  }

public:
  template<typename S>
  void serialize(S &Stream) const {
    revng_assert(Root);

    ::serialize(Stream, *Root);
  }

  void serialize(std::string &Buffer) const {
    llvm::raw_string_ostream Stream(Buffer);
    serialize(Stream);
  }

public:
  const T *get() const noexcept { return Root.get(); }
  T *get() noexcept {
    revng_assert(not AllReferencesAreCached);
    return Root.get();
  }

  const T &operator*() const { return *Root; }
  T &operator*() {
    revng_assert(not AllReferencesAreCached);
    return *Root;
  }

  const T *operator->() const noexcept { return Root.operator->(); }
  T *operator->() noexcept {
    revng_assert(not AllReferencesAreCached);
    return Root.operator->();
  }

public:
  bool verify() const debug_function { return verifyReferences(); }

private:
  void initializeUncachedReferences() {
    visitReferences([this](auto &Element) {
      Element.Root = Root.get();
      Element.evictCachedTarget();
    });
    AllReferencesAreCached = false;
  }

public:
  void initializeReferences() {
    revng_assert(not AllReferencesAreCached);
    visitReferences([this](auto &Element) { Element.Root = Root.get(); });
  }

  void cacheReferences() {
    if (not AllReferencesAreCached)
      visitReferencesInternal([](auto &Element) { Element.cacheTarget(); });
    AllReferencesAreCached = true;
  }

  void evictCachedReferences() {
    if (AllReferencesAreCached)
      visitReferencesInternal([](auto &E) { E.evictCachedTarget(); });
    AllReferencesAreCached = false;
  }

private:
  template<typename L>
  void visitReferencesInternal(L &&InnerVisitor) {
    auto Visitor = [&InnerVisitor](auto &Element) {
      using type = std::remove_cvref_t<decltype(Element)>;
      if constexpr (StrictSpecializationOf<type, TupleTreeReference>)
        std::invoke(std::forward<L>(InnerVisitor), Element);
    };

    visitTupleTree(*Root, Visitor, [](auto &) {});
  }

public:
  template<typename L>
  void visitReferences(L &&InnerVisitor) {
    revng_assert(not AllReferencesAreCached);
    visitReferencesInternal(std::forward<L>(InnerVisitor));
  }

  template<typename L>
  void visitReferences(L &&InnerVisitor) const {
    auto Visitor = [&InnerVisitor](const auto &Element) {
      using type = std::remove_cvref_t<decltype(Element)>;
      if constexpr (StrictSpecializationOf<type, TupleTreeReference>)
        std::invoke(std::forward<L>(InnerVisitor), Element);
    };

    visitTupleTree(*Root, Visitor, [](auto) {});
  }

private:
  bool verifyReferences() const {
    bool Result = true;

    visitReferences([&Result, RootPointer = Root.get()](const auto &Element) {
      Result = Result and Element.getRoot() == RootPointer
               and not Element.isConst()
               and (Element.empty() or Element.isValid());
    });

    return Result;
  }
};
