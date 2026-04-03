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

#include "revng/ADT/Concepts.h"
#include "revng/ADT/KeyedObjectContainer.h"
#include "revng/ADT/UpcastablePointer.h"
#include "revng/Support/Assert.h"
#include "revng/Support/Debug.h"
#include "revng/Support/YAMLTraits.h"
#include "revng/TupleTree/Tracking.h"
#include "revng/TupleTree/TupleTreeCompatible.h"
#include "revng/TupleTree/TupleTreePath.h"
#include "revng/TupleTree/TupleTreeReference.h"
#include "revng/TupleTree/Visits.h"

template<typename T>
struct DisableTracking {
  const T *TrackedObject = nullptr;

public:
  DisableTracking(const T &TrackedObject) : TrackedObject(&TrackedObject) {
    // Since the model classes may have been generated either with or without
    // tracking, DisableTracking should do nothing if the concept returns false.
    if constexpr (T::HasTracking)
      revng::Tracking::push(*this->TrackedObject);
  }

  DisableTracking(const DisableTracking &Other) = delete;
  DisableTracking &operator=(const DisableTracking &Other) = delete;

  DisableTracking(DisableTracking &&Other) {
    TrackedObject = Other.TrackedObject;
    Other.TrackedObject = nullptr;
  }
  DisableTracking &operator=(DisableTracking &&Other) {
    if (this == &Other) {
      return *this;
    }

    onDestruction();
    TrackedObject = Other.TrackedObject;
    Other.TrackedObject = nullptr;

    return *this;
  }

  ~DisableTracking() { onDestruction(); }

private:
  void onDestruction() {
    if constexpr (T::HasTracking) {
      if (TrackedObject != nullptr) {
        revng::Tracking::pop(*TrackedObject);
      }
    }
    TrackedObject = nullptr;
  }
};

namespace detail {

template<typename T>
concept HasVersion = requires {
  requires static_cast<int>(TupleLikeTraits<T>::Fields::Version) == 0;
};

}

template<TupleTreeCompatible T>
class TupleTree {
private:
  std::unique_ptr<T> Root;
  bool CachingEnabled = false;

public:
  TupleTree() : Root(new T), CachingEnabled(false) {}

  // Allow expensive copy
  TupleTree(const TupleTree &Other) : Root(std::make_unique<T>()) {
    *this = Other;
  }
  TupleTree &operator=(const TupleTree &Other) {
    if (Other.get() == nullptr) {
      Root = nullptr;
      CachingEnabled = false;
      return *this;
    }

    if (this != &Other) {
      *Root = *Other.Root;
      CachingEnabled = Other.CachingEnabled;
      initializeReferences();
    }
    return *this;
  }

  // Moving is fine
  TupleTree(TupleTree &&Other) { *this = std::move(Other); }
  TupleTree &operator=(TupleTree &&Other) {
    if (Other.Root == nullptr) {
      Root = nullptr;
      CachingEnabled = false;

      Other.Root.reset();
      Other.CachingEnabled = false;

      return *this;
    }

    if (this != &Other) {
      Root = std::move(Other.Root);
      CachingEnabled = Other.CachingEnabled;

      Other.Root.reset();
      Other.CachingEnabled = false;
    }
    return *this;
  }

  template<StrictSpecializationOf<TupleTreeReference> TTR>
  void replaceReferences(const std::map<TTR, TTR> &Map) {
    auto Visitor = [&Map]<typename TTRA>(TTRA &Reference) {
      // Here TTRA can be any TupleTreeReference<X, Binary>, actually check
      // that it is of the type we want to replace
      if constexpr (std::is_same_v<TTRA, TTR>) {
        auto It = Map.find(Reference);
        if (It != Map.end())
          Reference = It->second;
      }
    };
    visitReferencesInternal(Visitor);
    disableReferenceCaching();
  }

  template<StrictSpecializationOf<TupleTreeReference> TTR,
           std::predicate<const TTR &> PredicateType>
  void replaceReferencesIf(const TTR &NewReference, PredicateType &&Predicate) {
    auto Visitor = [&Predicate, &NewReference]<typename TTRA>(TTRA &Reference) {
      if constexpr (std::is_same_v<TTR, TTRA>) {
        if (Predicate(Reference))
          Reference = NewReference;
      }
    };
    visitReferencesInternal(Visitor);
    disableReferenceCaching();
  }

public:
  static llvm::Expected<TupleTree> fromString(llvm::StringRef YAMLString) {
    TupleTree Result;

    auto MaybeRoot = revng::detail::fromStringImpl<T>(YAMLString);
    if (not MaybeRoot)
      return MaybeRoot.takeError();

    *Result.Root = std::move(*MaybeRoot);

    if constexpr (detail::HasVersion<T>) {
      DisableTracking Guard(*Result.Root);
      if (Result.Root->Version() == 0) {
        Result.Root->Version() = T::SchemaVersion;
      }
    }

    // Update references to root
    Result.initializeReferences();

    return Result;
  }

  static llvm::Expected<TupleTree>
  fromFileOrSTDIN(const llvm::StringRef &Path) {
    auto MaybeBuffer = llvm::MemoryBuffer::getFileOrSTDIN(Path);
    if (not MaybeBuffer)
      return llvm::errorCodeToError(MaybeBuffer.getError());

    return fromString((*MaybeBuffer)->getBuffer());
  }

  static llvm::Expected<TupleTree> fromFile(const llvm::StringRef &Path) {
    auto MaybeBuffer = llvm::MemoryBuffer::getFile(Path);
    if (not MaybeBuffer)
      return llvm::errorCodeToError(MaybeBuffer.getError());

    return fromString((*MaybeBuffer)->getBuffer());
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
    revng_assert(not CachingEnabled);
    return Root.get();
  }

  const T &operator*() const { return *Root; }
  T &operator*() {
    revng_assert(not CachingEnabled);
    return *Root;
  }

  const T *operator->() const noexcept { return Root.operator->(); }
  T *operator->() noexcept {
    revng_assert(not CachingEnabled);
    return Root.operator->();
  }

public:
  bool verify() const debug_function { return verifyReferences(false); }
  void assertValid() const { verifyReferences(true); }

public:
  void initializeReferences() {
    DisableTracking Guard(*Root);
    visitReferencesInternal([this](auto &Element) {
      Element.setRoot(Root.get());
      if (CachingEnabled)
        Element.enableCaching();
      else
        Element.disableCaching();
    });
  }

  void enableReferenceCaching() {
    DisableTracking Guard(*Root);
    if (not CachingEnabled)
      visitReferencesInternal([](auto &E) { E.enableCaching(); });
    CachingEnabled = true;
  }

  void disableReferenceCaching() {
    DisableTracking Guard(*Root);
    if (CachingEnabled)
      visitReferencesInternal([](auto &E) { E.disableCaching(); });
    CachingEnabled = false;
  }

  bool isReferenceCachingEnabled() { return CachingEnabled; }

  template<typename Pre, typename Post>
  void visit(Pre PreCallable, Post PostCallable) const {
    using PreVisitor = typename TupleTreeVisitor<T>::template ConstVisitor<Pre>;
    PreVisitor PreInstance(PreCallable);
    using PostVisitor = typename TupleTreeVisitor<T>::template ConstVisitor<
      Post>;
    PostVisitor PostInstance(PostCallable);
    visitImpl(PreInstance, PostInstance);
  }

  template<typename Pre, typename Post>
  void visit(Pre PreCallable, Post PostCallable) {
    using PreVisitor = typename TupleTreeVisitor<T>::template Visitor<Pre>;
    PreVisitor PreInstance(PreCallable);
    using PostVisitor = typename TupleTreeVisitor<T>::template Visitor<Post>;
    PostVisitor PostInstance(PostCallable);
    visitImpl(PreInstance, PostInstance);
  }

private:
  void visitImpl(typename TupleTreeVisitor<T>::ConstVisitorBase &Pre,
                 typename TupleTreeVisitor<T>::ConstVisitorBase &Post) const;

  void visitImpl(typename TupleTreeVisitor<T>::VisitorBase &Pre,
                 typename TupleTreeVisitor<T>::VisitorBase &Post);

  template<typename L>
  void visitReferencesInternal(L &&InnerVisitor) {
    auto Visitor = [&InnerVisitor](auto &Element) {
      using type = std::remove_cvref_t<decltype(Element)>;
      if constexpr (StrictSpecializationOf<type, TupleTreeReference>)
        std::invoke(std::forward<L>(InnerVisitor), Element);
    };

    visit(Visitor, [](auto &) {});
  }

public:
  template<typename L>
  void visitReferences(L &&InnerVisitor) {
    revng_assert(not CachingEnabled);
    visitReferencesInternal(std::forward<L>(InnerVisitor));
  }

  template<typename L>
  void visitReferences(L &&InnerVisitor) const {
    auto Visitor = [&InnerVisitor](const auto &Element) {
      using type = std::remove_cvref_t<decltype(Element)>;
      if constexpr (StrictSpecializationOf<type, TupleTreeReference>)
        std::invoke(std::forward<L>(InnerVisitor), Element);
    };

    visit(Visitor, [](auto) {});
  }

private:
  bool verifyReferences(bool Assert) const;
};
