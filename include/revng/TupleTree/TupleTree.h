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
#include "revng/ADT/KeyedObjectTraits.h"
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

public:
  TupleTree() : Root(new T) {}

  // Allow expensive copy
  TupleTree(const TupleTree &Other) : Root(std::make_unique<T>()) {
    *this = Other;
  }
  TupleTree &operator=(const TupleTree &Other) {
    if (this != &Other) {
      *Root = *Other.Root;
      initializeReferences();
    }
    return *this;
  }

  // Moving is fine
  TupleTree(TupleTree &&Other) = default;
  TupleTree &operator=(TupleTree &&Other) = default;

  template<IsTupleTreeReference TTR>
  void replaceReferences(const std::map<TTR, TTR> &Map) {
    auto Visitor = [&Map](TTR &Reference) {
      auto It = Map.find(Reference);
      if (It != Map.end())
        Reference = It->second;
    };
    visitReferences(Visitor);
  }

public:
  static llvm::ErrorOr<TupleTree> deserialize(llvm::StringRef YAMLString) {
    TupleTree Result;

    Result.Root = std::make_unique<T>();
    llvm::yaml::Input YAMLInput(YAMLString);

    auto MaybeRoot = detail::deserializeImpl<T>(YAMLString);
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
  T *get() noexcept { return Root.get(); }
  const T *get() const noexcept { return Root.get(); }
  T &operator*() { return *Root; }
  const T &operator*() const { return *Root; }
  T *operator->() noexcept { return Root.operator->(); }
  const T *operator->() const noexcept { return Root.operator->(); }

public:
  bool verify() const debug_function { return verifyReferences(); }

  void initializeReferences() {
    visitReferences([this](auto &Element) { Element.Root = Root.get(); });
  }

  template<typename L>
  void visitReferences(const L &InnerVisitor) {
    auto Visitor = [&InnerVisitor](auto &Element) {
      using type = std::remove_cvref_t<decltype(Element)>;
      if constexpr (IsTupleTreeReference<type>)
        InnerVisitor(Element);
    };

    visitTupleTree(*Root, Visitor, [](auto &) {});
  }

  template<typename L>
  void visitReferences(const L &InnerVisitor) const {
    auto Visitor = [&InnerVisitor](const auto &Element) {
      using type = std::remove_cvref_t<decltype(Element)>;
      if constexpr (IsTupleTreeReference<type>)
        InnerVisitor(Element);
    };

    visitTupleTree(*Root, Visitor, [](auto) {});
  }

private:
  bool verifyReferences() const {
    bool Result = true;

    visitReferences([&Result, this](const auto &Element) {
      const auto SameRoot = [&]() {
        const auto GetPtrToConstRoot =
          [](const auto &RootPointer) -> const T * { return RootPointer; };

        return std::visit(GetPtrToConstRoot, Element.Root) == Root.get();
      };
      Result = Result and SameRoot();
    });

    return Result;
  }
};
