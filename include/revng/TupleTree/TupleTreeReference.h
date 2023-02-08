#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <compare>
#include <optional>
#include <type_traits>
#include <variant>

#include "llvm/ADT/StringRef.h"

#include "revng/ADT/Concepts.h"
#include "revng/Support/Assert.h"
#include "revng/Support/YAMLTraits.h"
#include "revng/TupleTree/TupleTreePath.h"
#include "revng/TupleTree/Visits.h"

template<typename T, typename RootType>
class TupleTreeReference {
public:
  using RootT = RootType;
  using RootVariant = std::variant<RootT *, const RootT *>;
  using TargetVariant = std::variant<T *, const T *>;

private:
  RootVariant Root = static_cast<RootT *>(nullptr);
  TupleTreePath Path;
  TargetVariant CachedTarget = static_cast<T *>(nullptr);

public:
  TupleTreeReference() = default;
  TupleTreeReference(ConstOrNot<RootT> auto *R, const TupleTreePath &P) :
    Root{ RootVariant{ R } },
    Path{ P },
    CachedTarget{ static_cast<T *>(nullptr) } {}

  TupleTreeReference(const TupleTreeReference &) = default;
  TupleTreeReference &operator=(const TupleTreeReference &) = default;
  TupleTreeReference(TupleTreeReference &) = default;
  TupleTreeReference &operator=(TupleTreeReference &) = default;

public:
  const RootT *getRoot() const {
    const auto GetPtrToConstRoot =
      [](const auto &RootPointer) -> const RootT * { return RootPointer; };
    return std::visit(GetPtrToConstRoot, Root);
  }

  void setRoot(ConstOrNot<RootT> auto *NewRoot) { Root = NewRoot; }

private:
  // Friend class that is allowed to manage the cached pointer to the target
  template<TupleTreeCompatible>
  friend class TupleTree;

  T *getCached() {
    if (std::holds_alternative<RootT *>(Root)) {
      revng_assert(std::holds_alternative<T *>(CachedTarget));
      return std::get<T *>(CachedTarget);
    } else if (std::holds_alternative<const RootT *>(Root)) {
      revng_abort("Called getCached() with const Root, use getCachedConst!");
    } else {
      revng_abort("Invalid root variant!");
    }
  }

  const T *getCachedConst() const {
    const auto GetConstPtrVisitor = [](const auto &Cached) -> const T * {
      return Cached;
    };
    return std::visit(GetConstPtrVisitor, CachedTarget);
  }

  bool isCached() const { return getCachedConst() != nullptr; }

  bool cacheTarget() {
    if (isValid()) {
      if (std::holds_alternative<const RootT *>(Root)) {
        CachedTarget = getConst();
      } else if (std::holds_alternative<RootT *>(Root)) {
        CachedTarget = get();
      } else {
        revng_abort("Invalid root variant!");
      }
    }
    return isCached();
  }

  void evictCachedTarget() { CachedTarget = static_cast<T *>(nullptr); }

public:
  static TupleTreeReference
  fromString(ConstOrNot<TupleTreeReference::RootT> auto *Root,
             llvm::StringRef Path) {
    std::optional<TupleTreePath> OptionalPath = stringAsPath<RootT>(Path);
    if (not OptionalPath.has_value())
      return TupleTreeReference{};
    return TupleTreeReference{ Root, *OptionalPath };
  }

  bool operator==(const TupleTreeReference &Other) const {
    // The paths are the same even if they are referred to different roots
    if (isCached() and Other.isCached())
      return getCachedConst() == Other.getCachedConst();
    return Path == Other.Path;
  }

  std::strong_ordering operator<=>(const TupleTreeReference &Other) const {
    // The paths are the same even if they are referred to different roots
    if (isCached() and Other.isCached())
      return getCachedConst() <=> Other.getCachedConst();
    return Path <=> Other.Path;
  }

public:
  std::string toString() const { return *pathAsString<RootT>(Path); }

  const TupleTreePath &path() const { return Path; }

private:
  bool hasNullRoot() const {
    const auto IsNullVisitor = [](const auto &Pointer) {
      return Pointer == nullptr;
    };
    return std::visit(IsNullVisitor, Root);
  }

  bool canGet() const {
    return Root.index() != std::variant_npos and not hasNullRoot();
  }

public:
  const T *get() const {
    revng_assert(canGet());

    if (isCached())
      return getCachedConst();

    if (Path.size() == 0)
      return nullptr;

    const auto GetByPathVisitor = [&Path = Path](const auto &RootPointer) {
      return getByPath<T>(Path, *RootPointer);
    };

    return std::visit(GetByPathVisitor, Root);
  }

  T *get() {
    revng_assert(canGet());

    if (isCached())
      return getCached();

    if (Path.size() == 0)
      return nullptr;

    if (std::holds_alternative<RootT *>(Root)) {
      return getByPath<T>(Path, *std::get<RootT *>(Root));
    } else if (std::holds_alternative<const RootT *>(Root)) {
      revng_abort("Called get() with const root, use getConst!");
    } else {
      revng_abort("Invalid root variant!");
    }
  }

  const T *getConst() const {
    revng_assert(canGet());

    if (isCached())
      return getCachedConst();

    if (Path.size() == 0)
      return nullptr;

    if (std::holds_alternative<const RootT *>(Root)) {
      return getByPath<T>(Path, *std::get<const RootT *>(Root));
    } else if (std::holds_alternative<RootT *>(Root)) {
      return getByPath<T>(Path, *std::get<RootT *>(Root));
    } else {
      revng_abort("Invalid root variant!");
    }
  }

  bool isConst() const { return std::holds_alternative<const RootT *>(Root); }

  bool empty() const debug_function { return Path.empty(); }

  bool isValid() const debug_function {
    if (not canGet() or Path.empty())
      return false;
    const T *TargetPointer = getConst();
    const T *CachedPointer = getCachedConst();
    if (not CachedPointer)
      return TargetPointer;
    return TargetPointer and TargetPointer == CachedPointer;
  }
};

template<StrictSpecializationOf<TupleTreeReference> T>
struct llvm::yaml::ScalarTraits<T> {

  static void output(const T &Obj, void *, llvm::raw_ostream &Out) {
    Out << Obj.toString();
  }

  static llvm::StringRef input(llvm::StringRef Path, void *, T &Obj) {
    // We temporarily initialize Root to nullptr, a post-processing phase will
    // take care of fixup these
    Obj = T::fromString(static_cast<typename T::RootT *>(nullptr), Path);
    return {};
  }

  static auto mustQuote(llvm::StringRef) {
    return llvm::yaml::QuotingType::Double;
  }
};

/// Specialization for the std::variant we have in TupleTreeReference
template<bool X, typename T>
inline void writeToLog(Logger<X> &This,
                       const std::variant<T *, const T *> &Var,
                       int Ignored) {
  if (Var.index() == std::variant_npos)
    writeToLog(This, llvm::StringRef("std::variant_npos"), Ignored);
  else if (std::holds_alternative<T *>(Var))
    writeToLog(This, std::get<T *>(Var), Ignored);
  else if (std::holds_alternative<const T *>(Var))
    writeToLog(This, std::get<const T *>(Var), Ignored);
}
