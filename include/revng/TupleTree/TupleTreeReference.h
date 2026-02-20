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

private:
  static inline T *const NoCachedTarget = static_cast<T *>(nullptr);
  static inline T
    *const CachingDisabled = reinterpret_cast<T *>(static_cast<uintptr_t>(-1));

private:
  RootVariant Root = static_cast<RootT *>(nullptr);
  TupleTreePath Path;
  // This field is used for lazily caching the object pointed by the reference,
  // after the path from the root object has been resolved. Since this can be
  // often set while the object is `const` it's required for it to be mutable.
  // Moreover the const-ness of the returned pointer is dictated by the
  // const-ness of `Root`. All public interfaces (`get*()`) respect the
  // const-ness by checking the const-ness of `Root`.
  mutable T *CachedTarget = CachingDisabled;

public:
  TupleTreeReference() = default;
  TupleTreeReference(ConstOrNot<RootT> auto *R, const TupleTreePath &P) :
    Root{ RootVariant{ R } }, Path{ P }, CachedTarget{ CachingDisabled } {}

  TupleTreeReference(const TupleTreeReference &) = default;
  TupleTreeReference &operator=(const TupleTreeReference &) = default;
  TupleTreeReference(TupleTreeReference &&) = default;
  TupleTreeReference &operator=(TupleTreeReference &&) = default;

public:
  const RootT *getRoot() const {
    const auto GetPtrToConstRoot =
      [](const auto &RootPointer) -> const RootT * { return RootPointer; };
    return std::visit(GetPtrToConstRoot, Root);
  }

  void setRoot(ConstOrNot<RootT> auto *NewRoot) {
    Root = NewRoot;
    if (isCachingEnabled())
      CachedTarget = NoCachedTarget;
  }

private:
  // Friend class that is allowed to manage the cached pointer to the target
  template<TupleTreeCompatible>
  friend class TupleTree;

  bool isCached() const {
    return CachedTarget != NoCachedTarget and CachedTarget != CachingDisabled;
  }

  void enableCaching() {
    if (CachedTarget == CachingDisabled)
      CachedTarget = NoCachedTarget;
  }

  void disableCaching() { CachedTarget = CachingDisabled; }

  bool isCachingEnabled() const { return CachedTarget != CachingDisabled; }

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
    return Path == Other.Path;
  }

  std::strong_ordering operator<=>(const TupleTreeReference &Other) const {
    // The paths are the same even if they are referred to different roots
    return Path <=> Other.Path;
  }

public:
  std::string toString() const { return *pathAsString<RootT>(Path); }

  void dump() const debug_function { dbg << toString() << "\n"; }

  const TupleTreePath &path() const { return Path; }

private:
  bool hasValidRoot() const {
    const auto IsNullVisitor = [](const auto &Pointer) {
      return Pointer != nullptr;
    };
    return std::visit(IsNullVisitor, Root);
  }

  template<ConstOrNot<T> U>
  U *cacheAndGet(U *Ptr) const {
    revng_assert(Ptr != CachingDisabled);
    // The caller might have legitimately returned nullptr, which we used as a
    // placeholder, in this case return the pointer directly and don't cache it
    // to avoid issues.
    if (Ptr == NoCachedTarget)
      return Ptr;

    // Here we assume that the caller has called `isCached` beforehand
    if (isCachingEnabled())
      CachedTarget = const_cast<T *>(Ptr);

    return Ptr;
  }

public:
  const T *getConst() const {
    revng_assert(hasValidRoot());

    if (Path.size() == 0)
      return nullptr;

    if (isCached())
      return CachedTarget;

    const auto GetByPathVisitor = [&Path = Path](const auto &RootPointer) {
      return static_cast<const T *>(getByPath<T>(Path, *RootPointer));
    };

    return cacheAndGet(std::visit(GetByPathVisitor, Root));
  }

  T *get() {
    revng_assert(hasValidRoot());

    if (isConst())
      revng_abort("Called get() with const root, use getConst!");

    if (Path.size() == 0)
      return nullptr;

    if (isCached())
      return CachedTarget;

    return cacheAndGet(getByPath<T>(Path, *std::get<RootT *>(Root)));
  }

  const T *get() const { return getConst(); }

  bool isConst() const { return std::holds_alternative<const RootT *>(Root); }

  bool empty() const debug_function { return Path.empty(); }

  bool isValid() const debug_function {
    if (not hasValidRoot() or Path.empty())
      return false;

    const T *TargetPointer = getConst();
    if (CachedTarget == NoCachedTarget or CachedTarget == CachingDisabled)
      return TargetPointer != nullptr;
    else
      return TargetPointer != nullptr and TargetPointer == CachedTarget;
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
template<typename T>
inline void
writeToLog(Logger &This, const std::variant<T *, const T *> &Var, int Ignored) {
  if (Var.index() == std::variant_npos)
    writeToLog(This, llvm::StringRef("std::variant_npos"), Ignored);
  else if (std::holds_alternative<T *>(Var))
    writeToLog(This, std::get<T *>(Var), Ignored);
  else if (std::holds_alternative<const T *>(Var))
    writeToLog(This, std::get<const T *>(Var), Ignored);
}
