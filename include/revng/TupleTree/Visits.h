#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <cstdint>
#include <tuple>

#include "revng/ADT/KeyedObjectContainer.h"
#include "revng/ADT/UpcastablePointer.h"
#include "revng/Support/YAMLTraits.h"
#include "revng/TupleTree/TupleLikeTraits.h"
#include "revng/TupleTree/TupleTreeCompatible.h"
#include "revng/TupleTree/TupleTreePath.h"

template<typename RootT>
struct TupleTreeVisitor;

//
// visitTupleTree implementation
//

namespace tupletree::detail {

template<size_t I = 0, typename Visitor, typename T>
void visitTuple(Visitor &V, T &Obj) {
  if constexpr (I < std::tuple_size_v<T>) {
    // Visit the field
    visitTupleTree(V, get<I>(Obj));

    // Visit next element in tuple
    visitTuple<I + 1>(V, Obj);
  }
}

} // namespace tupletree::detail

// UpcastablePointerLike-like
template<typename Visitor, UpcastablePointerLike T>
void visitTupleTree(Visitor &V, T &Obj) {
  upcast(Obj, [&V](auto &Upcasted) { visitTupleTree(V, Upcasted); });
}

// Tuple-like
template<typename Visitor, TupleSizeCompatible T>
void visitTupleTree(Visitor &V, T &Obj) {
  V.PreVisit(Obj);
  tupletree::detail::visitTuple(V, Obj);
  V.PostVisit(Obj);
}

// Container-like
template<typename Visitor, KeyedObjectContainer T>
void visitTupleTree(Visitor &V, T &Obj) {
  V.PreVisit(Obj);
  using value_type = typename T::value_type;
  for (value_type &Element : Obj) {
    visitTupleTree(V, Element);
  }
  V.PostVisit(Obj);
}

// All the others
template<typename Visitor, NotTupleTreeCompatible T>
void visitTupleTree(Visitor &V, T &Element) {
  V.PreVisit(Element);
  V.PostVisit(Element);
}

template<typename Pre, typename Post, typename T>
void visitTupleTree(T &Element,
                    const Pre &PreVisitor,
                    const Post &PostVisitor) {
  struct {
    const Pre &PreVisit;
    const Post &PostVisit;
  } Visitor{ PreVisitor, PostVisitor };
  visitTupleTree(Visitor, Element);
}

//
// tupleIndexByName
//
template<TraitedTupleLike T, size_t I = 0>
size_t tupleIndexByName(llvm::StringRef Name) {
  if constexpr (I < std::tuple_size_v<T>) {
    llvm::StringRef ThisName = TupleLikeTraits<T>::FieldNames[I];
    if (Name == ThisName)
      return I;
    else
      return tupleIndexByName<T, I + 1>(Name);
  } else {
    return -1;
  }
}

//
// getByKey
//
namespace tupletree::detail {

template<typename ResultT, size_t I = 0, typename RootT, typename KeyT>
ResultT *getByKeyTuple(RootT &M, KeyT Key) {
  if constexpr (I < std::tuple_size_v<RootT>) {
    if (I == Key) {
      using tuple_element = typename std::tuple_element<I, RootT>::type;
      revng_assert((std::is_same_v<tuple_element, ResultT>) );
      return reinterpret_cast<ResultT *>(&get<I>(M));
    } else {
      return getByKeyTuple<ResultT, I + 1>(M, Key);
    }
  } else {
    return nullptr;
  }
}

} // namespace tupletree::detail

template<typename ResultT, UpcastablePointerLike RootT, typename KeyT>
ResultT getByKey(RootT &M, KeyT Key) {
  auto Dispatcher = [&](auto &Upcasted) { return getByKey(Upcasted, Key); };
  return upcast(M, Dispatcher, ResultT{});
}

template<typename ResultT, TupleSizeCompatible RootT, typename KeyT>
ResultT getByKey(RootT &M, KeyT Key) {
  return tupletree::detail::getByKeyTuple<ResultT>(M, Key);
}

template<typename ResultT, KeyedObjectContainer RootT, typename KeyT>
ResultT *getByKey(RootT &M, KeyT Key) {
  for (auto &Element : M) {
    using KOT = KeyedObjectTraits<std::remove_reference_t<decltype(Element)>>;
    if (KOT::key(Element) == Key)
      return &Element;
  }
  return nullptr;
}

//
// callOnPathSteps (no instance)
//
template<TupleSizeCompatible RootT, typename Visitor>
bool callOnPathSteps(Visitor &V, llvm::ArrayRef<TupleTreeKeyWrapper> Path);

template<NotTupleTreeCompatible T, typename Visitor>
bool callOnPathSteps(Visitor &, llvm::ArrayRef<TupleTreeKeyWrapper>) {
  //"Unandled call on step";
  return false;
}

template<NotUpcastablePointerLike T, typename Visitor>
bool callOnPathStepsImpl(Visitor &V, llvm::ArrayRef<TupleTreeKeyWrapper> Path) {
  return callOnPathSteps<T, Visitor>(V, Path.slice(1));
}

template<UpcastablePointerLike RootT, typename Visitor>
bool callOnPathStepsImpl(Visitor &V, llvm::ArrayRef<TupleTreeKeyWrapper> Path) {
  auto Dispatcher = [&](auto &Upcasted) -> bool {
    return callOnPathStepsImpl<std::decay_t<decltype(Upcasted)>>(V, Path);
  };
  using KOT = KeyedObjectTraits<RootT>;
  using key_type = decltype(KOT::key(std::declval<RootT>()));
  auto TargetKey = Path[0].get<key_type>();
  // TODO: in case of nullptr we should abort
  auto Temporary = KeyedObjectTraits<RootT>::fromKey(TargetKey);
  return upcast(Temporary, Dispatcher, false);
}

template<KeyedObjectContainer RootT, typename Visitor>
bool callOnPathSteps(Visitor &V, llvm::ArrayRef<TupleTreeKeyWrapper> Path) {
  using value_type = typename RootT::value_type;
  using KOT = KeyedObjectTraits<value_type>;
  using key_type = decltype(KOT::key(std::declval<value_type>()));
  auto TargetKey = Path[0].get<key_type>();

  V.template visitContainerElement<RootT>(TargetKey);
  if (Path.size() > 1) {
    return callOnPathStepsImpl<value_type>(V, Path);
  }

  return true;
}

namespace tupletree::detail {

template<typename RootT, size_t I = 0, typename Visitor>
bool callOnPathStepsTuple(Visitor &V,
                          llvm::ArrayRef<TupleTreeKeyWrapper> Path) {
  if constexpr (I < std::tuple_size_v<RootT>) {
    if (Path.size() == 0)
      return true;

    if (Path[0].get<size_t>() == I) {
      using next_type = typename std::tuple_element<I, RootT>::type;
      V.template visitTupleElement<RootT, I>();
      if (Path.size() > 1) {
        return callOnPathStepsImpl<next_type>(V, Path);
      }
    } else {
      return callOnPathStepsTuple<RootT, I + 1>(V, Path);
    }
  }

  return true;
}

} // namespace tupletree::detail

template<TupleSizeCompatible RootT, typename Visitor>
bool callOnPathSteps(Visitor &V, llvm::ArrayRef<TupleTreeKeyWrapper> Path) {
  return tupletree::detail::callOnPathStepsTuple<RootT>(V, Path);
}

//
// callOnPathSteps (with instance)
//

namespace tupletree::detail {

template<NotTupleTreeCompatible T, typename Visitor>
bool callOnPathSteps(Visitor &,
                     llvm::ArrayRef<TupleTreeKeyWrapper>,
                     T &,
                     const llvm::StringRef) {
  // Unandled call on step
  return false;
}

template<size_t I = 0, typename RootT, typename Visitor>
bool callOnPathStepsTuple(Visitor &V,
                          llvm::ArrayRef<TupleTreeKeyWrapper> Path,
                          RootT &M,
                          const llvm::StringRef FullPath) {
  if constexpr (I < std::tuple_size_v<RootT>) {
    if (Path[0].get<size_t>() == I) {
      using next_type = typename std::tuple_element<I, RootT>::type;
      next_type &Element = get<I>(M);
      V.template visitTupleElement<RootT, I>(Element);
      if (Path.size() > 1) {
        return callOnPathSteps(V, Path.slice(1), Element, FullPath);
      }
    } else {
      return callOnPathStepsTuple<I + 1>(V, Path, M, FullPath);
    }
  }

  return true;
}

} // namespace tupletree::detail

template<UpcastablePointerLike RootT, typename Visitor>
bool callOnPathSteps(Visitor &V,
                     llvm::ArrayRef<TupleTreeKeyWrapper> Path,
                     RootT &M,
                     const llvm::StringRef FullPath) {
  auto Dispatcher = [&](auto &Upcasted) -> bool {
    return callOnPathStepsTuple(V, Path, Upcasted, FullPath);
  };
  // TODO: in case of nullptr we should abort
  return upcast(M, Dispatcher, false);
}

template<TupleSizeCompatible RootT, typename Visitor>
bool callOnPathSteps(Visitor &V,
                     llvm::ArrayRef<TupleTreeKeyWrapper> Path,
                     RootT &M,
                     const llvm::StringRef FullPath) {
  return tupletree::detail::callOnPathStepsTuple(V, Path, M, FullPath);
}

template<KeyedObjectContainer RootT, typename Visitor>
bool callOnPathSteps(Visitor &V,
                     llvm::ArrayRef<TupleTreeKeyWrapper> Path,
                     RootT &M,
                     const llvm::StringRef FullPath) {
  using value_type = typename RootT::value_type;
  using KOT = KeyedObjectTraits<value_type>;
  using key_type = decltype(KOT::key(std::declval<value_type>()));
  auto TargetKey = Path[0].get<key_type>();

  auto It = M.find(TargetKey);
  if (It == M.end()) {
    return false;
  }

  auto *Matching = &*It;

  V.template visitContainerElement<RootT>(TargetKey, *Matching);
  if (Path.size() > 1) {
    return callOnPathSteps(V, Path.slice(1), *Matching, FullPath);
  }

  return true;
}

//
// callByPath (no instance)
//
namespace tupletree::detail {

template<typename Visitor>
struct CallByPathVisitor {
  size_t PathSize;
  Visitor &V;

  template<typename T, int I>
  void visitTupleElement() {
    --PathSize;
    if (PathSize == 0)
      V.template visitTupleElement<T, I>();
  }

  template<typename T, typename KeyT>
  void visitContainerElement(KeyT Key) {
    PathSize -= 1;
    if (PathSize == 0)
      V.template visitContainerElement<T>(Key);
  }
};

} // namespace tupletree::detail

template<typename RootT, typename Visitor>
bool callByPath(Visitor &V, const TupleTreePath &Path) {
  using namespace tupletree::detail;
  CallByPathVisitor<Visitor> CBPV{ Path.size(), V };
  return callOnPathSteps<RootT>(CBPV, Path.toArrayRef());
}

//
// callByPath (with instance)
//
namespace tupletree::detail {

template<typename Visitor>
struct CallByPathVisitorWithInstance {
  size_t PathSize;
  Visitor &V;

  template<typename T, size_t I, typename K>
  void visitTupleElement(K &Element) {
    --PathSize;
    if (PathSize == 0)
      V.template visitTupleElement<T, I>(Element);
  }

  template<typename T,
           StrictSpecializationOf<UpcastablePointer> K,
           typename KeyT>
  void visitContainerElement(KeyT Key, K &Element) {
    PathSize -= 1;
    if (PathSize == 0)
      V.template visitContainerElement<T>(Key, *Element.get());
  }

  template<typename T, typename K, typename KeyT>
    requires(not StrictSpecializationOf<K, UpcastablePointer>)
  void visitContainerElement(KeyT Key, K &Element) {
    PathSize -= 1;
    if (PathSize == 0)
      V.template visitContainerElement<T>(Key, Element);
  }
};

} // namespace tupletree::detail

template<typename RootT, typename Visitor>
bool callByPath(Visitor &V, const TupleTreePath &Path, RootT &M) {
  return callByPath(V, Path, M, "");
}

template<typename RootT, typename Visitor>
bool callByPath(Visitor &V,
                const TupleTreePath &Path,
                RootT &M,
                const llvm::StringRef OriginalPath) {
  using namespace tupletree::detail;
  CallByPathVisitorWithInstance<Visitor> CBPV{ Path.size(), V };
  return callOnPathSteps(CBPV, Path.toArrayRef(), M, OriginalPath);
}

//
// getByPath
//
template<typename ResultT, typename RootT>
ResultT *getByPath(const TupleTreePath &Path, RootT &M);

//
// pathAsString
//

namespace tupletree::detail {

class DumpPathVisitor {
private:
  llvm::raw_string_ostream Stream;

public:
  DumpPathVisitor(std::string &Result) : Stream(Result) {}

  template<TraitedTupleLike T, int I>
  void visitTupleElement() {
    Stream << "/" << TupleLikeTraits<T>::FieldNames[I];
  }

  template<typename T, typename KeyT>
  void visitContainerElement(KeyT Key) {
    Stream << "/" << getNameFromYAMLScalar(Key);
  }
};

} // namespace tupletree::detail

template<typename T>
std::optional<std::string> pathAsString(const TupleTreePath &Path) {
  std::string Result;
  {
    tupletree::detail::DumpPathVisitor PV(Result);
    if (not callOnPathSteps<T>(PV, Path.toArrayRef())) {
      return {};
    }
  }
  return Result;
}

class PathMatcher {
private:
  TupleTreePath Path;
  std::vector<size_t> Free;

private:
  PathMatcher() = default;

public:
  template<typename T>
  static std::optional<PathMatcher> create(llvm::StringRef Path);

public:
  const TupleTreePath &path() const { return Path; }

public:
  template<typename... Ts>
  TupleTreePath apply(Ts... Args) const {
    revng_assert(sizeof...(Args) == Free.size());
    TupleTreePath Result = Path;
    applyImpl<0, Ts...>(Result, Args...);
    return Result;
  }

  template<typename... Args>
  std::optional<std::tuple<Args...>> match(const TupleTreePath &Search) {
    revng_assert(sizeof...(Args) == Free.size());

    if (Path.size() != Search.size())
      return {};

    //
    // Check non-variable parts match
    //
    std::vector<size_t> Terminator{ Path.size() };
    size_t LastEnd = 0;
    for (auto Index : llvm::concat<size_t>(Free, Terminator)) {
      for (size_t I = LastEnd; I < Index; ++I) {
        if (Search[I] != Path[I])
          return {};
      }

      LastEnd = Index + 1;
    }

    //
    // Check variable parts match
    //
    for (auto I : Free)
      if (not Path[I].matches(Search[I]))
        return {};

    //
    // Compute result
    //
    std::tuple<Args...> Result;
    extractKeys(Search, Result);
    return Result;
  }

private:
  template<size_t I, typename T>
  void depositKey(TupleTreePath &Result, T Arg) const {
    auto Index = Free.at(I);
    Result[Index] = ConcreteTupleTreeKeyWrapper<T>(Arg);
  }

  template<size_t I, typename T>
  void applyImpl(TupleTreePath &Result, T Arg) const {
    depositKey<I>(Result, Arg);
  }

  template<size_t I, typename T, typename... Ts>
  void applyImpl(TupleTreePath &Result, T Arg, Ts... Args) const {
    depositKey<I>(Result, Arg);
    applyImpl<I + 1, Ts...>(Result, Args...);
  }

  template<typename T, size_t I = 0>
  void extractKeys(const TupleTreePath &Search, T &Tuple) const {
    if constexpr (I < std::tuple_size_v<T>) {
      using element = std::tuple_element_t<I, T>;
      std::get<I>(Tuple) = Search[Free[I]].get<element>();
      extractKeys<T, I + 1>(Search, Tuple);
    }
  }

private:
  template<TraitedTupleLike T, size_t I = 0>
  static bool visitTuple(llvm::StringRef Current,
                         llvm::StringRef Rest,
                         PathMatcher &Result);

  template<UpcastablePointerLike T>
  static bool
  dispatchToConcreteType(llvm::StringRef String, PathMatcher &Result);

  template<UpcastablePointerLike T>
  static bool visitTupleTreeNode(llvm::StringRef String, PathMatcher &Result);

  template<TupleSizeCompatible T>
  static bool visitTupleTreeNode(llvm::StringRef String, PathMatcher &Result);

  template<KeyedObjectContainer T>
  static bool visitTupleTreeNode(llvm::StringRef String, PathMatcher &Result);

  template<NotTupleTreeCompatible T>
  static bool visitTupleTreeNode(llvm::StringRef Path, PathMatcher &Result);
};

template<UpcastablePointerLike P, typename L>
void invokeBySerializedKey(llvm::StringRef SerializedKey, L &&Callable) {
  using element_type = std::remove_reference_t<decltype(*std::declval<P>())>;
  using KOT = KeyedObjectTraits<element_type>;
  using KeyT = decltype(KOT::key(std::declval<element_type>()));
  auto Key = getValueFromYAMLScalar<KeyT>(SerializedKey);
  invokeByKey<P, KeyT, L>(Key, std::forward<L>(Callable));
}

template<UpcastablePointerLike T>
bool PathMatcher::dispatchToConcreteType(llvm::StringRef String,
                                         PathMatcher &Result) {

  auto Parts = String.split('/');

  bool Res = false;
  auto Dispatch = [&]<typename Upcasted>(const Upcasted *Arg) {
    Res = PathMatcher::visitTupleTreeNode<Upcasted>(Parts.second, Result);
  };

  invokeBySerializedKey<T>(Parts.first, Dispatch);

  return Res;
}

template<UpcastablePointerLike T>
bool PathMatcher::visitTupleTreeNode(llvm::StringRef String,
                                     PathMatcher &Result) {
  return dispatchToConcreteType<T>(String, Result);
}

template<TupleSizeCompatible T>
bool PathMatcher::visitTupleTreeNode(llvm::StringRef String,
                                     PathMatcher &Result) {
  if (String.size() == 0)
    return true;

  auto [Before, After] = String.split('/');
  return visitTuple<T>(Before, After, Result);
}

template<KeyedObjectContainer T>
bool PathMatcher::visitTupleTreeNode(llvm::StringRef String,
                                     PathMatcher &Result) {
  if (String.size() == 0)
    return true;

  auto [Before, After] = String.split('/');

  using Key = std::remove_cv_t<typename T::key_type>;
  using Value = typename T::value_type;

  if constexpr (StrictSpecializationOf<Value, UpcastablePointer>) {
    auto [PreDash, PostDash] = Before.split("-");
    if (PreDash == "*") {
      // Mark as free
      Result.Free.push_back(Result.Path.size());

      //
      // Extract the Kind of the abstract type in the UpcastableType
      //

      // Get the kind type for the abstract type
      // TODO: add using for model::Type's Kind
      using Kind = typename Value::element_type::KindType;

      // Extract Kind from "Kind-*" and deserialize it
      Kind MatcherKind = getValueFromYAMLScalar<Kind>(PostDash);
      static_assert(std::is_enum_v<std::decay_t<Kind>>);

      // Push in Path a Key initializing only the first field (the kind)
      Key Component;
      auto &TheKind = std::get<std::tuple_size_v<Key> - 1>(Component);
      using KindType = decltype(TheKind);
      static_assert(std::is_enum_v<std::decay_t<KindType>>);
      TheKind = MatcherKind;
      Result.Path.emplace_back<Key, true>(Component);
    } else {
      Result.Path.push_back(getValueFromYAMLScalar<Key>(Before));
    }

    return dispatchToConcreteType<Value>(String, Result);
  } else {
    if (Before == "*") {
      Result.Free.push_back(Result.Path.size());
      Result.Path.emplace_back<Key>();
    } else {
      Result.Path.push_back(getValueFromYAMLScalar<Key>(Before));
    }

    return visitTupleTreeNode<Value>(After, Result);
  }
}

template<NotTupleTreeCompatible T>
bool PathMatcher::visitTupleTreeNode(llvm::StringRef Path,
                                     PathMatcher &Result) {
  return Path.size() == 0;
}

template<TraitedTupleLike T, size_t I>
bool PathMatcher::visitTuple(llvm::StringRef Current,
                             llvm::StringRef Rest,
                             PathMatcher &Result) {
  if constexpr (I < std::tuple_size_v<T>) {
    if (TupleLikeTraits<T>::FieldNames[I] == Current) {
      Result.Path.push_back(size_t(I));
      using element = typename std::tuple_element_t<I, T>;
      return PathMatcher::visitTupleTreeNode<element>(Rest, Result);
    } else {
      return visitTuple<T, I + 1>(Current, Rest, Result);
    }
  } else {
    // Not found
    return false;
  }
}

template<typename T>
std::optional<TupleTreePath> stringAsPath(llvm::StringRef Path);

template<typename ResultT, typename RootT>
ResultT *getByPath(llvm::StringRef Path, RootT &M) {
  auto MaybeKeyVector = stringAsPath<RootT>(Path);
  if (not MaybeKeyVector)
    return {};
  else
    return getByPath<ResultT>(*MaybeKeyVector, M);
}

//
// validateTupleTree
//
template<typename T>
concept TupleTreeScalar = not TupleSizeCompatible<T>
                          and not KeyedObjectContainer<T>
                          and not UpcastablePointerLike<T>;

template<TupleSizeCompatible T, typename L, size_t I = 0>
constexpr bool validateTupleTree(L);

template<TupleTreeScalar T, typename L>
constexpr bool validateTupleTree(L);

template<KeyedObjectContainer T, typename L>
constexpr bool validateTupleTree(L);

template<UpcastablePointerLike T, typename L>
constexpr bool validateTupleTree(L);

template<UpcastablePointerLike T, typename L>
constexpr bool validateTupleTree(L Check) {
  return Check((T *) nullptr)
         and validateTupleTree<typename T::element_type>(Check);
}

template<KeyedObjectContainer T, typename L>
constexpr bool validateTupleTree(L Check) {
  return Check((T *) nullptr)
         and validateTupleTree<typename T::value_type>(Check);
}

template<TupleTreeScalar T, typename L>
constexpr bool validateTupleTree(L Check) {
  return Check((std::remove_const_t<T> *) nullptr);
}

template<TupleSizeCompatible T, typename L, size_t I>
constexpr bool validateTupleTree(L Check) {
  if constexpr (I == 0 and not Check((T *) nullptr))
    return false;

  if constexpr (I < std::tuple_size_v<T>) {
    if constexpr (not validateTupleTree<std::tuple_element_t<I, T>>(Check))
      return false;
    return validateTupleTree<T, L, I + 1>(Check);
  }

  return true;
}

namespace revng {

template<typename T>
concept SetOrKOC = StrictSpecializationOf<T, std::set>
                   || KeyedObjectContainer<T>;

} // namespace revng
