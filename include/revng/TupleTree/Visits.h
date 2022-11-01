#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <cstdint>
#include <tuple>

#include "revng/ADT/KeyedObjectContainer.h"
#include "revng/ADT/UpcastablePointer.h"
#include "revng/TupleTree/TupleLikeTraits.h"
#include "revng/TupleTree/TupleTreeCompatible.h"
#include "revng/TupleTree/TupleTreePath.h"

//
// visitTupleTree implementation
//

namespace tupletree::detail {

template<typename Visitor, typename T, std::size_t... I>
void visitTupleImpl(Visitor &V, T &Obj, std::index_sequence<I...>) {
  (visitTupleTree(V, get<I>(Obj)), ...);
}

template<typename Visitor, typename T>
void visitTuple(Visitor &V, T &Obj) {
  visitTupleImpl(V, Obj, std::make_index_sequence<std::tuple_size_v<T>>());
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
// clang-format off
template<typename Visitor, typename T> requires (not TupleTreeCompatible<T>)
void visitTupleTree(Visitor &V, T &Element) {
  // clang-format on
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
template<TraitedTupleLike T, std::size_t... I>
size_t tupleIndexByNameImpl(llvm::StringRef Name, std::index_sequence<I...>) {
  size_t Result = -1;

  ((Result = Name == TupleLikeTraits<T>::FieldNames[I] ? I : Result), ...);

  return Result;
}

template<TraitedTupleLike T>
size_t tupleIndexByName(llvm::StringRef Name) {
  return tupleIndexByNameImpl(Name,
                              std::make_index_sequence<std::tuple_size_v<T>>());
}

//
// getByKey
//
namespace tupletree::detail {

template<typename ResultT, typename RootT, typename KeyT, std::size_t... I>
ResultT *getByKeyTupleImpl(RootT &M, KeyT Key, std::index_sequence<I...>) {
  ResultT *Result = nullptr;
  ((Result = I == Key ? reinterpret_cast<ResultT *>(&get<I>(M)) : Result), ...);
  return Result;
}

template<typename ResultT, size_t I = 0, typename RootT, typename KeyT>
ResultT *getByKeyTuple(RootT &M, KeyT Key) {
  return getByKeyTupleImpl(M,
                           Key,
                           std::make_index_sequence<
                             std::tuple_size_v<RootT>>());
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

// clang-format off
template<typename T, typename Visitor> requires (not TupleTreeCompatible<T>)
bool callOnPathSteps(Visitor &, llvm::ArrayRef<TupleTreeKeyWrapper>) {
  return false;
}

template<typename T, typename Visitor> requires(not UpcastablePointerLike<T>)
bool callOnPathStepsImpl(Visitor &V, llvm::ArrayRef<TupleTreeKeyWrapper> Path) {
  return callOnPathSteps<T, Visitor>(V, Path.slice(1));
}
// clang-format on

template<UpcastablePointerLike RootT, typename Visitor>
bool callOnPathStepsImpl(Visitor &V, llvm::ArrayRef<TupleTreeKeyWrapper> Path) {
  auto Dispatcher = [&](auto &Upcasted) {
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

template<typename RootT, typename Visitor, std::size_t... I>
bool callOnPathStepsTupleImpl(Visitor &V,
                              llvm::ArrayRef<TupleTreeKeyWrapper> Path,
                              std::index_sequence<I...>) {
  if (Path.size() == 0)
    return true;

  bool Result = true;

  ((Result = Path[0].get<size_t>() == I ?
               (V.template visitTupleElement<RootT, I>(),
                Path.size() > 1 ?
                  callOnPathStepsImpl<std::tuple_element_t<I, RootT>>(V, Path) :
                  Result) :
               Result),
   ...);

  return Result;
}

template<typename RootT, typename Visitor>
bool callOnPathStepsTuple(Visitor &V,
                          llvm::ArrayRef<TupleTreeKeyWrapper> Path) {
  auto Indexes = std::make_index_sequence<std::tuple_size_v<RootT>>();
  return callOnPathStepsTupleImpl<RootT>(V, Path, Indexes);
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

// clang-format off
template<typename T, typename Visitor> requires (not TupleTreeCompatible<T>)
bool callOnPathSteps(Visitor &, llvm::ArrayRef<TupleTreeKeyWrapper>, T &) {
  // clang-format on
  return false;
}

template<typename RootT, typename Visitor, std::size_t... I>
bool callOnPathStepsTupleImpl(Visitor &V,
                              llvm::ArrayRef<TupleTreeKeyWrapper> Path,
                              RootT &M,
                              std::index_sequence<I...>) {
  bool Result = true;

  ((Result = Path[0].get<size_t>() == I ?
               (V.template visitTupleElement<RootT, I>(get<I>(M)),
                Path.size() > 1 ? callOnPathSteps(V, Path.slice(1), get<I>(M)) :
                                  Result) :
               Result),
   ...);

  return Result;
}

template<typename RootT, typename Visitor>
bool callOnPathStepsTuple(Visitor &V,
                          llvm::ArrayRef<TupleTreeKeyWrapper> Path,
                          RootT &M) {
  auto Indices = std::make_index_sequence<std::tuple_size_v<RootT>>();
  return callOnPathStepsTupleImpl<RootT>(V, Path, M, Indices);
}

} // namespace tupletree::detail

template<UpcastablePointerLike RootT, typename Visitor>
bool callOnPathSteps(Visitor &V,
                     llvm::ArrayRef<TupleTreeKeyWrapper> Path,
                     RootT &M) {
  auto Dispatcher = [&](auto &Upcasted) {
    return callOnPathStepsTuple(V, Path, Upcasted);
  };
  // TODO: in case of nullptr we should abort
  return upcast(M, Dispatcher, false);
}

template<TupleSizeCompatible RootT, typename Visitor>
bool callOnPathSteps(Visitor &V,
                     llvm::ArrayRef<TupleTreeKeyWrapper> Path,
                     RootT &M) {
  return tupletree::detail::callOnPathStepsTuple(V, Path, M);
}

template<KeyedObjectContainer RootT, typename Visitor>
bool callOnPathSteps(Visitor &V,
                     llvm::ArrayRef<TupleTreeKeyWrapper> Path,
                     RootT &M) {
  using value_type = typename RootT::value_type;
  using KOT = KeyedObjectTraits<value_type>;
  using key_type = decltype(KOT::key(std::declval<value_type>()));
  auto TargetKey = Path[0].get<key_type>();

  auto It = M.find(TargetKey);
  if (It == M.end())
    return false;

  auto *Matching = &*It;

  V.template visitContainerElement<RootT>(TargetKey, *Matching);
  if (Path.size() > 1) {
    return callOnPathSteps(V, Path.slice(1), *Matching);
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

  // clang-format off
  template<typename T, typename K, typename KeyT>
    requires (not StrictSpecializationOf<K, UpcastablePointer>)
  void visitContainerElement(KeyT Key, K &Element) {
    // clang-format on
    PathSize -= 1;
    if (PathSize == 0)
      V.template visitContainerElement<T>(Key, Element);
  }
};

} // namespace tupletree::detail

template<typename RootT, typename Visitor>
bool callByPath(Visitor &V, const TupleTreePath &Path, RootT &M) {
  using namespace tupletree::detail;
  CallByPathVisitorWithInstance<Visitor> CBPV{ Path.size(), V };
  return callOnPathSteps(CBPV, Path.toArrayRef(), M);
}

//
// getByPath
//
namespace tupletree::detail {

template<typename ResultT>
struct GetByPathVisitor {
  ResultT *Result = nullptr;

  template<typename T, typename K, typename KeyT>
  void visitContainerElement(KeyT, K &) {
    Result = nullptr;
  }

  template<typename T, typename KeyT>
  void visitContainerElement(KeyT, ResultT &Element) {
    Result = &Element;
  }

  template<typename, size_t, typename K>
  void visitTupleElement(K &) {
    Result = nullptr;
  }

  template<typename, size_t>
  void visitTupleElement(ResultT &Element) {
    Result = &Element;
  }
};

} // namespace tupletree::detail

template<typename ResultT, typename RootT>
ResultT *getByPath(const TupleTreePath &Path, RootT &M) {
  using namespace tupletree::detail;
  GetByPathVisitor<ResultT> GBPV;
  if (not callByPath(GBPV, Path, M))
    return nullptr;
  else
    return GBPV.Result;
}

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
    if (not callOnPathSteps<T>(PV, Path.toArrayRef()))
      return {};
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
  static std::optional<PathMatcher> create(llvm::StringRef Path) {
    revng_assert(Path.startswith("/"));
    PathMatcher Result;
    if (visitTupleTreeNode<T>(Path.substr(1), Result))
      return Result;
    else
      return {};
  }

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

  template<typename T, std::size_t... I>
  void extractKeysImpl(const TupleTreePath &Search,
                       T &Tuple,
                       std::index_sequence<I...>) const {
    ((std::get<I>(Tuple) = Search[Free[I]].get<std::tuple_element_t<I, T>>()),
     ...);
  }

  template<typename T, size_t I = 0>
  void extractKeys(const TupleTreePath &Search, T &Tuple) const {
    return extractKeysImpl(Search,
                           Tuple,
                           std::make_index_sequence<std::tuple_size_v<T>>());
  }

private:
  template<TraitedTupleLike T, std::size_t... I>
  static bool visitTupleImpl(llvm::StringRef Current,
                             llvm::StringRef Rest,
                             PathMatcher &Matcher,
                             std::index_sequence<I...>);

  template<TraitedTupleLike T>
  static bool visitTuple(llvm::StringRef Current,
                         llvm::StringRef Rest,
                         PathMatcher &Matcher) {
    return visitTupleImpl<T>(Current,
                             Rest,
                             Matcher,
                             std::make_index_sequence<std::tuple_size_v<T>>());
  }

  template<UpcastablePointerLike T>
  static bool
  dispatchToConcreteType(llvm::StringRef String, PathMatcher &Result);

  template<UpcastablePointerLike T>
  static bool visitTupleTreeNode(llvm::StringRef String, PathMatcher &Result);

  template<TupleSizeCompatible T>
  static bool visitTupleTreeNode(llvm::StringRef String, PathMatcher &Result);

  template<KeyedObjectContainer T>
  static bool visitTupleTreeNode(llvm::StringRef String, PathMatcher &Result);

  // clang-format off
  template<typename T> requires (not TupleTreeCompatible<T>)
  static bool visitTupleTreeNode(llvm::StringRef Path, PathMatcher &Result);
  // clang-format on
};

template<UpcastablePointerLike P, typename L>
void invokeBySerializedKey(llvm::StringRef SerializedKey, const L &Callable) {
  using element_type = std::remove_reference_t<decltype(*std::declval<P>())>;
  using KOT = KeyedObjectTraits<element_type>;
  using KeyT = decltype(KOT::key(std::declval<element_type>()));
  auto Key = getValueFromYAMLScalar<KeyT>(SerializedKey);
  invokeByKey<P, KeyT, L>(Key, Callable);
}

template<UpcastablePointerLike T>
bool PathMatcher::dispatchToConcreteType(llvm::StringRef String,
                                         PathMatcher &Result) {

  auto Splitted = String.split('/');

  bool Res = false;
  auto Dispatch = [&]<typename Upcasted>(const Upcasted *Arg) {
    Res = PathMatcher::visitTupleTreeNode<Upcasted>(Splitted.second, Result);
  };

  invokeBySerializedKey<T>(Splitted.first, Dispatch);

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

  if (Before == "*") {
    Result.Free.push_back(Result.Path.size());
    Result.Path.emplace_back<Key>();
  } else {
    Result.Path.push_back(getValueFromYAMLScalar<Key>(Before));
  }

  if constexpr (StrictSpecializationOf<Value, UpcastablePointer>)
    return dispatchToConcreteType<Value>(String, Result);
  else
    return visitTupleTreeNode<Value>(After, Result);
}

// clang-format off
template<typename T> requires (not TupleTreeCompatible<T>)
bool PathMatcher::visitTupleTreeNode(llvm::StringRef Path,
                                     PathMatcher &Result) {
  // clang-format on
  return Path.size() == 0;
}

template<TraitedTupleLike T, std::size_t... I>
bool PathMatcher::visitTupleImpl(llvm::StringRef Current,
                                 llvm::StringRef Rest,
                                 PathMatcher &Matcher,
                                 std::index_sequence<I...>) {
  bool Result = false;

  ((Result = TupleLikeTraits<T>::FieldNames[I] == Current ?
               (Matcher.Path.push_back(I),
                PathMatcher::visitTupleTreeNode<
                  typename std::tuple_element_t<I, T>>(Rest, Matcher)) :
               Result),
   ...);

  return Result;
}

template<typename T>
std::optional<TupleTreePath> stringAsPath(llvm::StringRef Path) {
  if (Path.empty())
    return std::nullopt;

  auto Result = PathMatcher::create<T>(Path);
  if (Result)
    return Result->path();
  else
    return std::nullopt;
}

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
template<TupleSizeCompatible T, typename L>
constexpr bool validateTupleTree(L);

template<typename T, typename L>
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

template<typename T, typename L>
constexpr bool validateTupleTree(L Check) {
  return Check((std::remove_const_t<T> *) nullptr);
}

template<TupleSizeCompatible T, typename L, std::size_t... I>
constexpr bool validateTupleTreeImpl(L Check, std::index_sequence<I...>) {
  if (not Check((T *) nullptr))
    return false;

  bool Result = true;

  ((Result = (Result
              and not validateTupleTree<std::tuple_element_t<I, T>>(Check)) ?
               false :
               Result),
   ...);

  return Result;
}

template<TupleSizeCompatible T, typename L>
constexpr bool validateTupleTree(L Check) {
  return validateTupleTreeImpl<
    T>(Check, std::make_index_sequence<std::tuple_size_v<T>>());
}
