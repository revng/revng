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
template<typename Visitor, HasTupleSize T>
void visitTupleTree(Visitor &V, T &Obj) {
  V.PreVisit(Obj);
  tupletree::detail::visitTuple(V, Obj);
  V.PostVisit(Obj);
}

// Container-like
template<typename Visitor, IsKeyedObjectContainer T>
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
template<typename T, size_t I = 0>
size_t tupleIndexByName(llvm::StringRef Name) {
  if constexpr (I < std::tuple_size_v<T>) {
    llvm::StringRef ThisName = TupleLikeTraits<T>::FieldsName[I];
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

template<typename ResultT, HasTupleSize RootT, typename KeyT>
ResultT getByKey(RootT &M, KeyT Key) {
  return tupletree::detail::getByKeyTuple<ResultT>(M, Key);
}

template<typename ResultT, IsKeyedObjectContainer RootT, typename KeyT>
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
template<HasTupleSize RootT, typename Visitor>
bool callOnPathSteps(Visitor &V, llvm::ArrayRef<TupleTreeKeyWrapper> Path);

template<NotTupleTreeCompatible RootT, typename Visitor>
bool callOnPathSteps(Visitor &V, llvm::ArrayRef<TupleTreeKeyWrapper> Path) {
  return false;
}

template<NotUpcastablePointerLike RootT, typename Visitor>
bool callOnPathStepsImpl(Visitor &V, llvm::ArrayRef<TupleTreeKeyWrapper> Path) {
  return callOnPathSteps<RootT, Visitor>(V, Path.slice(1));
}

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

template<IsKeyedObjectContainer RootT, typename Visitor>
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

template<HasTupleSize RootT, typename Visitor>
bool callOnPathSteps(Visitor &V, llvm::ArrayRef<TupleTreeKeyWrapper> Path) {
  return tupletree::detail::callOnPathStepsTuple<RootT>(V, Path);
}

//
// callOnPathSteps (with instance)
//

namespace tupletree::detail {

template<NotTupleTreeCompatible RootT, typename Visitor>
bool callOnPathSteps(Visitor &V,
                     llvm::ArrayRef<TupleTreeKeyWrapper> Path,
                     RootT &M) {
  return false;
}

template<size_t I = 0, typename RootT, typename Visitor>
bool callOnPathStepsTuple(Visitor &V,
                          llvm::ArrayRef<TupleTreeKeyWrapper> Path,
                          RootT &M) {
  if constexpr (I < std::tuple_size_v<RootT>) {
    if (Path[0].get<size_t>() == I) {
      using next_type = typename std::tuple_element<I, RootT>::type;
      next_type &Element = get<I>(M);
      V.template visitTupleElement<RootT, I>(Element);
      if (Path.size() > 1) {
        return callOnPathSteps(V, Path.slice(1), Element);
      }
    } else {
      return callOnPathStepsTuple<I + 1>(V, Path, M);
    }
  }

  return true;
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

template<HasTupleSize RootT, typename Visitor>
bool callOnPathSteps(Visitor &V,
                     llvm::ArrayRef<TupleTreeKeyWrapper> Path,
                     RootT &M) {
  return tupletree::detail::callOnPathStepsTuple(V, Path, M);
}

template<IsKeyedObjectContainer RootT, typename Visitor>
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

  template<typename T, IsUpcastablePointer K, typename KeyT>
  void visitContainerElement(KeyT Key, K &Element) {
    PathSize -= 1;
    if (PathSize == 0)
      V.template visitContainerElement<T>(Key, *Element.get());
  }

  template<typename T, IsNotUpcastablePointer K, typename KeyT>
  void visitContainerElement(KeyT Key, K &Element) {
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

  template<typename T, int I>
  void visitTupleElement() {
    Stream << "/" << TupleLikeTraits<T>::FieldsName[I];
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

  template<typename T, size_t I = 0>
  void extractKeys(const TupleTreePath &Search, T &Tuple) const {
    if constexpr (I < std::tuple_size_v<T>) {
      using element = std::tuple_element_t<I, T>;
      std::get<I>(Tuple) = Search[Free[I]].get<element>();
      extractKeys<T, I + 1>(Search, Tuple);
    }
  }

private:
  template<typename T, size_t I = 0>
  static bool visitTuple(llvm::StringRef Current,
                         llvm::StringRef Rest,
                         PathMatcher &Result);

  template<UpcastablePointerLike T>
  static bool
  dispatchToCorrectUpcastable(llvm::StringRef String, PathMatcher &Result);

  template<UpcastablePointerLike T>
  static bool visitTupleTreeNode(llvm::StringRef String, PathMatcher &Result);

  template<HasTupleSize T>
  static bool visitTupleTreeNode(llvm::StringRef String, PathMatcher &Result);

  template<IsKeyedObjectContainer T>
  static bool visitTupleTreeNode(llvm::StringRef String, PathMatcher &Result);

  template<NotTupleTreeCompatible T>
  static bool visitTupleTreeNode(llvm::StringRef Path, PathMatcher &Result);
};

template<UpcastablePointerLike P, typename L>
void invokeFromSerializedKey(llvm::StringRef SerializedKey, const L &Callable) {
  using element_type = std::remove_reference_t<decltype(*std::declval<P>())>;
  using KOT = KeyedObjectTraits<element_type>;
  using KeyT = decltype(KOT::key(std::declval<element_type>()));
  auto Key = getValueFromYAMLScalar<KeyT>(SerializedKey);
  invokeFromKey<P, KeyT, L>(Key, Callable);
}

template<UpcastablePointerLike T>
bool PathMatcher::dispatchToCorrectUpcastable(llvm::StringRef String,
                                              PathMatcher &Result) {

  auto Splitted = String.split('/');

  bool Res = false;
  auto Dispatch = [&]<typename Upcasted>(const Upcasted *Arg) {
    Res = PathMatcher::visitTupleTreeNode<Upcasted>(Splitted.second, Result);
  };

  invokeFromSerializedKey<T>(Splitted.first, Dispatch);

  return Res;
}

template<UpcastablePointerLike T>
bool PathMatcher::visitTupleTreeNode(llvm::StringRef String,
                                     PathMatcher &Result) {
  return dispatchToCorrectUpcastable<T>(String, Result);
}

template<HasTupleSize T>
bool PathMatcher::visitTupleTreeNode(llvm::StringRef String,
                                     PathMatcher &Result) {
  if (String.size() == 0)
    return true;

  auto [Before, After] = String.split('/');
  return visitTuple<T>(Before, After, Result);
}

template<IsKeyedObjectContainer T>
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

  if constexpr (IsUpcastablePointer<Value>)
    return dispatchToCorrectUpcastable<Value>(String, Result);
  else
    return visitTupleTreeNode<Value>(After, Result);
}

template<NotTupleTreeCompatible T>
bool PathMatcher::visitTupleTreeNode(llvm::StringRef Path,
                                     PathMatcher &Result) {
  return Path.size() == 0;
}

template<typename T, size_t I>
bool PathMatcher::visitTuple(llvm::StringRef Current,
                             llvm::StringRef Rest,
                             PathMatcher &Result) {
  if constexpr (I < std::tuple_size_v<T>) {
    if (TupleLikeTraits<T>::FieldsName[I] == Current) {
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
template<HasTupleSize T, typename L, size_t I = 0>
constexpr bool validateTupleTree(L);

template<typename T, typename L>
constexpr bool validateTupleTree(L);

template<IsKeyedObjectContainer T, typename L>
constexpr bool validateTupleTree(L);

template<UpcastablePointerLike T, typename L>
constexpr bool validateTupleTree(L);

template<UpcastablePointerLike T, typename L>
constexpr bool validateTupleTree(L Check) {
  return Check((T *) nullptr)
         and validateTupleTree<typename T::element_type>(Check);
}

template<IsKeyedObjectContainer T, typename L>
constexpr bool validateTupleTree(L Check) {
  return Check((T *) nullptr)
         and validateTupleTree<typename T::value_type>(Check);
}

template<typename T, typename L>
constexpr bool validateTupleTree(L Check) {
  return Check((std::remove_const_t<T> *) nullptr);
}

template<HasTupleSize T, typename L, size_t I>
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
