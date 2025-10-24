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
  if (Obj != nullptr) {
    V.PreVisit(Obj);
    upcast(Obj, [&V](auto &Upcasted) { visitTupleTree(V, Upcasted); });
    V.PostVisit(Obj);
  }
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
// `callOnPathSteps` without an instance
//

template<NotTupleTreeCompatible RootT, typename Visitor>
bool callOnPathSteps(Visitor &, llvm::ArrayRef<TupleTreeKeyWrapper> Path) {
  if (Path.empty())
    return true;

  revng_abort("Unsupported step");
}

namespace tupletree::detail {

template<TupleSizeCompatible RootT,
         size_t I = 0,
         typename KindT,
         typename Visitor>
bool polymorphicTupleImpl(Visitor &V,
                          llvm::ArrayRef<TupleTreeKeyWrapper> Path,
                          KindT Kind) {
  if constexpr (I < std::tuple_size_v<RootT>) {
    if (Path[0].get<unsigned long>() == I) {
      if constexpr (std::is_same_v<KindT, size_t>)
        V.template visitTupleElement<RootT, I>();
      else
        V.template visitPolymorphicElement<RootT, I>(Kind);

      using next_type = typename std::tuple_element<I, RootT>::type;
      return callOnPathSteps<next_type>(V, Path.slice(1));
    } else {
      return polymorphicTupleImpl<RootT, I + 1>(V, Path, Kind);
    }
  }

  return false;
}

template<TupleSizeCompatible RootT, typename Visitor>
bool tupleImpl(Visitor &V, llvm::ArrayRef<TupleTreeKeyWrapper> Path) {
  return polymorphicTupleImpl<RootT, 0, size_t>(V, Path, 0);
}

} // namespace tupletree::detail

template<UpcastablePointerLike RootT, typename Visitor>
bool callOnPathSteps(Visitor &V, llvm::ArrayRef<TupleTreeKeyWrapper> Path) {
  if (Path.empty())
    return true;

  revng_assert(Path.size() > 1);

  using KindType = std::decay_t<decltype(std::declval<RootT>()->Kind())>;
  KindType Kind = Path[0].get<KindType>();
  auto Dispatcher = [&V, &Path, &Kind]<TupleSizeCompatible UT>(UT &) -> bool {
    return tupletree::detail::polymorphicTupleImpl<UT>(V, Path.slice(1), Kind);
  };

  // Technically changing kind manually is unsafe, but since no-one is ever
  // going to touch the object (we only care about the type), this is an easy
  // way to trick "upcast" into doing what we want without introducing more
  // complexity (or trying to fill in the rest of the key) to do this properly.
  std::decay_t<typename RootT::element_type> Temporary;
  Temporary.Kind() = Kind;
  return upcast(&Temporary, Dispatcher, false);
}

template<TupleSizeCompatible RootT, typename Visitor>
bool callOnPathSteps(Visitor &V, llvm::ArrayRef<TupleTreeKeyWrapper> Path) {
  if (Path.empty())
    return true;

  return tupletree::detail::tupleImpl<RootT>(V, Path);
}

template<KeyedObjectContainer RootT, typename Visitor>
bool callOnPathSteps(Visitor &V, llvm::ArrayRef<TupleTreeKeyWrapper> Path) {
  if (Path.empty())
    return true;

  using value_type = typename RootT::value_type;
  using KOT = KeyedObjectTraits<value_type>;
  using key_type = decltype(KOT::key(std::declval<value_type>()));
  auto TargetKey = Path[0].get<key_type>();

  V.template visitContainerElement<RootT>(TargetKey);

  using NextStep = std::conditional_t<std::is_const_v<RootT>,
                                      const typename RootT::value_type,
                                      typename RootT::value_type>;
  return callOnPathSteps<NextStep>(V, Path.slice(1));
}

//
// `callOnPathSteps` with an instance
//

template<NotTupleTreeCompatible RootT, typename Visitor>
bool callOnPathSteps(Visitor &,
                     llvm::ArrayRef<TupleTreeKeyWrapper> Path,
                     RootT &) {
  if (Path.empty())
    return true;

  revng_abort("Unsupported step");
}

namespace tupletree::detail {

template<TupleSizeCompatible RootT,
         size_t I = 0,
         typename KindT,
         typename Visitor>
bool polymorphicTupleImpl(Visitor &V,
                          llvm::ArrayRef<TupleTreeKeyWrapper> Path,
                          RootT &M,
                          KindT Kind) {
  if constexpr (I < std::tuple_size_v<RootT>) {
    if (Path[0].get<size_t>() == I) {
      auto &Element = get<I>(M);
      if constexpr (std::is_same_v<KindT, size_t>)
        V.template visitTupleElement<RootT, I>(Element);
      else
        V.template visitPolymorphicElement<RootT, I>(Kind, Element);

      using next_type = typename std::tuple_element<I, RootT>::type;
      return callOnPathSteps<next_type>(V, Path.slice(1), Element);
    } else {
      return polymorphicTupleImpl<RootT, I + 1>(V, Path, M, Kind);
    }
  }

  return false;
}

template<TupleSizeCompatible RootT, typename Visitor>
bool tupleImpl(Visitor &V, llvm::ArrayRef<TupleTreeKeyWrapper> Path, RootT &M) {
  return polymorphicTupleImpl<RootT, 0, size_t>(V, Path, M, 0);
}

} // namespace tupletree::detail

template<UpcastablePointerLike RootT, typename Visitor>
bool callOnPathSteps(Visitor &V,
                     llvm::ArrayRef<TupleTreeKeyWrapper> Path,
                     RootT &M) {
  auto Dispatcher = [&V, &Path]<TupleSizeCompatible UT>(UT &Upcasted) -> bool {
    if (Path.empty())
      return true;

    using KindType = std::decay_t<decltype(std::declval<RootT>()->Kind())>;
    return tupletree::detail::polymorphicTupleImpl<UT>(V,
                                                       Path.slice(1),
                                                       Upcasted,
                                                       Path[0].get<KindType>());
  };

  return upcast(M, Dispatcher, false);
}

template<TupleSizeCompatible RootT, typename Visitor>
bool callOnPathSteps(Visitor &V,
                     llvm::ArrayRef<TupleTreeKeyWrapper> Path,
                     RootT &M) {
  if (Path.empty())
    return true;

  return tupletree::detail::tupleImpl<RootT>(V, Path, M);
}

template<typename T>
concept HasTryGet = requires(T a) {
  { &a.tryGet };
};

template<KeyedObjectContainer RootT, typename Visitor>
bool callOnPathSteps(Visitor &V,
                     llvm::ArrayRef<TupleTreeKeyWrapper> Path,
                     RootT &M) {
  if (Path.empty())
    return true;

  using value_type = typename RootT::value_type;
  using KOT = KeyedObjectTraits<value_type>;
  using key_type = decltype(KOT::key(std::declval<value_type>()));
  auto TargetKey = Path[0].get<key_type>();

  decltype(&*M.find(TargetKey)) Entry;
  if constexpr (HasTryGet<RootT>)
    Entry = M.tryGet(TargetKey);
  else if (auto Iter = M.find(TargetKey); Iter != M.end())
    Entry = &*Iter;
  else
    return false;

  V.template visitContainerElement<RootT>(TargetKey, *Entry);

  using NextStep = std::conditional_t<std::is_const_v<RootT>,
                                      const typename RootT::value_type,
                                      typename RootT::value_type>;
  return callOnPathSteps<NextStep>(V, Path.slice(1), *Entry);
}

//
// `callByPath` without an instance
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

  template<typename T, size_t I, typename KindType>
  void visitPolymorphicElement(KindType Kind) {
    PathSize -= 2;
    if (PathSize == 0)
      V.template visitPolymorphicElement<T, I>(Kind);
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
// `callByPath` with an instance
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

  template<typename T, size_t I, typename K, typename KindType>
  void visitPolymorphicElement(KindType Kind, K &Element) {
    PathSize -= 2;
    if (PathSize == 0)
      V.template visitPolymorphicElement<T, I>(Kind, Element);
  }

  template<typename T,
           StrictSpecializationOf<UpcastablePointer> K,
           typename KeyT>
  void visitContainerElement(KeyT Key, K &Element) {
    --PathSize;
    if (PathSize == 0)
      V.template visitContainerElement<T>(Key, *Element.get());
  }

  template<typename T, typename K, typename KeyT>
    requires(not StrictSpecializationOf<K, UpcastablePointer>)
  void visitContainerElement(KeyT Key, K &Element) {
    --PathSize;
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

template<typename RootT>
constexpr bool IsConst = std::is_const_v<std::remove_reference_t<RootT>>;

template<typename ResultT, typename RootT>
using getByPathRV = std::conditional_t<IsConst<RootT>, const ResultT, ResultT>;

} // namespace tupletree::detail

template<typename ResultT, typename RootT>
tupletree::detail::getByPathRV<ResultT, RootT> *
getByPath(const TupleTreePath &Path, RootT &M);

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

  template<TraitedTupleLike T, int I, typename KindType>
    requires(std::is_enum_v<KindType>)
  void visitPolymorphicElement(KindType Kind) {
    Stream << "/" << toString(Kind)
           << "::" << TupleLikeTraits<T>::FieldNames[I];
  }

  template<typename T, typename KeyT>
  void visitContainerElement(KeyT Key) {
    Stream << "/" << getNameFromYAMLScalar(Key);
  }
};

} // namespace tupletree::detail

template<typename T>
std::optional<std::string> pathAsString(const TupleTreePath &Path);

//
// Path matcher
//

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
    Result[Index] = TupleTreeKeyWrapper(T(Arg));
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
  static bool visitTupleTreeNode(llvm::StringRef String, PathMatcher &Result);

  template<TupleSizeCompatible T>
  static bool visitTupleTreeNode(llvm::StringRef String, PathMatcher &Result);

  template<TupleSizeCompatible T, typename KindType>
    requires(std::is_enum_v<KindType>)
  static bool visitTupleTreeNode(llvm::StringRef String,
                                 PathMatcher &Result,
                                 KindType Kind);

  template<KeyedObjectContainer T>
  static bool visitTupleTreeNode(llvm::StringRef String, PathMatcher &Result);

  template<NotTupleTreeCompatible T>
  static bool visitTupleTreeNode(llvm::StringRef Path, PathMatcher &Result);
};

namespace tupletree::details {

template<typename RootT, size_t I = 0, typename Visitor>
bool selectOptionImpl(llvm::StringRef Kind, Visitor &V) {
  if constexpr (I < std::tuple_size_v<RootT>) {
    using ElementT = std::tuple_element_t<I, RootT>;
    if (TupleLikeTraits<ElementT>::Name == Kind)
      return V.template operator()<ElementT>();
    else
      return selectOptionImpl<RootT, I + 1>(Kind, V);
  }

  return false;
}

} // namespace tupletree::details

template<UpcastablePointerLike T>
bool PathMatcher::visitTupleTreeNode(llvm::StringRef String,
                                     PathMatcher &Result) {
  if (String.size() == 0)
    return true;

  auto &&[Kind, RHS] = String.split("::");
  auto Dispatch = [&RHS, &Result]<typename Upcasted>() {
    auto &&[Before, After] = RHS.split('/');
    return PathMatcher::visitTuple<Upcasted>(Before, After, Result);
  };

  using ElementType = typename T::element_type;
  using KindType = std::decay_t<decltype(std::declval<ElementType>().Kind())>;
  Result.Path.push_back(getValueFromYAMLScalar<KindType>(Kind));

  using Options = typename concrete_types_traits<ElementType>::type;
  return tupletree::details::selectOptionImpl<Options>(Kind, Dispatch);
}

template<TupleSizeCompatible T>
bool PathMatcher::visitTupleTreeNode(llvm::StringRef String,
                                     PathMatcher &Result) {
  if (String.size() == 0)
    return true;

  auto &&[Before, After] = String.split('/');
  return visitTuple<T>(Before, After, Result);
}

template<KeyedObjectContainer T>
bool PathMatcher::visitTupleTreeNode(llvm::StringRef String,
                                     PathMatcher &Result) {
  if (String.size() == 0)
    return true;

  auto &&[Before, After] = String.split('/');

  using Key = std::remove_cv_t<typename T::key_type>;
  using Value = typename T::value_type;

  if constexpr (StrictSpecializationOf<Value, UpcastablePointer>) {
    auto &&[PreDash, PostDash] = Before.split("-");
    if (PreDash == "*") {
      // Mark as free
      Result.Free.push_back(Result.Path.size());

      //
      // Extract the Kind of the abstract type in
      // the `model::UpcastableTypeDefinition`
      //
      // TODO: Consider using the kind from the next step instead.

      // Get the kind type for the abstract type
      using Kind = typename Value::element_type::TypeOfKind;

      // Extract Kind from "Kind-*" and deserialize it
      Kind MatcherKind = getValueFromYAMLScalar<Kind>(PostDash);
      static_assert(std::is_enum_v<std::decay_t<Kind>>);

      // Push in Path a Key initializing only the first field (the kind)
      Key Component;
      auto &TheKind = std::get<std::tuple_size_v<Key> - 1>(Component);
      using KindType = decltype(TheKind);
      static_assert(std::is_enum_v<std::decay_t<KindType>>);
      TheKind = MatcherKind;
      static_assert(Key::LastFieldIsKind);
      Result.Path.emplace_back<Key>(Component);
    } else {
      Result.Path.push_back(getValueFromYAMLScalar<Key>(Before));
    }
  } else {
    if (Before == "*") {
      Result.Free.push_back(Result.Path.size());
      Result.Path.emplace_back<Key>();
    } else {
      Result.Path.push_back(getValueFromYAMLScalar<Key>(Before));
    }
  }

  return visitTupleTreeNode<Value>(After, Result);
}

template<NotTupleTreeCompatible T>
bool PathMatcher::visitTupleTreeNode(llvm::StringRef Path,
                                     PathMatcher &Result) {
  return Path.empty();
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
// Validation
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
  // TODO: is there a better way to limit it?
  return Check((T *) nullptr);
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
