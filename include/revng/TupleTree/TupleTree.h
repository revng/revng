#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <array>
#include <optional>
#include <set>
#include <type_traits>
#include <variant>
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/ADT/KeyedObjectContainer.h"
#include "revng/ADT/KeyedObjectTraits.h"
#include "revng/ADT/TupleTreePath.h"
#include "revng/ADT/UpcastablePointer.h"
#include "revng/Support/Assert.h"
#include "revng/Support/Debug.h"
#include "revng/Support/YAMLTraits.h"

//
// slice
//

/// Copy into a std::array a slice of an llvm::ArrayRef
template<size_t Start, size_t Size, typename T>
std::array<T, Size> slice(llvm::ArrayRef<T> Old) {
  std::array<T, Size> Result;
  auto StartIt = Old.begin() + Start;
  std::copy(StartIt, StartIt + Size, Result.begin());
  return Result;
}

/// Copy into a std::array a slice of a std::array
template<size_t Start, size_t Size, typename T, size_t OldSize>
std::array<T, Size> slice(const std::array<T, OldSize> &Old) {
  std::array<T, Size> Result;
  auto StartIt = Old.begin() + Start;
  std::copy(StartIt, StartIt + Size, Result.begin());
  return Result;
}

//
// TupleLikeTraits
//

/// Trait to provide name of the tuple-like class and its fields
template<typename T>
struct TupleLikeTraits {
  enum class Fields {};
};

template<typename T>
concept HasTupleLikeTraits = requires {
  typename TupleLikeTraits<T>::tuple;
  typename TupleLikeTraits<T>::Fields;
  { TupleLikeTraits<T>::Name };
  { TupleLikeTraits<T>::FieldsName };
};

//
// Implementation of MappingTraits for TupleLikeTraits implementors
//

/// Tuple-like can implement llvm::yaml::MappingTraits inheriting this class
template<typename T, typename TupleLikeTraits<T>::Fields... Optionals>
struct TupleLikeMappingTraits {
  using Fields = typename TupleLikeTraits<T>::Fields;

  template<Fields Index, size_t I = 0>
  static constexpr bool isOptional() {
    constexpr size_t Count = sizeof...(Optionals);
    constexpr std::array<Fields, Count> OptionalsArray{ Optionals... };
    if constexpr (I < Count) {
      return (OptionalsArray[I] == Index) || isOptional<Index, I + 1>();
    } else {
      return false;
    }
  }

  // Recursive step
  template<size_t I = 0>
  static void mapping(llvm::yaml::IO &IO, T &Obj) {
    if constexpr (I < std::tuple_size_v<T>) {
      auto Name = TupleLikeTraits<T>::FieldsName[I];
      constexpr Fields Field = static_cast<Fields>(I);

      using tuple_element = std::tuple_element_t<I, T>;
      auto &Element = get<I>(Obj);
      if constexpr (isOptional<Field>()) {
        IO.mapOptional(Name, Element, tuple_element{});
      } else {
        IO.mapRequired(Name, Element);
      }

      // Recur
      mapping<I + 1>(IO, Obj);
    }
  }
};

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

template<UpcastablePointerLike RootT, typename Visitor>
bool callOnPathSteps(Visitor &V, llvm::ArrayRef<TupleTreeKeyWrapper> Path) {
  using element_type = pointee<RootT>;
  return callOnPathStepsTuple<element_type>(V, Path);
}

template<IsKeyedObjectContainer RootT, typename Visitor>
bool callOnPathSteps(Visitor &V, llvm::ArrayRef<TupleTreeKeyWrapper> Path) {
  using value_type = typename RootT::value_type;
  using KOT = KeyedObjectTraits<value_type>;
  using key_type = decltype(KOT::key(std::declval<value_type>()));
  auto TargetKey = Path[0].get<key_type>();

  V.template visitContainerElement<RootT>(TargetKey);
  if (Path.size() > 1) {
    return callOnPathSteps<typename RootT::value_type>(V, Path.slice(1));
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
        return callOnPathSteps<next_type>(V, Path.slice(1));
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
  static bool visitTupleTreeNode(llvm::StringRef String, PathMatcher &Result);

  template<HasTupleSize T>
  static bool visitTupleTreeNode(llvm::StringRef String, PathMatcher &Result);

  template<IsKeyedObjectContainer T>
  static bool visitTupleTreeNode(llvm::StringRef String, PathMatcher &Result);

  template<NotTupleTreeCompatible T>
  static bool visitTupleTreeNode(llvm::StringRef Path, PathMatcher &Result);
};

template<UpcastablePointerLike T>
bool PathMatcher::visitTupleTreeNode(llvm::StringRef String,
                                     PathMatcher &Result) {
  using element_type = std::remove_reference_t<decltype(*std::declval<T>())>;
  return PathMatcher::visitTupleTreeNode<element_type>(String, Result);
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

//
// FOR_EACH macro implemenation
//
#define GET_MACRO(_0,   \
                  _1,   \
                  _2,   \
                  _3,   \
                  _4,   \
                  _5,   \
                  _6,   \
                  _7,   \
                  _8,   \
                  _9,   \
                  _10,  \
                  _11,  \
                  _12,  \
                  _13,  \
                  _14,  \
                  _15,  \
                  _16,  \
                  NAME, \
                  ...)  \
  NAME
#define NUMARGS(...)     \
  GET_MACRO(_0,          \
            __VA_ARGS__, \
            16,          \
            15,          \
            14,          \
            13,          \
            12,          \
            11,          \
            10,          \
            9,           \
            8,           \
            7,           \
            6,           \
            5,           \
            4,           \
            3,           \
            2,           \
            1)

#define FE_0(ACTION, TOTAL, ARG)

#define FE_1(ACTION, TOTAL, ARG, X) ACTION(ARG, (TOTAL) -0, X)

#define FE_2(ACTION, TOTAL, ARG, X, ...) \
  ACTION(ARG, (TOTAL) -1, X)             \
  FE_1(ACTION, TOTAL, ARG, __VA_ARGS__)

#define FE_3(ACTION, TOTAL, ARG, X, ...) \
  ACTION(ARG, (TOTAL) -2, X)             \
  FE_2(ACTION, TOTAL, ARG, __VA_ARGS__)

#define FE_4(ACTION, TOTAL, ARG, X, ...) \
  ACTION(ARG, (TOTAL) -3, X)             \
  FE_3(ACTION, TOTAL, ARG, __VA_ARGS__)

#define FE_5(ACTION, TOTAL, ARG, X, ...) \
  ACTION(ARG, (TOTAL) -4, X)             \
  FE_4(ACTION, TOTAL, ARG, __VA_ARGS__)

#define FE_6(ACTION, TOTAL, ARG, X, ...) \
  ACTION(ARG, (TOTAL) -5, X)             \
  FE_5(ACTION, TOTAL, ARG, __VA_ARGS__)

#define FE_7(ACTION, TOTAL, ARG, X, ...) \
  ACTION(ARG, (TOTAL) -6, X)             \
  FE_6(ACTION, TOTAL, ARG, __VA_ARGS__)

#define FE_8(ACTION, TOTAL, ARG, X, ...) \
  ACTION(ARG, (TOTAL) -7, X)             \
  FE_7(ACTION, TOTAL, ARG, __VA_ARGS__)

#define FE_9(ACTION, TOTAL, ARG, X, ...) \
  ACTION(ARG, (TOTAL) -8, X)             \
  FE_8(ACTION, TOTAL, ARG, __VA_ARGS__)

#define FE_10(ACTION, TOTAL, ARG, X, ...) \
  ACTION(ARG, (TOTAL) -9, X)              \
  FE_9(ACTION, TOTAL, ARG, __VA_ARGS__)

#define FE_11(ACTION, TOTAL, ARG, X, ...) \
  ACTION(ARG, (TOTAL) -10, X)             \
  FE_10(ACTION, TOTAL, ARG, __VA_ARGS__)

#define FE_12(ACTION, TOTAL, ARG, X, ...) \
  ACTION(ARG, (TOTAL) -11, X)             \
  FE_11(ACTION, TOTAL, ARG, __VA_ARGS__)

#define FE_13(ACTION, TOTAL, ARG, X, ...) \
  ACTION(ARG, (TOTAL) -12, X)             \
  FE_12(ACTION, TOTAL, ARG, __VA_ARGS__)

#define FE_14(ACTION, TOTAL, ARG, X, ...) \
  ACTION(ARG, (TOTAL) -13, X)             \
  FE_13(ACTION, TOTAL, ARG, __VA_ARGS__)

#define FE_15(ACTION, TOTAL, ARG, X, ...) \
  ACTION(ARG, (TOTAL) -14, X)             \
  FE_14(ACTION, TOTAL, ARG, __VA_ARGS__)

#define FE_16(ACTION, TOTAL, ARG, X, ...) \
  ACTION(ARG, (TOTAL) -15, X)             \
  FE_15(ACTION, TOTAL, ARG, __VA_ARGS__)

/// Calls ACTION(ARG, INDEX, VA_ARG) for each VA_ARG in ...
#define FOR_EACH(ACTION, ARG, ...) \
  GET_MACRO(_0,                    \
            __VA_ARGS__,           \
            FE_16,                 \
            FE_15,                 \
            FE_14,                 \
            FE_13,                 \
            FE_12,                 \
            FE_11,                 \
            FE_10,                 \
            FE_9,                  \
            FE_8,                  \
            FE_7,                  \
            FE_6,                  \
            FE_5,                  \
            FE_4,                  \
            FE_3,                  \
            FE_2,                  \
            FE_1,                  \
            FE_0)                  \
  (ACTION, (NUMARGS(__VA_ARGS__) - 1), ARG, __VA_ARGS__)

//
// Macros to transform struct in tuple-like
//

#define TUPLE_TYPES(class, index, field) , decltype(class ::field)

#define TUPLE_FIELD_NAME(class, index, field) #field,

#define ENUM_ENTRY(class, index, field) field = index,

template<typename ToSkip, typename... A>
using skip_first_tuple = std::tuple<A...>;

#define INTROSPECTION_1(classname, ...)                                   \
  template<>                                                              \
  struct TupleLikeTraits<classname> {                                     \
    static constexpr const char *Name = #classname;                       \
                                                                          \
    using tuple = skip_first_tuple<                                       \
      void FOR_EACH(TUPLE_TYPES, classname, __VA_ARGS__)>;                \
                                                                          \
    static constexpr const char *FieldsName[std::tuple_size_v<tuple>] = { \
      FOR_EACH(TUPLE_FIELD_NAME, classname, __VA_ARGS__)                  \
    };                                                                    \
                                                                          \
    enum class Fields { FOR_EACH(ENUM_ENTRY, classname, __VA_ARGS__) };   \
  };

#define GET_IMPLEMENTATIONS(class, index, field) \
  else if constexpr (I == index) return x.field;

#define INTROSPECTION_2(class, ...)                   \
  template<int I>                                     \
  auto &get(class &&x) {                              \
    if constexpr (false)                              \
      return NULL;                                    \
    FOR_EACH(GET_IMPLEMENTATIONS, class, __VA_ARGS__) \
  }                                                   \
                                                      \
  template<int I>                                     \
  const auto &get(const class &x) {                   \
    if constexpr (false)                              \
      return NULL;                                    \
    FOR_EACH(GET_IMPLEMENTATIONS, class, __VA_ARGS__) \
  }                                                   \
                                                      \
  template<int I>                                     \
  auto &get(class &x) {                               \
    if constexpr (false)                              \
      return NULL;                                    \
    FOR_EACH(GET_IMPLEMENTATIONS, class, __VA_ARGS__) \
  }

#define INTROSPECTION(class, ...)     \
  INTROSPECTION_1(class, __VA_ARGS__) \
  INTROSPECTION_2(class, __VA_ARGS__)

#define INTROSPECTION_NS(ns, class, ...)  \
  INTROSPECTION_1(ns::class, __VA_ARGS__) \
  namespace ns {                          \
  INTROSPECTION_2(class, __VA_ARGS__)     \
  }

template<size_t Index, HasTupleLikeTraits T>
struct std::tuple_element<Index, T> {
  using type = std::tuple_element_t<Index, typename TupleLikeTraits<T>::tuple>;
};

template<typename T>
using TupleLikeTraitsTuple = typename TupleLikeTraits<T>::tuple;

template<HasTupleLikeTraits T>
struct std::tuple_size<T>
  : std::integral_constant<size_t, std::tuple_size_v<TupleLikeTraitsTuple<T>>> {
};

template<TupleTreeCompatible T>
class TupleTree;

template<typename T, typename R>
concept ConstOrNot = std::is_same_v<R, T> or std::is_same_v<const R, T>;

template<typename T, typename RootType>
class TupleTreeReference {
public:
  using RootT = RootType;
  using RootVariant = std::variant<RootT *, const RootT *>;

public:
  RootVariant Root = static_cast<RootT *>(nullptr);
  TupleTreePath Path;

public:
  static TupleTreeReference
  fromPath(ConstOrNot<TupleTreeReference::RootT> auto *Root,
           const TupleTreePath &Path) {
    return TupleTreeReference{ .Root = RootVariant{ Root }, .Path = Path };
  }

  static TupleTreeReference
  fromString(ConstOrNot<TupleTreeReference::RootT> auto *Root,
             llvm::StringRef Path) {
    std::optional<TupleTreePath> OptionalPath = stringAsPath<RootT>(Path);
    if (not OptionalPath.has_value())
      return TupleTreeReference{};
    return fromPath(Root, *OptionalPath);
  }

  bool operator==(const TupleTreeReference &Other) const {
    // The paths are the same even if they are referred to different roots
    return Path == Other.Path;
  }

  bool operator<(const TupleTreeReference &Other) const {
    // The paths are the same even if they are referred to different roots
    return Path < Other.Path;
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

    if (Path.size() == 0)
      return nullptr;

    const auto GetByPathVisitor = [&Path = Path](const auto &RootPointer) {
      return getByPath<T>(Path, *RootPointer);
    };

    return std::visit(GetByPathVisitor, Root);
  }

  T *get() {
    revng_assert(canGet());

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

  bool isValid() const debug_function {
    return not hasNullRoot() and not Path.empty() and get() != nullptr;
  }
};

template<typename T>
concept IsTupleTreeReference = is_specialization_v<T, TupleTreeReference>;

template<IsTupleTreeReference T>
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

template<TupleTreeCompatible T>
class TupleTree {
private:
  std::unique_ptr<T> Root;

public:
  TupleTree() : Root(new T) {}

  // Prevent accidental copy
  TupleTree(const TupleTree &Other) = delete;
  TupleTree &operator=(const TupleTree &Other) = delete;

  // Moving is fine
  TupleTree(TupleTree &&Other) = default;
  TupleTree &operator=(TupleTree &&Other) = default;

  // Explicit cloning
  TupleTree clone(const TupleTree &Other) const {
    TupleTree Result;

    // Copy the root
    Result.Root.reset(new T(*Root));

    // Update references to root
    Result.initializeReferences();

    return Result;
  }

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
    YAMLInput >> *Result.Root;

    std::error_code EC = YAMLInput.error();
    if (EC)
      return EC;

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
  auto get() const noexcept { return Root.get(); }
  auto &operator*() const { return *Root; }
  auto *operator->() const noexcept { return Root.operator->(); }

public:
  bool verify() const debug_function { return verifyReferences(); }

  void initializeReferences() {
    visitReferences([this](auto &Element) { Element.Root = Root.get(); });
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

  template<typename L>
  void visitReferences(const L &InnerVisitor) {
    auto Visitor = [&InnerVisitor](auto &Element) {
      using type = std::remove_cvref_t<decltype(Element)>;
      if constexpr (IsTupleTreeReference<type>)
        InnerVisitor(Element);
    };

    visitTupleTree(*Root, Visitor, [](auto) {});
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
};

template<typename E>
struct NamedEnumScalarTraits {
  template<typename T>
  static void enumeration(T &IO, E &V) {
    for (unsigned I = 0; I < E::Count; ++I) {
      auto Value = static_cast<E>(I);
      IO.enumCase(V, getName(Value).data(), Value);
    }
  }
};

/// \brief Specialization for the std::variant we have in TupleTreeReference
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
