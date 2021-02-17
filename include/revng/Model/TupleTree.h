#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <array>
#include <set>
#include <type_traits>
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/YAMLTraits.h"

#include "revng/ADT/KeyTraits.h"
#include "revng/ADT/KeyedObjectContainer.h"
#include "revng/ADT/KeyedObjectTraits.h"
#include "revng/Support/Assert.h"
#include "revng/Support/Debug.h"
#include "revng/Support/YAMLTraits.h"

//
// has_yaml
//
namespace tupletree::detail {

using namespace llvm::yaml;

template<typename T>
constexpr bool has_yaml_v = has_DocumentListTraits<T>::value
                            or has_MappingTraits<T, EmptyContext>::value
                            or has_SequenceTraits<T>::value
                            or has_BlockScalarTraits<T>::value
                            or has_CustomMappingTraits<T>::value
                            or has_PolymorphicTraits<T>::value
                            or has_ScalarTraits<T>::value
                            or has_ScalarEnumerationTraits<T>::value;

struct NoYaml {};

} // namespace tupletree::detail

template<typename T>
constexpr bool has_yaml_v = tupletree::detail::has_yaml_v<T>;

static_assert(!has_yaml_v<tupletree::detail::NoYaml>);
static_assert(has_yaml_v<int>);
static_assert(has_yaml_v<std::vector<int>>);

template<typename T, typename K = void>
using enable_if_has_yaml_t = std::enable_if_t<has_yaml_v<T>, K>;

template<typename T, typename K = void>
using enable_if_has_not_yaml_t = std::enable_if_t<not has_yaml_v<T>, K>;

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
  // static const char *name();

  // template<size_t I=0>
  // static const char *fieldName();
};

//
// Implementation of MappingTraits for TupleLikeTraits implementors
//

/// Tuple-liek can implement llvm::yaml::MappingTraits inheriting this class
template<typename T>
struct TupleLikeMappingTraits {
  // Recursive step
  template<size_t I = 0>
  static void mapping(llvm::yaml::IO &io, T &Obj) {
    // Define the field using getTupleFieldName and the associated field
    io.mapRequired(TupleLikeTraits<T>::template fieldName<I>(), get<I>(Obj));

    // Recur
    mapping<I + 1>(io, Obj);
  }

  // Base case
  template<>
  void mapping<std::tuple_size_v<T>>(llvm::yaml::IO &io, T &Obj) {}
};

//
// visit implementation
//

namespace tupletree::detail {

template<size_t I = 0, typename Visitor, typename T>
enable_if_tuple_end_t<I, T> visitTuple(Visitor &V, T &Obj) {
}

template<size_t I = 0, typename Visitor, typename T>
enable_if_not_tuple_end_t<I, T> visitTuple(Visitor &V, T &Obj) {
  // Visit the field
  visit(V, get<I>(Obj));

  // Visit next element in tuple
  visitTuple<I + 1>(V, Obj);
}

} // namespace tupletree::detail

// Tuple-like
template<typename Visitor, typename T>
enable_if_has_tuple_size_t<T, void> visit(Visitor &V, T &Obj) {
  V.preVisit(Obj);
  tupletree::detail::visitTuple(V, Obj);
  V.postVisit(Obj);
}

// Container-like
template<typename Visitor, typename T>
enable_if_is_container_t<T> visit(Visitor &V, T &Obj) {
  V.preVisit(Obj);
  using value_type = typename T::value_type;
  for (value_type &Element : Obj) {
    visit(V, Element);
  }
  V.postVisit(Obj);
}

// All the others
template<typename Visitor, typename T>
std::enable_if_t<not(is_container_v<T> or has_tuple_size_v<T>)>
visit(Visitor &V, T &Element) {
  V.preVisit(Element);
  V.postVisit(Element);
}

/// Default visitor, doing nothing
struct DefaultTupleTreeVisitor {
  template<typename T>
  void preVisit(T &) {}

  template<typename T>
  void postVisit(T &) {}
};

//
// tupleIndexByName
//
template<typename T, size_t I = 0>
enable_if_tuple_end_t<I, T, size_t> tupleIndexByName(llvm::StringRef Name) {
  return -1;
}

template<typename T, size_t I = 0>
enable_if_not_tuple_end_t<I, T, size_t> tupleIndexByName(llvm::StringRef Name) {
  llvm::StringRef ThisName = TupleLikeTraits<T>::template fieldName<I>();
  if (Name == ThisName)
    return I;
  else
    return tupleIndexByName<T, I + 1>(Name);
}

//
// getByKey
//
namespace tupletree::detail {

template<typename ResultT, size_t I = 0, typename RootT, typename KeyT>
enable_if_tuple_end_t<I, RootT, ResultT *> getByKeyTuple(RootT &M, KeyT Key) {
  return nullptr;
}

template<typename ResultT, size_t I = 0, typename RootT, typename KeyT>
enable_if_not_tuple_end_t<I, RootT, ResultT *>
getByKeyTuple(RootT &M, KeyT Key) {
  if (I == Key) {
    using tuple_element = typename std::tuple_element<I, RootT>::type;
    revng_assert((std::is_same_v<tuple_element, ResultT>) );
    return reinterpret_cast<ResultT *>(&get<I>(M));
  } else {
    return getByKeyTuple<ResultT, I + 1>(M, Key);
  }
}

} // namespace tupletree::detail

template<typename ResultT, typename RootT, typename KeyT>
enable_if_has_tuple_size_t<RootT, ResultT *> getByKey(RootT &M, KeyT Key) {
  return tupletree::detail::getByKeyTuple<ResultT>(M, Key);
}

template<typename ResultT, typename RootT, typename KeyT>
enable_if_is_container_t<RootT, ResultT *> getByKey(RootT &M, KeyT Key) {
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
template<typename RootT, typename Visitor>
enable_if_has_tuple_size_t<RootT, bool>
callOnPathSteps(Visitor &V, llvm::ArrayRef<KeyInt> Path);

template<typename RootT, typename Visitor>
enable_if_is_not_container_or_tuple_t<RootT, bool>
callOnPathSteps(Visitor &V, llvm::ArrayRef<KeyInt> Path) {
  return false;
}

template<typename RootT, typename Visitor>
enable_if_is_container_t<RootT, bool>
callOnPathSteps(Visitor &V, llvm::ArrayRef<KeyInt> Path) {
  using value_type = typename RootT::value_type;
  using KOT = KeyedObjectTraits<value_type>;
  using key_type = decltype(KOT::key(std::declval<value_type>()));
  constexpr size_t IntsCount = KeyTraits<key_type>::IntsCount;
  auto PathStep = slice<0, IntsCount>(Path);
  auto TargetKey = KeyTraits<key_type>::fromInts(PathStep);

  V.template visitContainerElement<RootT>(TargetKey);
  if (Path.size() > IntsCount) {
    return callOnPathSteps<typename RootT::value_type>(V,
                                                       Path.slice(IntsCount));
  }

  return true;
}

namespace tupletree::detail {

template<typename RootT, size_t I = 0, typename Visitor>
enable_if_tuple_end_t<I, RootT, bool>
callOnPathStepsTuple(Visitor &V, llvm::ArrayRef<KeyInt> Path) {
  return true;
}

template<typename RootT, size_t I = 0, typename Visitor>
enable_if_not_tuple_end_t<I, RootT, bool>
callOnPathStepsTuple(Visitor &V, llvm::ArrayRef<KeyInt> Path) {
  if (Path[0] == I) {
    using next_type = typename std::tuple_element<I, RootT>::type;
    V.template visitTupleElement<RootT, I>();
    if (Path.size() > 1) {
      return callOnPathSteps<next_type>(V, Path.slice(1));
    }
  } else {
    return callOnPathStepsTuple<RootT, I + 1>(V, Path);
  }

  return true;
}

} // namespace tupletree::detail

template<typename RootT, typename Visitor>
enable_if_has_tuple_size_t<RootT, bool>
callOnPathSteps(Visitor &V, llvm::ArrayRef<KeyInt> Path) {
  return tupletree::detail::callOnPathStepsTuple<RootT>(V, Path);
}

//
// callOnPathSteps (with instance)
//

namespace tupletree::detail {

template<size_t I = 0, typename RootT, typename Visitor>
enable_if_tuple_end_t<I, RootT, bool>
callOnPathStepsTuple(Visitor &V, llvm::ArrayRef<KeyInt> Path, RootT &M) {
  return true;
}

template<typename RootT, typename Visitor>
enable_if_is_not_container_or_tuple_t<RootT, bool>
callOnPathSteps(Visitor &V, llvm::ArrayRef<KeyInt> Path, RootT &M) {
  return false;
}

template<size_t I = 0, typename RootT, typename Visitor>
enable_if_not_tuple_end_t<I, RootT, bool>
callOnPathStepsTuple(Visitor &V, llvm::ArrayRef<KeyInt> Path, RootT &M) {
  if (Path[0] == I) {
    using next_type = typename std::tuple_element<I, RootT>::type;
    next_type &Element = get<I>(M);
    V.template visitTupleElement<RootT, I>(Element);
    if (Path.size() > 1) {
      return callOnPathSteps(V, Path.slice(1), Element);
    }
  } else {
    return callOnPathStepsTuple<I + 1>(V, Path, M);
  }

  return true;
}

} // namespace tupletree::detail

template<typename RootT, typename Visitor>
enable_if_has_tuple_size_t<RootT, bool>
callOnPathSteps(Visitor &V, llvm::ArrayRef<KeyInt> Path, RootT &M) {
  return tupletree::detail::callOnPathStepsTuple(V, Path, M);
}

template<typename RootT, typename Visitor>
enable_if_is_container_t<RootT, bool>
callOnPathSteps(Visitor &V, llvm::ArrayRef<KeyInt> Path, RootT &M) {
  using value_type = typename RootT::value_type;
  using KOT = KeyedObjectTraits<value_type>;
  using key_type = decltype(KOT::key(std::declval<value_type>()));
  constexpr size_t IntsCount = KeyTraits<key_type>::IntsCount;
  auto PathStep = slice<0, IntsCount>(Path);
  auto TargetKey = KeyTraits<key_type>::fromInts(PathStep);

  value_type *Matching = nullptr;
  for (value_type &Element : M) {
    using KOT = KeyedObjectTraits<value_type>;
    if (KOT::key(Element) == TargetKey) {
      Matching = &Element;
      break;
    }
  }

  if (Matching == nullptr)
    return false;

  V.template visitContainerElement<RootT>(TargetKey, *Matching);
  if (Path.size() > IntsCount) {
    return callOnPathSteps(V, Path.slice(IntsCount), *Matching);
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
    constexpr size_t IntsCount = KeyTraits<KeyT>::IntsCount;
    PathSize -= IntsCount;
    if (PathSize == 0)
      V.template visitContainerElement<T>(Key);
  }
};

} // namespace tupletree::detail

template<typename RootT, typename Visitor>
bool callByPath(Visitor &V, const KeyIntVector &Path) {
  using namespace tupletree::detail;
  CallByPathVisitor<Visitor> CBPV{ Path.size(), V };
  return callOnPathSteps<RootT>(CBPV, Path);
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

  template<typename T, typename K, typename KeyT>
  void visitContainerElement(KeyT Key, K &Element) {
    constexpr size_t IntsCount = KeyTraits<KeyT>::IntsCount;
    PathSize -= IntsCount;
    if (PathSize == 0)
      V.template visitContainerElement<T>(Key, Element);
  }
};

} // namespace tupletree::detail

template<typename RootT, typename Visitor>
bool callByPath(Visitor &V, const KeyIntVector &Path, RootT &M) {
  using namespace tupletree::detail;
  CallByPathVisitorWithInstance<Visitor> CBPV{ Path.size(), V };
  return callOnPathSteps(CBPV, Path, M);
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
ResultT *getByPath(const KeyIntVector &Path, RootT &M) {
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
    Stream << "/" << TupleLikeTraits<T>::template fieldName<I>();
  }

  template<typename T, typename KeyT>
  void visitContainerElement(KeyT Key) {
    Stream << "/" << getNameFromYAMLScalar(Key);
  }
};

} // namespace tupletree::detail

template<typename T>
std::optional<std::string> pathAsString(const KeyIntVector &Path) {
  std::string Result;
  {
    tupletree::detail::DumpPathVisitor PV(Result);
    if (not callOnPathSteps<T>(PV, Path))
      return {};
  }
  return Result;
}

class PathMatcher {
private:
  KeyIntVector Path;
  std::vector<std::pair<size_t, size_t>> Free;

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
  const KeyIntVector &path() const { return Path; }

public:
  template<typename... Ts>
  KeyIntVector apply(Ts... Args) const {
    revng_assert(sizeof...(Args) == Free.size());
    KeyIntVector Result = Path;
    applyImpl<0, Ts...>(Result, Args...);
    return Result;
  }

  template<typename... Args>
  std::optional<std::tuple<Args...>> match(const KeyIntVector &Search) {
    revng_assert(sizeof...(Args) == Free.size());

    if (Path.size() != Search.size())
      return {};

    //
    // Check non-variable parts match
    //
    using Pair = std::pair<KeyInt, KeyInt>;
    std::vector<Pair> Terminator{ { Path.size(), 0 } };

    size_t LastEnd = 0;
    for (auto [Start, Size] : llvm::concat<Pair>(Free, Terminator)) {
      for (size_t I = LastEnd; I < Start; ++I) {
        if (Search[I] != Path[I])
          return {};
      }

      LastEnd = Start + Size;
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
  void depositKey(KeyIntVector &Result, T Arg) const {
    auto [Start, Size] = Free.at(I);
    revng_assert(Size == KeyTraits<T>::IntsCount);

    for (auto P : llvm::enumerate(KeyTraits<T>::toInts(Arg)))
      Result[Start + P.index()] = P.value();
  }

  template<size_t I, typename T>
  void applyImpl(KeyIntVector &Result, T Arg) const {
    depositKey<I>(Result, Arg);
  }

  template<size_t I, typename T, typename... Ts>
  void applyImpl(KeyIntVector &Result, T Arg, Ts... Args) const {
    depositKey<I>(Result, Arg);
    applyImpl<I + 1, Ts...>(Result, Args...);
  }

  template<typename T, size_t I = 0>
  void extractKeys(const KeyIntVector &Search, T &Tuple) const {
    if constexpr (I < std::tuple_size_v<T>) {
      using element = std::tuple_element_t<I, T>;
      using IntsArray = typename KeyTraits<element>::IntsArray;

      IntsArray Ints;
      auto [Start, Size] = Free[I];
      for (auto P : llvm::enumerate(Ints))
        P.value() = Search[Start + P.index()];

      std::get<I>(Tuple) = KeyTraits<element>::fromInts(Ints);

      extractKeys<T, I + 1>(Search, Tuple);
    }
  }

private:
  template<typename T, size_t I = 0>
  static bool visitTuple(llvm::StringRef Current,
                         llvm::StringRef Rest,
                         PathMatcher &Result);

  template<typename T>
  static enable_if_has_tuple_size_t<T, bool>
  visitTupleTreeNode(llvm::StringRef String, PathMatcher &Result);

  template<typename T>
  static enable_if_is_container_t<T, bool>
  visitTupleTreeNode(llvm::StringRef String, PathMatcher &Result);

  template<typename T>
  static enable_if_is_not_container_or_tuple_t<T, bool>
  visitTupleTreeNode(llvm::StringRef Path, PathMatcher &Result);
};

template<typename T>
enable_if_has_tuple_size_t<T, bool>
PathMatcher::visitTupleTreeNode(llvm::StringRef String, PathMatcher &Result) {
  if (String.size() == 0)
    return true;

  auto [Before, After] = String.split('/');
  return visitTuple<T>(Before, After, Result);
}

template<typename T>
enable_if_is_container_t<T, bool>
PathMatcher::visitTupleTreeNode(llvm::StringRef String, PathMatcher &Result) {
  if (String.size() == 0)
    return true;

  auto [Before, After] = String.split('/');

  using Key = std::remove_cv_t<typename T::key_type>;
  using Value = typename T::value_type;

  if (Before == "*") {
    auto Count = KeyTraits<Key>::IntsCount;
    Result.Free.push_back({ Result.Path.size(), Count });
    for (size_t I = 0; I < Count; ++I)
      Result.Path.push_back(0);
  } else {
    for (KeyInt I : KeyTraits<Key>::toInts(getValueFromYAMLScalar<Key>(Before)))
      Result.Path.push_back(I);
  }

  return visitTupleTreeNode<Value>(After, Result);
}

template<typename T>
enable_if_is_not_container_or_tuple_t<T, bool>
PathMatcher::visitTupleTreeNode(llvm::StringRef Path, PathMatcher &Result) {
  return Path.size() == 0;
}

template<typename T, size_t I>
bool PathMatcher::visitTuple(llvm::StringRef Current,
                             llvm::StringRef Rest,
                             PathMatcher &Result) {
  if constexpr (I < std::tuple_size_v<T>) {
    if (TupleLikeTraits<T>::template fieldName<I>() == Current) {
      Result.Path.push_back(I);
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
std::optional<KeyIntVector> stringAsPath(llvm::StringRef Path) {
  auto Result = PathMatcher::create<T>(Path);
  if (Result)
    return Result->path();
  else
    return {};
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
#define TUPLE_ELEMENTS(class, index, field) \
  template<>                                \
  struct std::tuple_element<index, class> { \
    using type = decltype(class ::field);   \
  };

#define GET_IMPLEMENTATIONS(class, index, field) \
  else if constexpr (I == index) return x.field;

#define GET_TUPLE_FIELD_NAME(class, index, field) \
  template<>                                      \
  const char *fieldName<index>() {                \
    return #field;                                \
  }

#define INTROSPECTION_1(class, ...)                            \
  template<>                                                   \
  struct std::tuple_size<class>                                \
    : std::integral_constant<size_t, NUMARGS(__VA_ARGS__)> {}; \
                                                               \
  FOR_EACH(TUPLE_ELEMENTS, class, __VA_ARGS__)                 \
                                                               \
  template<>                                                   \
  struct TupleLikeTraits<class> {                              \
    static const char *name() { return #class; }               \
                                                               \
    template<size_t I = 0>                                     \
    static const char *fieldName();                            \
                                                               \
    FOR_EACH(GET_TUPLE_FIELD_NAME, class, __VA_ARGS__)         \
  };

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
