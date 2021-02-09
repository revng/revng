#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <set>
#include <vector>

#include "llvm/Support/WithColor.h"

#include "revng/ADT/KeyedObjectContainer.h"
#include "revng/ADT/ZipMapIterator.h"
#include "revng/Model/TupleTree.h"

template<typename>
struct is_std_vector : std::false_type {};

template<typename T, typename A>
struct is_std_vector<std::vector<T, A>> : std::true_type {};

template<typename T>
constexpr bool has_push_back_v = is_std_vector<T>::value;

template<typename T, typename K = void>
using enable_if_has_push_back_t = std::enable_if_t<has_push_back_v<T>, K>;

template<typename T>
constexpr bool has_insert_or_assign_v = not has_push_back_v<T>;

namespace detail {
template<typename T, typename K = void>
using ei_hioa_t = std::enable_if_t<has_insert_or_assign_v<T>, K>;
}

template<typename T, typename K = void>
using enable_if_has_insert_or_assign_t = detail::ei_hioa_t<T, K>;

template<typename C>
enable_if_has_insert_or_assign_t<C>
addToContainer(C &Container, const typename C::value_type &Value) {
  Container.insert_or_assign(Value);
}

template<typename C>
enable_if_has_push_back_t<C>
addToContainer(C &Container, const typename C::value_type &Value) {
  Container.push_back(Value);
}

template<typename T>
struct TupleTreeDiff {
  struct Change {
    KeyIntVector Path;
    void *Old;
    void *New;
  };

  std::vector<Change> Changes;
  // TODO: invalidated instances

  TupleTreeDiff invert() const {
    TupleTreeDiff Result = *this;
    for (Change &C : Result.Changes) {
      std::swap(C.Old, C.New);
    }
    return Result;
  }

  void add(const KeyIntVector &Path, void *What) {
    Changes.push_back({ Path, nullptr, What });
  }
  void remove(const KeyIntVector &Path, void *What) {
    Changes.push_back({ Path, What, nullptr });
  }
  void change(const KeyIntVector &Path, void *From, void *To) {
    Changes.push_back({ Path, From, To });
  }

  void dump() const;
  void apply(T &M) const;
};

//
// diff
//
namespace tupletreediff::detail {

template<typename M>
struct Diff {
  KeyIntVector Stack;
  TupleTreeDiff<M> Result;

  TupleTreeDiff<M> diff(M &LHS, M &RHS) {
    diffImpl(LHS, RHS);
    return Result;
  }

private:
  template<size_t I = 0, typename T>
  enable_if_tuple_end_t<I, T> diffTuple(T &LHS, T &RHS) {}

  template<size_t I = 0, typename T>
  enable_if_not_tuple_end_t<I, T> diffTuple(T &LHS, T &RHS) {
    using child_type = typename std::tuple_element<I, T>::type;

    Stack.push_back(I);
    diffImpl(get<I>(LHS), get<I>(RHS));
    Stack.pop_back();

    // Recur
    diffTuple<I + 1>(LHS, RHS);
  }

  template<typename T>
  enable_if_has_tuple_size_t<T> diffImpl(T &LHS, T &RHS) {
    diffTuple(LHS, RHS);
  }

  template<typename T>
  enable_if_is_sorted_container_t<T> diffImpl(T &LHS, T &RHS) {
    using value_type = typename T::value_type;
    using KOT = KeyedObjectTraits<value_type>;
    using key_type = decltype(KOT::key(std::declval<value_type>()));

    for (auto [LHSElement, RHSElement] : zipmap_range(LHS, RHS)) {
      if (LHSElement == nullptr) {
        // Added
        Result.add(Stack, RHSElement);
      } else if (RHSElement == nullptr) {
        // Removed
        Result.remove(Stack, LHSElement);
      } else {
        // Identical
        using KT = KeyTraits<key_type>;
        const auto &KeyInts = KT::toInts(KOT::key(*LHSElement));
        std::copy(KeyInts.begin(), KeyInts.end(), std::back_inserter(Stack));

        diffImpl(*LHSElement, *RHSElement);

        // Delete key from the stack
        Stack.resize(Stack.size() - KeyTraits<key_type>::IntsCount);
      }
    }
  }

  template<typename T>
  enable_if_is_unsorted_container_t<T> diffImpl(T &LHS, T &RHS) {
    using value_type = typename T::value_type;
    using KOT = KeyedObjectTraits<value_type>;
    using key_type = decltype(KOT::key(std::declval<value_type>()));
    std::map<key_type, value_type *> LHSMap, RHSMap;
    for (value_type &Element : LHS)
      LHSMap[KOT::key(Element)] = &Element;
    for (value_type &Element : RHS)
      RHSMap[KOT::key(Element)] = &Element;

    for (auto [LHSElement, RHSElement] : zipmap_range(LHSMap, RHSMap)) {
      if (LHSElement == nullptr) {
        // Added
        Result.add(Stack, RHSElement->second);
      } else if (RHSElement == nullptr) {
        // Removed
        Result.remove(Stack, LHSElement->second);
      } else {
        // Identical
        const auto &KeysInt = KeyTraits<key_type>::toInts(LHSElement->first);
        std::copy(KeysInt.begin(), KeysInt.end(), std::back_inserter(Stack));

        diffImpl(*LHSElement->second, *RHSElement->second);

        Stack.resize(Stack.size() - KeyTraits<key_type>::IntsCount);
      }
    }
  }

  template<typename T>
  std::enable_if_t<not(is_container_v<T> or has_tuple_size_v<T>)>
  diffImpl(T &LHS, T &RHS) {
    if (LHS != RHS)
      Result.change(Stack, &LHS, &RHS);
  }
};

} // namespace tupletreediff::detail

template<typename M>
TupleTreeDiff<M> diff(M &LHS, M &RHS) {
  return tupletreediff::detail::Diff<M>().diff(LHS, RHS);
}

//
// TupleTreeDiff::dump
//
namespace tupletreediff::detail {

template<typename T, typename S>
enable_if_has_yaml_t<T> stream(T *M, S &Stream) {
  using namespace llvm::yaml;
  Output Out(Stream);
  EmptyContext Ctx;
  yamlize(Out, *M, true, Ctx);
}

template<typename T, typename S>
enable_if_has_not_yaml_t<T> stream(T *M, S &Stream) {
  Stream << *M;
}

template<typename T>
void dumpWithPrefixAndColor(llvm::StringRef Prefix,
                            llvm::raw_ostream::Colors Color,
                            T *M) {
  std::string Buffer;
  llvm::WithColor Stream(llvm::outs());
  Stream.changeColor(Color);

  {
    llvm::raw_string_ostream StringStream(Buffer);
    stream(M, StringStream);
  }

  auto [LHS, RHS] = llvm::StringRef(Buffer).split('\n');
  while (RHS.size() != 0) {
    Stream << Prefix << LHS << "\n";
    std::tie(LHS, RHS) = RHS.split('\n');
  }

  Stream << Prefix << LHS << "\n";
}

struct DumpDiffVisitor {
  void *Old, *New;

  template<typename T, int I>
  void visitTupleElement() {
    using tuple_element = typename std::tuple_element<I, T>::type;
    visit<tuple_element>();
  }

  template<typename T, typename KeyT>
  void visitContainerElement(KeyT Key) {}

  template<typename T>
  enable_if_is_container_t<T> visit() {
    revng_assert((Old != nullptr) != (New != nullptr));
    dump<typename T::value_type>();
  }

  template<typename T>
  enable_if_is_not_container_t<T> visit() {
    revng_assert(Old != nullptr and New != nullptr);
    dump<T>();
  }

  template<typename T>
  void dump() {

    if (Old != nullptr) {
      dumpWithPrefixAndColor("-",
                             llvm::raw_ostream::RED,
                             reinterpret_cast<T *>(Old));
    }

    if (New != nullptr) {
      dumpWithPrefixAndColor("+",
                             llvm::raw_ostream::GREEN,
                             reinterpret_cast<T *>(New));
    }
  }
};

} // namespace tupletreediff::detail

template<typename T>
inline void TupleTreeDiff<T>::dump() const {
  using namespace tupletreediff::detail;

  KeyIntVector LastPath;
  for (const Change &C : Changes) {

    if (LastPath != C.Path) {
      std::string NewPath = pathAsString<T>(C.Path);
      llvm::outs() << "--- " << NewPath << "\n";
      llvm::outs() << "+++ " << NewPath << "\n";
      LastPath = C.Path;
    }

    DumpDiffVisitor PV2{ C.Old, C.New };
    callByPath<T>(PV2, C.Path);
  }
}

//
// TupleTreeDiff::apply
//
namespace tupletreediff::detail {

template<typename T>
struct ApplyDiffVisitor {
  using Change = typename TupleTreeDiff<T>::Change;
  const Change *C;

  template<typename TupleT, size_t I, typename K>
  void visitTupleElement(K &Element) {
    visit(Element);
  }

  template<typename TupleT, typename K, typename KeyT>
  void visitContainerElement(KeyT, K &Element) {
    visit(Element);
  }

  template<typename S>
  std::enable_if_t<is_iterable_v<S> and !std::is_same_v<std::string, S>>
  visit(S &M) {
    revng_assert((C->Old == nullptr) != (C->New == nullptr));

    using value_type = typename S::value_type;
    using KOT = KeyedObjectTraits<value_type>;
    using key_type = decltype(KOT::key(std::declval<value_type>()));

    size_t OldSize = M.size();
    if (C->Old != nullptr) {
      key_type Key = KOT::key(*reinterpret_cast<value_type *>(C->Old));
      auto End = M.end();
      auto CompareKeys = [Key](value_type &V) { return KOT::key(V) == Key; };
      auto FirstToDelete = std::remove_if(M.begin(), End, CompareKeys);
      M.erase(FirstToDelete, End);
      revng_assert(OldSize == M.size() + 1);
    } else if (C->New != nullptr) {
      // TODO: assert not there already
      addToContainer(M, *reinterpret_cast<value_type *>(C->New));
      revng_assert(OldSize == M.size() - 1);
    } else {
      revng_abort();
    }
  }

  template<typename S>
  std::enable_if_t<!(is_iterable_v<S> and !std::is_same_v<std::string, S>)>
  visit(S &M) {
    revng_assert(C->Old != nullptr and C->New != nullptr);
    auto *Old = reinterpret_cast<S *>(C->Old);
    auto *New = reinterpret_cast<S *>(C->New);
    revng_check(*Old == M);
    M = *New;
  }
};

} // namespace tupletreediff::detail

template<typename T>
inline void TupleTreeDiff<T>::apply(T &M) const {
  KeyIntVector LastPath;
  for (const Change &C : Changes) {
    tupletreediff::detail::ApplyDiffVisitor<T> ADV{ &C };
    callByPath(ADV, C.Path, M);
  }
}
