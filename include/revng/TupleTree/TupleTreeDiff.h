#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <optional>
#include <set>
#include <utility>
#include <variant>
#include <vector>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/ADT/KeyedObjectContainer.h"
#include "revng/ADT/STLExtras.h"
#include "revng/ADT/ZipMapIterator.h"
#include "revng/Support/Assert.h"
#include "revng/TupleTree/TupleTree.h"
#include "revng/TupleTree/TupleTreePath.h"

template<typename>
struct is_std_vector : std::false_type {};

template<typename T, typename A>
struct is_std_vector<std::vector<T, A>> : std::true_type {};

template<typename T>
concept HasPushBack = is_std_vector<T>::value;

template<typename T>
concept HasInsertOrAssign = not HasPushBack<T>;

template<HasInsertOrAssign C>
void addToContainer(C &Container, const typename C::value_type &Value) {
  Container.insert_or_assign(Value);
}

template<HasPushBack C>
void addToContainer(C &Container, const typename C::value_type &Value) {
  Container.push_back(Value);
}

template<typename T>
struct TupleTreeEntries;

template<typename T>
using TupleTreeEntriesT = typename TupleTreeEntries<T>::Types;

namespace detail {
template<typename Model>
struct CheckTypeIsCorrect {
  using Variant = TupleTreeEntriesT<Model>;
  const Variant *Alternatives;
  bool IsCorrect = false;

  template<typename T, int I>
  void visitTupleElement() {
    using tuple_element = typename std::tuple_element<I, T>::type;
    visit<tuple_element>();
  }

  template<typename T, typename KeyT>
  void visitContainerElement(KeyT Key) {}

  template<SortedContainer T>
  void visit() {
    check<typename T::value_type>();
  }

  template<typename T>
  void visit() {
    check<T>();
  }

  template<typename T>
  void check() {
    IsCorrect = std::holds_alternative<T>(*Alternatives);
  }
};

template<typename Model>
bool checkTypeIsCorrect(const TupleTreePath &Path,
                        const TupleTreeEntriesT<Model> &Content) {
  detail::CheckTypeIsCorrect<Model> Checker{ &Content };
  callByPath<Model>(Checker, Path);
  return Checker.IsCorrect;
}

} // namespace detail

template<typename T>
struct Change {
public:
  using EntryType = std::optional<TupleTreeEntriesT<T>>;

public:
  TupleTreePath Path;
  EntryType Old = std::nullopt;
  EntryType New = std::nullopt;

public:
  Change() = default;
  using TupleTreeType = T;
  explicit Change(TupleTreePath Path,
                  std::optional<TupleTreeEntriesT<T>> Old,
                  std::optional<TupleTreeEntriesT<T>> New) :
    Path(std::move(Path)), Old(std::move(Old)), New(std::move(New)) {}

public:
  static Change createRemoval(TupleTreePath Path, TupleTreeEntriesT<T> Old) {
    revng_check(detail::checkTypeIsCorrect<T>(Path, Old));
    return Change(std::move(Path), std::move(Old), std::nullopt);
  }

  static Change createAddition(TupleTreePath Path, TupleTreeEntriesT<T> New) {
    revng_check(detail::checkTypeIsCorrect<T>(Path, New));
    return Change(std::move(Path), std::nullopt, std::move(New));
  }

  static Change createChange(TupleTreePath Path,
                             TupleTreeEntriesT<T> Old,
                             TupleTreeEntriesT<T> New) {
    revng_check(detail::checkTypeIsCorrect<T>(Path, New));
    revng_check(detail::checkTypeIsCorrect<T>(Path, Old));
    return Change(std::move(Path), std::move(Old), std::move(New));
  }
};

template<typename T>
using ChangesVector = std::vector<Change<T>>;

template<typename T>
struct TupleTreeDiff {

public:
  using Change = Change<T>;
  ChangesVector<T> Changes;
  // TODO: invalidated instances

public:
  TupleTreeDiff invert() const {
    TupleTreeDiff Result = *this;
    for (Change &C : Result.Changes) {
      std::swap(C.Old, C.New);
    }
    return Result;
  }

public:
  template<typename ToAdd>
  void add(const TupleTreePath &Path, ToAdd What) {
    Changes.push_back(Change::createAddition(Path, What));
  }

  template<typename ToRemove>
  void remove(const TupleTreePath &Path, ToRemove What) {
    Changes.push_back(Change::createRemoval(Path, What));
  }

  template<typename ToChange>
  void
  change(const TupleTreePath &Path, const ToChange &From, const ToChange &To) {
    Changes.push_back(Change::createChange(Path, From, To));
  }

public:
  void dump(llvm::raw_ostream &OutputStream) const;

  void dump() const {
    llvm::raw_os_ostream OutputStream(dbg);
    dump(OutputStream);
  }

  void apply(TupleTree<T> &M) const;
};

template<typename T>
concept DiffSpecialization = is_specialization_v<T, TupleTreeDiff>;

template<typename T>
concept ChangeSpecialization = is_specialization_v<T, Change>;

template<DiffSpecialization T>
struct llvm::yaml::MappingTraits<T> {
  static void mapping(IO &IO, T &Info) {
    IO.mapOptional("Changes", Info.Changes);
  }
};

template<ChangeSpecialization T, typename X>
struct llvm::yaml::SequenceElementTraits<T, X> {
  // NOLINTNEXTLINE
  static const bool flow = false;
};

namespace detail {
template<typename Model>
struct MapDiffVisitor {
  llvm::yaml::IO *Io;
  TupleTreeEntriesT<Model> *Change;
  const char *MappingName;

  template<typename T, int I>
  void visitTupleElement() {
    using tuple_element = typename std::tuple_element<I, T>::type;
    visit<tuple_element>();
  }

  template<typename T, typename KeyT>
  void visitContainerElement(KeyT Key) {}

  template<SortedContainer T>
  void visit() {
    dump<typename T::value_type>();
  }

  template<typename T>
  void visit() {
    dump<T>();
  }

  template<typename T>
  void dump() {
    if (Io->outputting()) {
      Io->mapRequired(MappingName, std::get<T>(*Change));
    } else {
      T Content;
      Io->mapRequired(MappingName, Content);
      *Change = std::move(Content);
    }
  }
};

} // namespace detail

template<ChangeSpecialization T>
struct llvm::yaml::MappingTraits<T> {
  using EntryType = typename T::EntryType;
  using Model = typename T::TupleTreeType;

  static void writeEntry(IO &IO, T &Info, const char *Name, EntryType &Entry) {
    if (not Entry.has_value())
      return;

    ::detail::MapDiffVisitor<Model> Visitor{ &IO, &*Entry, Name };
    callByPath<Model>(Visitor, Info.Path);
  }

  static void readEntry(IO &IO, T &Info, const char *Name, EntryType &Entry) {
    const auto &Keys = IO.keys();
    if (llvm::find(Keys, Name) == Keys.end())
      return;

    Entry = false;
    ::detail::MapDiffVisitor<Model> Visitor{ &IO, &*Entry, Name };
    callByPath<Model>(Visitor, Info.Path);
  }

  static void
  mapSingleEntry(IO &IO, T &Info, const char *Name, EntryType &Entry) {
    if (IO.outputting())
      writeEntry(IO, Info, Name, Entry);
    else
      readEntry(IO, Info, Name, Entry);
  }

  static void mapping(IO &IO, T &Info) {
    if (IO.outputting()) {
      std::string SerializedPath = *pathAsString<Model>(Info.Path);
      IO.mapRequired("Path", SerializedPath);
    } else {
      std::string SerializedPath;
      IO.mapRequired("Path", SerializedPath);
      auto MaybePath = stringAsPath<Model>(SerializedPath);
      revng_assert(MaybePath.has_value());
      Info.Path = std::move(*MaybePath);
    }

    mapSingleEntry(IO, Info, "Add", Info.New);
    mapSingleEntry(IO, Info, "Remove", Info.Old);
  }
};

//
// diff
//
namespace tupletreediff::detail {

template<typename M>
struct Diff {
  TupleTreePath Stack;
  TupleTreeDiff<M> Result;

  TupleTreeDiff<M> diff(M &LHS, M &RHS) {
    diffImpl(LHS, RHS);
    return Result;
  }

private:
  template<size_t I = 0, typename T>
  void diffTuple(T &LHS, T &RHS) {
    if constexpr (I < std::tuple_size_v<T>) {

      Stack.push_back(size_t(I));
      diffImpl(get<I>(LHS), get<I>(RHS));
      Stack.pop_back();

      // Recur
      diffTuple<I + 1>(LHS, RHS);
    }
  }

  template<IsUpcastablePointer T>
  void diffImpl(T &LHS, T &RHS) {
    LHS.upcast([&](auto &LHSUpcasted) {
      RHS.upcast([&](auto &RHSUpcasted) {
        using LHSType = std::remove_cvref_t<decltype(LHSUpcasted)>;
        using RHSType = std::remove_cvref_t<decltype(RHSUpcasted)>;
        if constexpr (std::is_same_v<LHSType, RHSType>) {
          diffImpl(LHSUpcasted, RHSUpcasted);
        } else {
          Result.change(Stack, LHS, RHS);
        }
      });
    });
  }

  template<HasTupleSize T>
  void diffImpl(T &LHS, T &RHS) {
    diffTuple(LHS, RHS);
  }

  template<SortedContainer T>
  void diffImpl(T &LHS, T &RHS) {
    for (auto [LHSElement, RHSElement] : zipmap_range(LHS, RHS)) {
      if (LHSElement == nullptr) {
        // Added
        Result.add(Stack, *RHSElement);
      } else if (RHSElement == nullptr) {
        // Removed
        Result.remove(Stack, *LHSElement);
      } else {
        // Identical
        using value_type = typename T::value_type;
        Stack.push_back(KeyedObjectTraits<value_type>::key(*LHSElement));
        diffImpl(*LHSElement, *RHSElement);
        Stack.pop_back();
      }
    }
  }

  template<NotTupleTreeCompatible T>
  void diffImpl(T &LHS, T &RHS) {
    if (LHS != RHS)
      Result.change(Stack, LHS, RHS);
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

template<typename T>
inline void TupleTreeDiff<T>::dump(llvm::raw_ostream &OutputStream) const {
  serialize(OutputStream, *this);
}

//
// TupleTreeDiff::apply
//
namespace tupletreediff::detail {

// clang-format off

// clang-format on

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

  template<SortedContainer S>
  void visit(S &M) {
    revng_assert((C->Old == std::nullopt) != (C->New == std::nullopt));

    using value_type = typename S::value_type;
    using KOT = KeyedObjectTraits<value_type>;
    using key_type = decltype(KOT::key(std::declval<value_type>()));

    size_t OldSize = M.size();
    if (C->Old != std::nullopt) {
      key_type Key = KOT::key(std::get<value_type>(*C->Old));
      auto End = M.end();
      auto CompareKeys = [Key](value_type &V) { return KOT::key(V) == Key; };
      auto FirstToDelete = std::remove_if(M.begin(), End, CompareKeys);
      M.erase(FirstToDelete, End);
      revng_assert(OldSize == M.size() + 1);
    } else if (C->New != std::nullopt) {
      // TODO: assert not there already
      addToContainer(M, std::get<value_type>(*C->New));
      revng_assert(OldSize == M.size() - 1);
    } else {
      revng_abort();
    }
  }

  template<typename S>
  void visit(S &M) requires(not SortedContainer<S>) {
    revng_assert(C->Old != std::nullopt and C->New != std::nullopt);
    auto &Old = std::get<S>(*C->Old);
    auto &New = std::get<S>(*C->New);
    revng_check(Old == M);
    M = New;
  }
};

} // namespace tupletreediff::detail

template<typename T>
inline void TupleTreeDiff<T>::apply(TupleTree<T> &M) const {
  TupleTreePath LastPath;
  for (const Change &C : Changes) {
    tupletreediff::detail::ApplyDiffVisitor<T> ADV{ &C };
    callByPath(ADV, C.Path, *M);
  }
  M.initializeReferences();
}
