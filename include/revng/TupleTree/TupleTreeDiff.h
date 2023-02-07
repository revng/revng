#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <any>
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
#include "revng/TupleTree/DiffError.h"
#include "revng/TupleTree/TupleLikeTraits.h"
#include "revng/TupleTree/TupleTree.h"
#include "revng/TupleTree/TupleTreePath.h"

template<typename T>
concept HasValueType = requires(T &&) {
  typename T::value_type;
};

// clang-format off

template<typename T>
concept HasPushBack = HasValueType<T>
                      && requires(T &&C,
                                  const typename T::value_type &V) {
  { C.push_back(V) };
};

template<typename T>
concept HasInsertOrAssign = HasValueType<T>
                            && requires(T &&C,
                                        const typename T::value_type &V) {
  { C.insert_or_assign(V) };
};

// clang-format on

template<HasPushBack C>
void addToContainer(C &Container, const typename C::value_type &Value) {
  Container.push_back(Value);
}

template<HasInsertOrAssign C>
void addToContainer(C &Container, const typename C::value_type &Value) {
  Container.insert_or_assign(Value);
}

namespace revng::detail {
template<typename T>
concept Set = StrictSpecializationOf<T, std::set>;

template<typename T>
concept SetOrKOC = Set<T> || KeyedObjectContainer<T>;
} // namespace revng::detail

template<typename T>
struct TupleTreeEntries {};

template<typename T>
using AllowedTupleTreeTypes = typename TupleTreeEntries<T>::Types;

template<typename T>
concept TupleTreeRootLike = StrictSpecializationOf<AllowedTupleTreeTypes<T>,
                                                   std::variant>;

namespace detail {
  template<TupleTreeRootLike Model>
  struct CheckTypeIsCorrect {
    const AllowedTupleTreeTypes<Model> *Alternatives;
    bool IsCorrect = false;

    template<typename T, int I>
    void visitTupleElement() {
      using tuple_element = typename std::tuple_element<I, T>::type;
      visit<tuple_element>();
    }

    template<typename T, typename KeyT>
    void visitContainerElement(KeyT Key) {}

    template<revng::detail::SetOrKOC T>
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

  template<TupleTreeRootLike Model>
  llvm::Error checkTypeIsCorrect(const TupleTreePath &Path,
                                 const AllowedTupleTreeTypes<Model> &Content) {
    CheckTypeIsCorrect<Model> Checker{ &Content };
    auto Result = callByPath<Model>(Checker, Path);
    revng_assert(Result == true);
    if (not Checker.IsCorrect)
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "Type check has failed");
    return llvm::Error::success();
  }

} // namespace detail

template<TupleTreeRootLike T>
struct Change {
public:
  using Variant = AllowedTupleTreeTypes<T>;
  using EntryType = std::optional<Variant>;

public:
  TupleTreePath Path;
  EntryType Old = std::nullopt;
  EntryType New = std::nullopt;

public:
  Change() = default;
  using TupleTreeType = T;
  explicit Change(TupleTreePath Path, EntryType Old, EntryType New) :
    Path(std::move(Path)), Old(std::move(Old)), New(std::move(New)) {}

public:
  static Change createRemoval(TupleTreePath Path, Variant Old) {
    revng_check(not detail::checkTypeIsCorrect<T>(Path, Old));
    return Change(std::move(Path), std::move(Old), std::nullopt);
  }

  static Change createAddition(TupleTreePath Path, Variant New) {
    revng_check(not detail::checkTypeIsCorrect<T>(Path, New));
    return Change(std::move(Path), std::nullopt, std::move(New));
  }

  static Change createChange(TupleTreePath Path, Variant Old, Variant New) {
    revng_check(not detail::checkTypeIsCorrect<T>(Path, New));
    revng_check(not detail::checkTypeIsCorrect<T>(Path, Old));
    return Change(std::move(Path), std::move(Old), std::move(New));
  }
};

template<TupleTreeRootLike T>
struct TupleTreeDiff {
public:
  static llvm::Expected<TupleTreeDiff<T>> deserialize(llvm::StringRef Input) {
    return ::deserialize<TupleTreeDiff<T>>(Input);
  }

public:
  using Change = Change<T>;
  std::vector<Change> Changes;
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

  llvm::Error apply(TupleTree<T> &M) const;
};

/// TODO: use non-strict specialization after it's available.
template<StrictSpecializationOf<TupleTreeDiff> T>
struct llvm::yaml::MappingTraits<T> {
  static void mapping(IO &IO, T &Info) {
    IO.mapOptional("Changes", Info.Changes);
  }
};

/// TODO: use non-strict specialization after it's available.
template<StrictSpecializationOf<Change> T, typename X>
struct llvm::yaml::SequenceElementTraits<T, X> {
  // NOLINTNEXTLINE
  static const bool flow = false;
};

namespace detail {
template<TupleTreeRootLike Model>
struct MapDiffVisitor {
  llvm::yaml::IO *Io;
  AllowedTupleTreeTypes<Model> *Change;
  const char *MappingName;

  template<typename T, int I>
  void visitTupleElement() {
    using tuple_element = typename std::tuple_element<I, T>::type;
    visit<tuple_element>();
  }

  template<typename T, typename KeyT>
  void visitContainerElement(KeyT Key) {}

  template<revng::detail::SetOrKOC T>
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

/// TODO: use non-strict specialization after it's available.
template<StrictSpecializationOf<Change> T>
struct llvm::yaml::MappingTraits<T> {
  using EntryType = typename T::EntryType;
  using Model = typename T::TupleTreeType;

  static void writeEntry(IO &IO, T &Info, const char *Name, EntryType &Entry) {
    if (not Entry.has_value())
      return;

    ::detail::MapDiffVisitor<Model> Visitor{ &IO, &*Entry, Name };
    bool Result = callByPath<Model>(Visitor, Info.Path);
    revng_assert(Result == true);
  }

  static void readEntry(IO &IO, T &Info, const char *Name, EntryType &Entry) {
    const auto &Keys = IO.keys();
    if (llvm::find(Keys, Name) == Keys.end())
      return;

    Entry.emplace();
    revng_assert(Entry.has_value());
    ::detail::MapDiffVisitor<Model> Visitor{ &IO, &*Entry, Name };
    bool Result = callByPath<Model>(Visitor, Info.Path);
    revng_assert(Result == true);
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
      if (!MaybePath.has_value()) {
        auto *EL = static_cast<revng::DiffError *>(IO.getContext());
        Info.Path = TupleTreePath();
      } else {
        Info.Path = std::move(*MaybePath);
      }
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

  TupleTreeDiff<M> diff(const M &LHS, const M &RHS) {
    diffImpl(LHS, RHS);
    return Result;
  }

private:
  template<size_t I = 0, typename T>
  void diffTuple(const T &LHS, const T &RHS) {
    if constexpr (I < std::tuple_size_v<T>) {

      Stack.push_back(size_t(I));
      diffImpl(get<I>(LHS), get<I>(RHS));
      Stack.pop_back();

      // Recur
      diffTuple<I + 1>(LHS, RHS);
    }
  }

  template<StrictSpecializationOf<UpcastablePointer> T>
  void diffImpl(const T &LHS, const T &RHS) {
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

  template<TupleSizeCompatible T>
  void diffImpl(const T &LHS, const T &RHS) {
    diffTuple(LHS, RHS);
  }

  template<revng::detail::SetOrKOC T>
  void diffImpl(const T &LHS, const T &RHS) {
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
  void diffImpl(const T &LHS, const T &RHS) {
    if (LHS != RHS) {
      Result.change(Stack, LHS, RHS);
    }
  }
};

} // namespace tupletreediff::detail

template<TupleTreeRootLike M>
TupleTreeDiff<M> diff(const M &LHS, const M &RHS) {
  return tupletreediff::detail::Diff<M>().diff(LHS, RHS);
}

//
// TupleTreeDiff::dump
//

template<TupleTreeRootLike T>
inline void TupleTreeDiff<T>::dump(llvm::raw_ostream &OutputStream) const {
  serialize(OutputStream, *this);
}

//
// TupleTreeDiff::apply
//
namespace tupletreediff::detail {

template<TupleTreeRootLike T>
struct ApplyDiffVisitor {
public:
  using Change = typename TupleTreeDiff<T>::Change;
  const Change *C;
  size_t ChangeIndex;
  revng::DiffError *EL;

private:
  void generateError() { generateError(""); }

  void generateError(const llvm::StringRef Reason,
                     revng::DiffLocation::KindType Kind) {
    std::string Description = "Error in applying diff";
    if (!Reason.empty())
      Description += ": " + Reason.str();
    std::optional<std::string> StringPath = pathAsString<T>(C->Path);
    if (StringPath != std::nullopt)
      Description += " on Path " + *StringPath;
    EL->addReason(std::move(Description),
                  revng::DiffLocation(ChangeIndex, Kind));
  }

public:
  template<typename TupleT, size_t I, typename K>
  void visitTupleElement(K &Element) {
    visit(Element);
  }

  template<typename TupleT, typename K, typename KeyT>
  void visitContainerElement(KeyT, K &Element) {
    visit(Element);
  }

  template<revng::detail::SetOrKOC S>
  void visit(S &M) {
    // This visitor handles subtree additions/deletions. Here we either have a
    // New or Old key to add/remove.
    if (C->Old == std::nullopt && C->New == std::nullopt) {
      generateError("Both 'Remove' and 'Add' are not present",
                    revng::DiffLocation::KindType::All);
      return;
    }
    if (C->Old != std::nullopt && C->New != std::nullopt) {
      generateError("Both 'Remove' and 'Add' are not present",
                    revng::DiffLocation::KindType::All);
      return;
    }

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
      if (OldSize - 1 != M.size())
        generateError("Subtree removal failed",
                      revng::DiffLocation::KindType::Old);
    } else if (C->New != std::nullopt) {
      // TODO: assert not there already
      addToContainer(M, std::get<value_type>(*C->New));
      if (OldSize + 1 != M.size())
        generateError("Subtree addition failed",
                      revng::DiffLocation::KindType::New);
    } else {
      generateError("Arrived at an impossible branch",
                    revng::DiffLocation::KindType::Path);
    }
  }

  template<typename S>
  void visit(S &M) {
    // This visitor handles key changes, so both Old and New are present. This
    // will check that the tree contains Old and then replace its contents with
    // New
    if (C->Old == std::nullopt || C->New == std::nullopt) {
      if (C->Old == std::nullopt)
        generateError("Missing 'Remove' key",
                      revng::DiffLocation::KindType::Old);
      if (C->New == std::nullopt)
        generateError("Missing 'Add' key", revng::DiffLocation::KindType::New);
      return;
    }

    auto &Old = std::get<S>(*C->Old);
    auto &New = std::get<S>(*C->New);

    if (Old != M) {
      generateError("'Remove' does not match the contents of the Tuple Tree",
                    revng::DiffLocation::KindType::Old);
      return;
    }

    M = New;
  }
};

} // namespace tupletreediff::detail

template<TupleTreeRootLike T>
inline llvm::Error TupleTreeDiff<T>::apply(TupleTree<T> &M) const {
  auto Error = std::make_unique<revng::DiffError>();
  size_t Index = 0;
  for (const Change &C : Changes) {

    if (C.Path.size() == 0) {
      Error
        ->addReason("Could not deserialize path",
                    revng::DiffLocation(Index,
                                        revng::DiffLocation::KindType::Path));
      continue;
    }
    tupletreediff::detail::ApplyDiffVisitor<T> ADV{ &C, Index, Error.get() };

    if (not callByPath(ADV, C.Path, *M, *pathAsString<T>(C.Path)))
      Error
        ->addReason("Path not present",
                    revng::DiffLocation(Index,
                                        revng::DiffLocation::KindType::Path));
    Index++;
  }

  M.evictCachedReferences();
  M.initializeReferences();
  return revng::DiffError::makeError(std::move(Error));
}
