#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <set>

#include "llvm/Support/YAMLTraits.h"

#include "revng/ADT/KeyedObjectTraits.h"
#include "revng/ADT/STLExtras.h"
#include "revng/ADT/UpcastablePointer.h"
#include "revng/Support/Assert.h"

template<typename T>
using KOTKey = decltype(KeyedObjectTraits<T>::key(std::declval<T>()));

template<typename T>
using KOTCompare = std::less<const KOTKey<T>>;

template<typename T, class Compare = KOTCompare<T>>
class MutableSet;

template<typename T, class Compare = KOTCompare<T>>
class SortedVector;

//
// IsKeyedObjectContainer
//
namespace detail {

template<typename T>
using no_cv_t = std::remove_cv_t<T>;

template<typename T, template<typename...> class Ref>
concept is_no_cv_specialization_v = is_specialization_v<no_cv_t<T>, Ref>;

template<typename T>
concept IsMutableSet = is_no_cv_specialization_v<T, MutableSet>;

template<typename T>
concept IsSortedVector = is_no_cv_specialization_v<T, SortedVector>;

template<typename T>
concept IsKOC = IsMutableSet<T> or IsSortedVector<T>;

} // namespace detail

template<typename T>
concept IsKeyedObjectContainer = detail::IsKOC<T>;

static_assert(IsKeyedObjectContainer<MutableSet<int>>);
static_assert(IsKeyedObjectContainer<SortedVector<int>>);

template<IsKeyedObjectContainer T>
struct llvm::yaml::SequenceTraits<T> {
  static size_t size(IO &io, T &Seq) { return Seq.size(); }

  class Inserter {
  private:
    using value_type = typename T::value_type;
    using KOT = KeyedObjectTraits<value_type>;
    using key_type = decltype(KOT::key(std::declval<value_type>()));

  private:
    T &Seq;
    decltype(Seq.begin()) It;
    bool IsOutputting;
    std::optional<typename T::BatchInserter> BatchInserter;
    value_type Instance;
    unsigned Index = 0;

  public:
    Inserter(IO &io, T &Seq) :
      Seq(Seq),
      It(Seq.begin()),
      IsOutputting(io.outputting()),
      Instance(KOT::fromKey(key_type())) {

      if constexpr (std::is_const_v<T>) {
        revng_assert(IsOutputting);
      } else {
        if (not IsOutputting)
          BatchInserter.emplace(std::move(Seq.batch_insert()));
      }
    }

    decltype(*It) &preflightElement(unsigned I) {
      revng_assert(Index == I);
      ++Index;

      if (IsOutputting)
        return *(It++);
      else
        return Instance;
    }

    void postflightElement(unsigned) {
      if (not IsOutputting) {
        BatchInserter->insert(Instance);
        Instance = KOT::fromKey(key_type());
      }
    };
  };
};

//
// Iterable
//
template<typename T>
concept Iterable = requires {
  // * begin/end
  // * operator!=
  std::begin(std::declval<T &>()) != std::end(std::declval<T &>());
  // * operator++
  ++std::declval<decltype(std::begin(std::declval<T &>())) &>();
  // * operator*
  void(*std::begin(std::declval<T &>()));
};

static_assert(Iterable<std::vector<int>>);
static_assert(not Iterable<int>);

//
// StringLike
//
template<typename T>
concept HasCStr = std::is_same_v<decltype(std::declval<T>().c_str()),
                                 const char *>;

static_assert(HasCStr<llvm::SmallString<4>>);
static_assert(not HasCStr<int>);

template<typename T>
concept StringLike = std::is_convertible_v<std::string, T> or HasCStr<T>;

static_assert(StringLike<llvm::SmallString<4>>);
static_assert(StringLike<std::string>);
static_assert(not StringLike<llvm::ArrayRef<int>>);
static_assert(StringLike<llvm::StringRef>);

//
// IsContainer
//

template<typename T>
concept IsContainer = Iterable<T> and not StringLike<T>;

// clang-format off
template<typename T>
concept NotTupleTreeCompatible = (not IsContainer<T>
                                  and not HasTupleSize<T>
                                  and not IsUpcastablePointer<T>);
// clang-format on

static_assert(IsContainer<std::vector<int>>);
static_assert(IsContainer<std::set<int>>);
static_assert(IsContainer<std::map<int, int>>);
static_assert(IsContainer<llvm::SmallVector<int, 4>>);
static_assert(!IsContainer<std::string>);
static_assert(!IsContainer<llvm::SmallString<4>>);
static_assert(!IsContainer<llvm::StringRef>);

//
// SortedContainer and UnsortedContainer
//
// TODO: this is not very nice
namespace detail {

template<typename T>
concept IsSet = is_specialization_v<T, std::set>;

} // namespace detail

template<typename T>
concept SortedContainer = detail::IsSet<T> or IsKeyedObjectContainer<T>;

template<typename T>
concept UnsortedContainer = IsContainer<T> and not SortedContainer<T>;
