#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <concepts>
#include <optional>
#include <set>

#include "llvm/Support/YAMLTraits.h"

#include "revng/ADT/STLExtras.h"
#include "revng/ADT/UpcastablePointer.h"
#include "revng/Support/Assert.h"

template<typename T>
struct KeyedObjectTraits;

template<typename T, typename Traits = KeyedObjectTraits<T>>
concept KeyedObjectContainerCompatible = requires(T A) {
  { Traits::key(A) };
  { Traits::fromKey(Traits::key(A)) } -> std::same_as<T>;
} && std::is_same_v<Traits, KeyedObjectTraits<T>>;

/// Inherit if T is the key of itself
template<typename T>
struct IdentityKeyedObjectTraits {
  static T key(const T &Obj) { return Obj; }

  static T fromKey(T Obj) { return Obj; }
};

/// Trivial specializations
template<std::integral T>
struct KeyedObjectTraits<T> : public IdentityKeyedObjectTraits<T> {};

template<>
struct KeyedObjectTraits<std::string>
  : public IdentityKeyedObjectTraits<std::string> {};

static_assert(KeyedObjectContainerCompatible<int>);

template<typename T>
concept KeyedObjectContainer = requires(T &&) { T::KeyedObjectContainerTag; };

namespace revng::detail {

template<KeyedObjectContainerCompatible T>
using KOT = KeyedObjectTraits<T>;

template<KeyedObjectContainerCompatible T>
using Key = std::decay_t<decltype(KOT<T>::key(std::declval<T>()))>;

} // namespace revng::detail

template<KeyedObjectContainerCompatible T>
using DefaultKeyObjectComparator = std::less<const revng::detail::Key<T>>;

template<KeyedObjectContainer T>
struct llvm::yaml::SequenceTraits<T> {
  static size_t size(IO &TheIO, T &Seq) { return Seq.size(); }

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
    Inserter(IO &TheIO, T &Seq) :
      Seq(Seq),
      It(Seq.begin()),
      IsOutputting(TheIO.outputting()),
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
