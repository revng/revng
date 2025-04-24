#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <concepts>

#include "revng/ADT/ConstexprString.h"
#include "revng/ADT/STLExtras.h"
#include "revng/Support/DynamicHierarchy.h"
#include "revng/Support/YAMLTraits.h"

namespace pipeline {

/// The rank tree is a tree used by targets to find out how many names
/// are required to name a target
class Rank : public DynamicHierarchy<Rank> {
public:
  Rank(llvm::StringRef Name) : DynamicHierarchy(Name) {}
  Rank(llvm::StringRef Name, Rank &Parent) : DynamicHierarchy(Name, Parent) {}
};

// Root rank specialization
template<ConstexprString String>
class RootRank : public Rank {
public:
  static constexpr bool RankTag = true;
  static constexpr llvm::StringRef RankName = String;
  using Type = void;
  using Parent = void;

public:
  static constexpr size_t Depth = 0;
  using Tuple = std::tuple<>;

public:
  explicit RootRank() : Rank(RankName) {}
};

/// A helper function used for defining a root rank.
///
/// Root rank doesn't have corresponding storage location and is only
/// used to defining a single logical starting point in the rank hierarhy.
template<ConstexprString Name>
pipeline::RootRank<Name> defineRootRank() {
  return pipeline::RootRank<Name>();
}

template<typename RankType>
concept RankSpecialization = requires(RankType &&Rank) {
  RankType::RankTag;

  { RankType::RankName } -> std::convertible_to<llvm::StringRef>;
  { RankType::Depth } -> std::convertible_to<size_t>;

  typename RankType::Type;
  typename RankType::Parent;
  typename RankType::Tuple;
};

namespace detail {

template<typename... TypesToAppend, typename... InitialTupleTypes>
inline constexpr std::tuple<InitialTupleTypes..., TypesToAppend...>
appendImpl(std::tuple<InitialTupleTypes...> Tuple);

/// A helper class used to produce a new tuple type which is an extension
/// of the passed types with additional elements added at the end.
template<typename Tuple, typename... TypesToAppend>
struct AppendToTupleHelper {
  using type = decltype(appendImpl<TypesToAppend...>(std::declval<Tuple>()));
};

} // namespace detail

template<ConstexprString String,
         HasScalarOrEnumTraits Key,
         RankSpecialization ParentRank>
class TypedRank : public Rank {
public:
  static constexpr bool RankTag = true;
  static constexpr llvm::StringRef RankName = String;
  using Type = Key;
  using Parent = ParentRank;

public:
  static_assert(Parent::RankTag == true);
  static constexpr size_t Depth = Parent::Depth + 1;
  using Tuple = typename detail::AppendToTupleHelper<typename Parent::Tuple,
                                                     Type>::type;

public:
  explicit TypedRank(Parent &ParentObj) : Rank(RankName, ParentObj) {}
};

/// A helper function for defining a new rank.
///
/// It accepts two template parameters and one normal argument:
/// \tparam Name is a name for the new rank.
/// \tparam Type is the type for the tuple representing the location of an.
/// object with this rank or rank depending on this one.
/// \arg ParentObject is the rank this rank extends.
template<ConstexprString Name,
         HasScalarOrEnumTraits Type,
         RankSpecialization Parent>
pipeline::TypedRank<Name, Type, Parent> defineRank(Parent &ParentObject) {
  return pipeline::TypedRank<Name, Type, Parent>(ParentObject);
}

namespace detail {

/// A helper variable that compares the depth of a rank against an expected
/// depth while also correctly handling the corner case of the rank being a root
/// of the rank tree.
template<typename RankOrVoid, size_t ExpectedDepth>
inline constexpr bool DepthCheck = false;

template<RankSpecialization Rank, size_t Expected>
constexpr bool DepthCheck<Rank, Expected> = Rank::Depth + 1 == Expected;

template<size_t ExpectedDepth>
inline constexpr bool DepthCheck<void, ExpectedDepth> = (ExpectedDepth == 0);

/// A helper struct that incapsulates reachability logic for connected ranks.
/// the `value` member is set to `true` if and only if the `To` rank can be
/// reached from `From` rank by consecutive `From = From::Parent` operations.
template<typename From, typename To>
struct ReachabilityHelper {
private:
  /// Make sure both parameters are in fact ranks.
  static_assert(RankSpecialization<From> && RankSpecialization<To>);

  /// Mark locations as reachable, if they are the same.
  static constexpr bool Found = std::is_same_v<From, To>;

  /// Make sure the `From` rank tree is not broken (no ranks are skipped).
  static constexpr size_t Depth = From::Depth;
  static_assert(DepthCheck<typename From::Parent, Depth>);

  /// Decide whether the next comparison should be performed.
  /// It's not needed if we already passed the desired depth and/or if we
  /// reached depth 0 (the root) as depth can never increase in a valid tree.
  using Next = std::conditional_t<
    (Depth <= To::Depth || Depth == 0),
    std::false_type,
    ReachabilityHelper<typename From::Parent, To>>;

public:
  static constexpr bool value = Found || Next::value;
};

} // namespace detail

/// A helper concept used for checking whether two ranks have common "roots"
/// Is evaluated to `true` if and only if the \tparam To rank can be reached
/// from the \tparam From rank by going up the tree.
template<typename To, typename From>
concept RankConvertibleTo = RankSpecialization<From> && RankSpecialization<To>
                            && detail::ReachabilityHelper<From, To>::value;

} // namespace pipeline
