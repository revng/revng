#pragma once

// clang-format language: cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <array>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/YAMLTraits.h"

#include "revng/ADT/STLExtras.h"
#include "revng/Pipeline/Rank.h"
#include "revng/Support/MetaAddress.h"
#include "revng/Support/YAMLTraits.h"

namespace pipeline {

/// A location represents a position of an entity in the ranked artifact system.
///
/// Each location has a rank associated with it, it's set using
/// \tparam Rank.
///
/// In serialized form, a location is the name of its rank plus a list of
/// the values of its keys (one key per rank depth) separated by '/'. The types
/// are determined by the parents of the rank (see `Rank::Parent`).
///
/// In deserialized form, the name gets lifted to a compilation time and can be
/// accessed at `Rank::RankName`. The tuple of the keys is publically inherited
/// from, so it can either be accessed as normal tuple (for example, using
/// `std::get<std::size_t>` after casting. There's a cast helper members
/// (`tuple()`)) or using special accessors like `at()`.
template<RankSpecialization Rank>
class Location : public Rank::Tuple {
private:
  using Tuple = typename Rank::Tuple;
  static_assert(std::tuple_size_v<Tuple> == Rank::Depth);
  static constexpr std::size_t Size = Rank::Depth;

private:
  constexpr static std::string_view Separator = "/";

public:
  using Tuple::Tuple;
  Location(const Location &) = default;
  Location(Location &&) = default;
  Location &operator=(const Location &) = default;
  Location &operator=(Location &&) = default;

  Tuple &tuple() { return *this; }
  const Tuple &tuple() const { return *this; }

  template<RankConvertibleTo<Rank> AnotherRank>
  auto &at(const AnotherRank &) {
    return std::get<AnotherRank::Depth - 1>(tuple());
  }
  template<RankConvertibleTo<Rank> AnotherRank>
  const auto &at(const AnotherRank &) const {
    return std::get<AnotherRank::Depth - 1>(tuple());
  }

  // clang-format off

  /// A static helper function for constructing locations from different
  /// location with related ranks.
  ///
  /// If output location's rank is higher, all the remaining keys are left
  /// uninitialized (or default constructed, depending on the type).
  ///
  /// If output location's rank is lower, all the extra keys get discarded.
  template<typename AnotherRank>
    requires (RankConvertibleTo<Rank, AnotherRank>
              || RankConvertibleTo<AnotherRank, Rank>)
  static constexpr Location<Rank>
  convert(const Location<AnotherRank> &Another) {
    // clang-format on
    Location<Rank> Result;

    constexpr auto Common = std::min(Rank::Depth, AnotherRank::Depth);
    constexprRepeat<Common>([&Result, &Another]<std::size_t I> {
      std::get<I>(Result.tuple()) = std::get<I>(Another.tuple());
    });

    return Result;
  }

public:
  /// Serializes a location into a string.
  ///
  /// TODO: look into the constexpr implementation once constexpr strings are
  /// implemented (clang-15+)
  std::string toString() const {
    std::string Result;

    Result += Separator;
    Result += Rank::RankName;
    constexprRepeat<Size>([&Result, this]<std::size_t Index> {
      Result += Separator;
      Result += serializeToString(std::get<Index>(tuple()));
    });

    return Result;
  }

  /// Deserializes the location from a string.
  ///
  /// If the string is not a valid location OR if its rank is different
  /// from this location's rank, `std::nullopt` is returned instead.
  static constexpr std::optional<Location<Rank>>
  fromString(std::string_view String) {
    Location<Rank> Result;

    auto MaybeSteps = constexprSplit<Size + 2>(Separator, String);
    if (!MaybeSteps.has_value())
      return std::nullopt;

    revng_assert(MaybeSteps.value().size() == Size + 2);
    constexpr std::string_view ExpectedName = Rank::RankName;
    if (MaybeSteps->at(0) != "" || MaybeSteps->at(1) != ExpectedName)
      return std::nullopt;

    auto Success = constexprAnd<Size>([&Result, &MaybeSteps]<std::size_t Idx> {
      using T = typename std::tuple_element<Idx, Tuple>::type;
      using revng::detail::deserializeImpl;
      auto MaybeValue = deserializeImpl<T>(MaybeSteps->at(Idx + 2));
      if (!MaybeValue)
        return false;

      std::get<Idx>(Result) = std::move(*MaybeValue);
      return true;
    });
    if (Success == false)
      return std::nullopt;

    return Result;
  }
};

// clang-format off

/// Constructs a new location from arbitrary arguments.
///
/// The first argument is used to indicate the rank of location to be
/// constructed, the rest are the arguments.
template<RankSpecialization Rank, typename ...Args>
  requires std::is_convertible_v<std::tuple<Args...>, typename Rank::Tuple>
inline constexpr Location<Rank> location(const Rank &, Args &&...As) {
  return Location<Rank>(As...);
}

/// Constructs a new location from arbitrary arguments and instantly serializes
/// it into its string representation.
template<RankSpecialization Rank, typename ...Args>
  requires std::is_convertible_v<std::tuple<Args...>, typename Rank::Tuple>
inline std::string serializedLocation(const Rank &R, Args &&...As) {
  return location(R, std::forward<Args>(As)...).toString();
}

/// A helper interface for location deserialization without the need to
/// explicitly mention the expected rank's type.
///
/// It takes a reference to the corresponding rank object as its first argument.
template<RankSpecialization Rank>
inline constexpr std::optional<Location<Rank>>
locationFromString(const Rank &, std::string_view String) {
  return Location<Rank>::fromString(String);
}

/// A helper interface for location conversion.
///
/// It discloses the static `convert` member in an easier-to-access fashion.
template<typename ResultRank, typename InputRank>
  requires(RankConvertibleTo<ResultRank, InputRank>)
inline constexpr Location<ResultRank>
convertLocation(const ResultRank &Result, const Location<InputRank> &Input) {
  return Location<ResultRank>::convert(Input);
}

// clang-format on

namespace detail {

/// Shorthand to "decay type then make a const pointer to it".
template<typename T>
using ConstP = std::add_pointer_t<std::add_const_t<std::decay_t<T>>>;

} // namespace detail

// clang-format off

/// A helper function used for deserializing any number of differently
/// ranked locations at once.
///
/// \arg Serialized is the string containing the serialized location.
/// \arg Expected indicates the expected rank to be returned.
/// \arg Supported lists all the other ranks that are supported, they must all
/// be convertible to the \arg Expected rank.
///
/// \returns a valid location of \arg Expected rank if the \arg Serialized
/// string contains a valid serialized form of any location type within the
/// \arg Supported list (including \arg Expected), `std::nullopt` otherwise.
template <typename ExpectedRank, typename ...SupportedRanks>
  requires (RankConvertibleTo<ExpectedRank, SupportedRanks> && ...)
inline constexpr std::optional<Location<ExpectedRank>>
genericLocationFromString(std::string_view Serialized,
                          const ExpectedRank &Expected,
                          const SupportedRanks &...Supported) {
  // clang-format on
  Location<ExpectedRank> Result;
  bool ParsedOnce = false;

  using TupleType = std::tuple<detail::ConstP<ExpectedRank>,
                               detail::ConstP<SupportedRanks>...>;
  TupleType Tuple{ &Expected, &Supported... };
  constexprRepeat<std::tuple_size_v<TupleType>>([&, Serialized]<std::size_t I> {
    auto MaybeLoc = locationFromString(*std::get<I>(Tuple), Serialized);
    if (MaybeLoc.has_value()) {
      Result = Location<ExpectedRank>::convert(*MaybeLoc);

      revng_assert(ParsedOnce == false,
                   "Duplicate supported ranks are not allowed");
      ParsedOnce = true;
    }
  });

  if (ParsedOnce)
    return Result;
  else
    return std::nullopt;
}

// clang-format off

/// The simplified interface for generic location deserialization allowing for
/// fetching a single key, indicated by \tparam Idx.
template <std::size_t Idx, typename ExpectedRank, typename ...SupportedRanks>
  requires (RankConvertibleTo<ExpectedRank, SupportedRanks> && ...)
constexpr std::optional<std::tuple_element_t<Idx, typename ExpectedRank::Tuple>>
genericLocationFromString(std::string_view Serialized,
                          const ExpectedRank &Expected,
                          const SupportedRanks &...Supported) {
  // clang-format on
  static_assert(Idx < std::decay_t<ExpectedRank>::Depth);
  auto Result = genericLocationFromString(Serialized, Expected, Supported...);
  if (Result.has_value())
    return std::get<Idx>(Result->tuple());
  else
    return std::nullopt;
}

} // namespace pipeline

template<pipeline::RankSpecialization Rank>
struct llvm::yaml::ScalarTraits<pipeline::Location<Rank>> {
  static void output(const pipeline::Location<Rank> &Value,
                     void *,
                     llvm::raw_ostream &Output) {
    Output << Value.toString();
  }

  static StringRef
  input(llvm::StringRef Scalar, void *, pipeline::Location<Rank> &Value) {
    auto MaybeValue = pipeline::Location<Rank>::fromString(Scalar);
    revng_assert(MaybeValue.has_value());
    Value = std::move(*MaybeValue);
    return StringRef();
  }

  static QuotingType mustQuote(StringRef) { return QuotingType::Double; }
};
