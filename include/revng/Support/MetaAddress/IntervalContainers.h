#pragma once

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

#include <cstdint>

#include "boost/icl/interval_map.hpp"

#include "revng/Support/MetaAddress.h"

struct IntervalMetaAddress : MetaAddress {
  IntervalMetaAddress() = default;
  IntervalMetaAddress(MetaAddress &Address) : MetaAddress(Address) {}

  uint64_t operator-(const IntervalMetaAddress &Other) const {
    std::optional<uint64_t> Diff = MetaAddress(*this) - MetaAddress(Other);
    revng_assert(Diff.has_value());
    return *Diff;
  }

  IntervalMetaAddress &operator++() {
    MetaAddress::operator+=(1u);
    return *this;
  }

  IntervalMetaAddress &operator--() {
    MetaAddress::operator-=(1u);
    return *this;
  }
};

namespace boost::icl {

template<>
struct is_discrete<IntervalMetaAddress> {
  using type = is_discrete;
  BOOST_STATIC_CONSTANT(bool, value = true);
};

template<>
struct identity_element<IntervalMetaAddress> {
  static IntervalMetaAddress value() { return IntervalMetaAddress{}; }
};

template<>
struct has_difference<IntervalMetaAddress> {
  using type = has_difference;
  BOOST_STATIC_CONSTANT(bool, value = true);
};

template<>
struct difference_type_of<IntervalMetaAddress> {
  using type = uint64_t;
};

template<>
struct size_type_of<IntervalMetaAddress> {
  using type = uint64_t;
};

} // namespace boost::icl
