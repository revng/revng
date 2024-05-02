#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <tuple>
#include <vector>

#include "revng/Support/MetaAddress.h"

class MetaAddressRangeSet {
private:
  // TODO: reimplement with non-linear lookup time
  std::vector<std::pair<MetaAddress, MetaAddress>> Ranges;

public:
  auto begin() const { return Ranges.begin(); }
  auto end() const { return Ranges.end(); }

public:
  bool contains(const MetaAddress &Address) const {
    revng_assert(Address.isValid());
    for (std::pair<MetaAddress, MetaAddress> Range : Ranges) {
      if (Range.first.addressLowerThanOrEqual(Address)
          and Address.addressLowerThan(Range.second)) {
        return true;
      }
    }
    return false;
  }

  bool contains(const MetaAddress &Start, const MetaAddress &End) const {
    revng_assert(Start.isValid() and End.isValid());

    for (const std::pair<MetaAddress, MetaAddress> &Range : Ranges) {
      if (Range.first.addressLowerThanOrEqual(Start)
          and Start.addressLowerThan(Range.second)
          and Range.first.addressLowerThanOrEqual(End)
          and End.addressLowerThan(Range.second)) {
        return true;
      }
    }

    return false;
  }

public:
  void add(const MetaAddress &Start, const MetaAddress &End) {
    Ranges.emplace_back(Start, End);
  }
};
