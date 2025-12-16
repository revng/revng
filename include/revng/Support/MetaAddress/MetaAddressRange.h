#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <tuple>
#include <vector>

#include "revng/Support/MetaAddress.h"

class MetaAddressRange : public std::pair<MetaAddress, MetaAddress> {
private:
  using Base = std::pair<MetaAddress, MetaAddress>;

public:
  MetaAddressRange(const MetaAddress &Start, const MetaAddress &End) :
    Base(Start, End) {
    revng_assert(Start.isValid());
    revng_assert(End.isValid());
    revng_assert(Start <= End);
  }

public:
  const MetaAddress &start() const { return first; }
  const MetaAddress &end() const { return second; }
  size_t size() const { return (end() - start()).value(); }

public:
  bool contains(const MetaAddress &Address) const {
    revng_assert(Address.isValid());
    return (start().addressLowerThanOrEqual(Address)
            and Address.addressLowerThan(end()));
  }

  bool contains(const MetaAddress &OtherStart,
                const MetaAddress &OtherEnd) const {
    revng_assert(OtherStart.isValid() and OtherEnd.isValid());
    return (start().addressLowerThanOrEqual(OtherStart)
            and OtherStart.addressLowerThanOrEqual(end())
            and start().addressLowerThanOrEqual(OtherEnd)
            and OtherEnd.addressLowerThanOrEqual(end()));
  }

  bool contains(const MetaAddressRange &Other) const {
    return contains(Other.start(), Other.end());
  }

  bool overlaps(const MetaAddressRange &Other) const {
    return contains(Other.start()) or Other.contains(start());
  }

  bool overlaps(const MetaAddress &OtherStart,
                const MetaAddress &OtherEnd) const {
    return overlaps(MetaAddressRange(OtherStart, OtherEnd));
  }
};

class MetaAddressRangeSet {
private:
  // TODO: reimplement with non-linear lookup time
  std::vector<MetaAddressRange> Ranges;

public:
  auto begin() const { return Ranges.begin(); }
  auto end() const { return Ranges.end(); }

public:
  bool contains(const MetaAddress &Address) const {
    revng_assert(Address.isValid());
    for (const MetaAddressRange &Range : Ranges)
      if (Range.contains(Address))
        return true;
    return false;
  }

  bool contains(const MetaAddress &Start, const MetaAddress &End) const {
    revng_assert(Start.isValid() and End.isValid());
    for (const MetaAddressRange &Range : Ranges)
      if (Range.contains(Start, End))
        return true;
    return false;
  }

  bool overlaps(const MetaAddress &Start, const MetaAddress &End) const {
    revng_assert(Start.isValid() and End.isValid());
    for (const MetaAddressRange &Range : Ranges)
      if (Range.overlaps(Start, End))
        return true;
    return false;
  }

  bool overlaps(const MetaAddressRange &Other) const {
    return overlaps(Other.start(), Other.end());
  }

public:
  void add(const MetaAddress &Start, const MetaAddress &End) {
    Ranges.emplace_back(Start, End);
  }
};
