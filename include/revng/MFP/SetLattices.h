#pragma once

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

#include <algorithm>

template<typename LE>
struct SetUnionLattice {
  using LatticeElement = LE;

  static LatticeElement combineValues(const LatticeElement &Left,
                                      const LatticeElement &Right) {
    LatticeElement Result;
    std::set_union(Left.begin(),
                   Left.end(),
                   Right.begin(),
                   Right.end(),
                   std::inserter(Result, Result.begin()));
    return Result;
  }

  static bool isLessOrEqual(const LatticeElement &Left,
                            const LatticeElement &Right) {
    if (Left.size() > Right.size())
      return false;
    return std::includes(Right.begin(), Right.end(), Left.begin(), Left.end());
  }
};

template<typename LE>
struct SetIntersectionLattice {
  using LatticeElement = LE;

  static LatticeElement combineValues(const LatticeElement &Left,
                                      const LatticeElement &Right) {
    LatticeElement Result;
    std::set_intersection(Left.begin(),
                          Left.end(),
                          Right.begin(),
                          Right.end(),
                          std::inserter(Result, Result.begin()));
    return Result;
  }

  static bool isLessOrEqual(const LatticeElement &Left,
                            const LatticeElement &Right) {
    if (Right.size() > Left.size())
      return false;
    return std::includes(Left.begin(), Left.end(), Right.begin(), Right.end());
  }
};
