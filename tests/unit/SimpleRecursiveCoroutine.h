#pragma once

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

#include <iostream>
#include <vector>

#include "revng/ADT/RecursiveCoroutine.h"

struct MyState {
  int Depth;
};

inline RecursiveCoroutine<int> get10() {
  rc_return 10;
}

inline RecursiveCoroutine<> myCoroutine(std::vector<MyState> &RCS, int Y) {
  int X = 1;
  std::cerr << "Pre " << Y << ':';
  RCS.back().Depth = Y;

  for (const MyState &S : RCS) {
    std::cerr << ' ' << S.Depth;
  }
  std::cerr << std::endl;

  if (Y < (rc_recur get10())) {
    RCS.emplace_back();
    rc_recur myCoroutine(RCS, Y + 1);
  }

  std::cerr << "Post " << Y << ' ' << X << std::endl;
  RCS.pop_back();
  rc_return;
}

inline RecursiveCoroutine<void> accumulateSums(int I, int &Result) {
  Result += I;
  if (I)
    rc_recur accumulateSums(I - 1, Result);
}
