#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
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

inline RecursiveCoroutine<> my_coroutine(std::vector<MyState> &RCS, int x) {
  int X = 1;
  std::cerr << "Pre " << x << ':';
  RCS.back().Depth = x;

  for (const MyState &S : RCS) {
    std::cerr << ' ' << S.Depth;
  }
  std::cerr << std::endl;

  if (x < (rc_recur get10())) {
    RCS.emplace_back();
    rc_recur my_coroutine(RCS, x + 1);
  }

  std::cerr << "Post " << x << ' ' << X << std::endl;
  RCS.pop_back();
  rc_return;
}
