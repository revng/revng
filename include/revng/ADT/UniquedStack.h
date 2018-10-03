#ifndef UNIQUEDSTACK_H
#define UNIQUEDSTACK_H

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <algorithm>
#include <set>
#include <vector>

// Local libraries includes
#include "revng/Support/Assert.h"

/// \brief Stack where an element cannot be re-inserted in it's already in the
///        stack
template<typename T>
class UniquedStack {
public:
  void insert(T Element) {
    if (Set.count(Element) == 0) {
      revng_assert(Element->getParent() != nullptr);
      Set.insert(Element);
      Queue.push_back(Element);
    }
  }

  bool empty() const { return Queue.empty(); }

  T pop() {
    T Result = Queue.back();
    Queue.pop_back();
    Set.erase(Result);
    return Result;
  }

  /// \brief Reverses the stack in its current status
  void reverse() { std::reverse(Queue.begin(), Queue.end()); }

  size_t size() const { return Queue.size(); }

private:
  std::set<T> Set;
  std::vector<T> Queue;
};

#endif // UNIQUEDSTACK_H
