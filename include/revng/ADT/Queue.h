#ifndef QUEUE_H
#define QUEUE_H

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <queue>
#include <set>

// Local libraries includes
#include "revng/Support/Assert.h"

/// \brief Queue where an element cannot be re-inserted if it's already in the
///        queue
template<typename T, bool Once>
class QueueImpl {
public:
  void insert(T Element) {
    if (Set.count(Element) == 0) {
      Set.insert(Element);
      Queue.push(Element);
    }
  }

  bool empty() const { return Queue.empty(); }

  T head() const { return Queue.front(); }

  T pop() {
    T Result = head();
    Queue.pop();
    if (!Once)
      Set.erase(Result);
    return Result;
  }

  size_t size() const { return Queue.size(); }

  std::set<T> visited() {
    revng_assert(Once);
    return std::move(Set);
  }

  void clear() {
    std::set<T>().swap(Set);
    std::queue<T>().swap(Queue);
  }

private:
  std::set<T> Set;
  std::queue<T> Queue;
};

template<typename T>
using UniquedQueue = QueueImpl<T, false>;

template<typename T>
using OnceQueue = QueueImpl<T, true>;

#endif // QUEUE_H
