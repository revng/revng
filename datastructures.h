#ifndef _DATASTRUCTURES_H
#define _DATASTRUCTURES_H

// Standard includes
#include <queue>
#include <set>
#include <stack>

/// \brief Queue where an element cannot be re-inserted if it's already in the
///        queue
template<typename T, bool Once>
class QueueImpl {
public:
  void insert(T Element) {
    if (Set.count(Element) == 0) {
      assert(Element->getParent() != nullptr);
      Set.insert(Element);
      Queue.push(Element);
    }
  }

  bool empty() const {
    return Queue.empty();
  }

  T pop() {
    T Result = Queue.front();
    Queue.pop();
    if (!Once)
      Set.erase(Result);
    return Result;
  }

  size_t size() const { return Queue.size(); }

  std::set<T> visited() {
    assert(Once);
    return std::move(Set);
  }

private:
  std::set<T> Set;
  std::queue<T> Queue;
};

template<typename T>
using UniquedQueue = QueueImpl<T, false>;

template<typename T>
using OnceQueue = QueueImpl<T, true>;

/// \brief Stack where an element cannot be re-inserted in it's already in the
///        stack
template<typename T>
class UniquedStack {
public:
  void insert(T Element) {
    if (Set.count(Element) == 0) {
      assert(Element->getParent() != nullptr);
      Set.insert(Element);
      Queue.push_back(Element);
    }
  }

  bool empty() const {
    return Queue.empty();
  }

  T pop() {
    T Result = Queue.back();
    Queue.pop_back();
    Set.erase(Result);
    return Result;
  }

  /// \brief Reverses the stack in its current status
  void reverse() {
    std::reverse(Queue.begin(), Queue.end());
  }

  size_t size() const { return Queue.size(); }

private:
  std::set<T> Set;
  std::vector<T> Queue;
};

#endif // _DATASTRUCTURES_H
