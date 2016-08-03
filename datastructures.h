#ifndef _DATASTRUCTURES_H
#define _DATASTRUCTURES_H

// Standard includes
#include <queue>
#include <set>

template<typename T>
class UniquedQueue {
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
    Set.erase(Result);
    return Result;
  }

  size_t size() const { return Queue.size(); }
private:
  std::set<T> Set;
  std::queue<T> Queue;
};

#endif // _DATASTRUCTURES_H
