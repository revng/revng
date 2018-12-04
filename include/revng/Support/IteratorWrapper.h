#ifndef ITERATORWRAPPER_H
#define ITERATORWRAPPER_H

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <iterator>

template<typename W>
class IteratorWrapper
  : public std::iterator<typename std::iterator_traits<W>::iterator_category,
                         typename std::iterator_traits<W>::value_type> {

private:
  using type = IteratorWrapper<W>;

public:
  using iterator_category = typename std::iterator_traits<W>::iterator_category;
  using value_type = typename std::iterator_traits<W>::value_type;
  using difference_type = typename std::iterator_traits<W>::difference_type;
  using reference = typename std::iterator_traits<W>::reference;
  using pointer = typename std::iterator_traits<W>::pointer;

public:
  IteratorWrapper(W Iterator) : Iterator(Iterator) {}

  type &operator=(const type &r) {
    Iterator = r.Iterator;
    return *this;
  }

  type &operator++() {
    ++Iterator;
    return *this;
  }

  type &operator--() {
    --Iterator;
    return *this;
  }

  type operator++(int) { return type(Iterator++); }

  type operator--(int) { return type(Iterator--); }

  type operator+(const difference_type &n) const { return type(Iterator + n); }

  type &operator+=(difference_type n) {
    Iterator += n;
    return *this;
  }

  type operator-(const difference_type &n) const { return type(Iterator - n); }

  type &operator-=(const difference_type &n) {
    Iterator -= n;
    return *this;
  }

  reference operator*() const { return *Iterator; }

  pointer operator->() const { return Iterator.operator->(); }

  reference operator[](const difference_type &n) const { return Iterator[n]; }

  bool operator==(const type &r2) const { return Iterator == r2.Iterator; }

  bool operator!=(const type &r2) { return Iterator != r2.Iterator; }

  bool operator<(const type &r2) { return Iterator < r2.Iterator; }

  bool operator>(const type &r2) { return Iterator > r2.Iterator; }

  bool operator<=(const type &r2) { return Iterator <= r2.Iterator; }

  bool operator>=(const type &r2) { return Iterator >= r2.Iterator; }

  template<typename O>
  type operator+(const IteratorWrapper<O> &r2) {
    return type(Iterator + r2.Iterator);
  }

  template<typename O>
  difference_type operator-(const IteratorWrapper<O> &r2) {
    return Iterator - r2.Iterator;
  }

private:
  W Iterator;
};

#endif // ITERATORWRAPPER_H
