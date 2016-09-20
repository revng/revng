#ifndef _ITERATORWRAPPER_H
#define _ITERATORWRAPPER_H

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <iterator>

template<typename Wrapped>
class IteratorWrapper :
  public std::iterator<typename Wrapped::iterator_category,
                       typename Wrapped::value_type> {

private:
  using type = IteratorWrapper<Wrapped>;

public:
  using iterator_category = typename Wrapped::iterator_category;
  using value_type = typename Wrapped::value_type;
  using difference_type = typename Wrapped::difference_type;
  using reference = typename Wrapped::reference;
  using pointer = typename Wrapped::pointer;
public:
  IteratorWrapper(Wrapped Iterator) : Iterator(Iterator) { }

  type& operator=(const type& r) {
    Iterator = r.Iterator;
    return *this;
  }

  type& operator++() {
    ++Iterator;
    return *this;
  }

  type& operator--() {
    --Iterator;
    return *this;
  }

  type operator++(int) {
    return type(Iterator++);
  }

  type operator--(int) {
    return type(Iterator--);
  }

  type operator+(const difference_type& n) const {
    return type(Iterator + n);
  }

  type& operator+=(difference_type n) {
    Iterator += n;
    return *this;
  }

  type operator-(const difference_type& n) const {
    return type(Iterator - n);
  }

  type& operator-=(const difference_type& n) {
    Iterator -= n;
    return *this;
  }

  reference operator*() const {
    return *Iterator;
  }

  pointer operator->() const {
    return Iterator.operator->();
  }

  reference operator[](const difference_type& n) const {
    return Iterator[n];
  }

  bool operator==(const type& r2) const {
    return Iterator == r2.Iterator;
  }

  bool operator!=(const type& r2) {
    return Iterator != r2.Iterator;
  }

  bool operator<(const type& r2) {
    return Iterator < r2.Iterator;
  }

  bool operator>(const type& r2) {
    return Iterator > r2.Iterator;
  }

  bool operator<=(const type& r2) {
    return Iterator <= r2.Iterator;
  }

  bool operator>=(const type& r2) {
    return Iterator >= r2.Iterator;
  }

  template<typename W>
  type operator+(const IteratorWrapper<W>& r2) {
    return type(Iterator + r2.Iterator);
  }

  template<typename W>
  difference_type operator-(const IteratorWrapper<W>& r2) {
    return Iterator - r2.Iterator;
  }

private:
  Wrapped Iterator;
};

#endif // _ITERATORWRAPPER_H
