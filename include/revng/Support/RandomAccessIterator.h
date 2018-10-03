#ifndef RAI_H
#define RAI_H

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <iterator>

// Local libraries includes
#include "revng/Support/Assert.h"

template<typename TypeT, typename Derived, bool Reference = true>
class RandomAccessIterator
  : public std::iterator<std::random_access_iterator_tag, TypeT> {

private:
  using iterator = typename std::iterator<std::random_access_iterator_tag,
                                          TypeT>;
  using type = RandomAccessIterator<TypeT, Derived, Reference>;
  using it_ref = typename iterator::reference;
  template<bool C, typename A, typename B>
  using conditional = std::conditional<C, A, B>;

public:
  using iterator_category = std::random_access_iterator_tag;
  using value_type = typename iterator::value_type;
  using difference_type = typename iterator::difference_type;
  using reference = typename conditional<Reference, it_ref, value_type>::type;
  using pointer = typename iterator::pointer;

private:
  const Derived &constThisDerived() const {
    return *static_cast<const Derived *>(this);
  }

  Derived &thisDerived() { return *static_cast<Derived *>(this); }

  reference get(unsigned Index) const { return constThisDerived().get(Index); }

  Derived clone(unsigned NewIndex) const {
    return Derived(constThisDerived(), NewIndex);
  }

  void assertCompatibility(const type &r) const {
    revng_assert(constThisDerived().isCompatible(r.constThisDerived()));
  }

protected:
  RandomAccessIterator() : Index(0) {}
  RandomAccessIterator(unsigned Index) : Index(Index) {}
  RandomAccessIterator(const type &r) : Index(r.Index) {}

  Derived &operator=(const type &r) {
    assertCompatibility(r);
    Index = r.Index;
    return thisDerived();
  }

public:
  Derived &operator++() {
    ++Index;
    return thisDerived();
  }

  Derived &operator--() {
    --Index;
    return thisDerived();
  }

  Derived operator++(int) { return clone(Index++); }

  Derived operator--(int) { return clone(Index--); }

  Derived operator+(const difference_type &n) const { return clone(Index + n); }

  Derived &operator+=(difference_type n) {
    Index += n;
    return thisDerived();
  }

  Derived operator-(const difference_type &n) const { return clone(Index - n); }

  Derived &operator-=(const difference_type &n) {
    Index -= n;
    return thisDerived();
  }

  reference operator*() const { return get(Index); }

  pointer operator->() const { return &get(Index); }

  reference operator[](const difference_type &n) const {
    return get(Index + n);
  }

  bool operator==(const type &r2) const {
    assertCompatibility(r2);
    return Index == r2.Index;
  }

  bool operator!=(const type &r2) {
    assertCompatibility(r2);
    return Index != r2.Index;
  }

  bool operator<(const type &r2) {
    assertCompatibility(r2);
    return Index < r2.Index;
  }

  bool operator>(const type &r2) {
    assertCompatibility(r2);
    return Index > r2.Index;
  }

  bool operator<=(const type &r2) {
    assertCompatibility(r2);
    return Index <= r2.Index;
  }

  bool operator>=(const type &r2) {
    assertCompatibility(r2);
    return Index >= r2.Index;
  }

  template<typename T, typename D, bool R>
  Derived operator+(const RandomAccessIterator<T, D, R> &r2) {
    assertCompatibility(r2);
    return clone(Index + r2.Index);
  }

  template<typename T, typename D, bool R>
  difference_type operator-(const RandomAccessIterator<T, D, R> &r2) const {
    assertCompatibility(r2);
    return Index - r2.Index;
  }

private:
  unsigned Index;
};

#endif // RAI_H
