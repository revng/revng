#ifndef TRANSFORMADAPTER_H
#define TRANSFORMADAPTER_H

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <functional>

// Local libraries includes
#include "revng/Support/IteratorWrapper.h"
#include "revng/Support/Range.h"

template<typename NewType, typename Wrapped>
class TransformIterator : public IteratorWrapper<Wrapped> {
private:
  using base = IteratorWrapper<Wrapped>;

public:
  using value_type = NewType;
  using reference = NewType;
  using pointer = typename base::pointer;
  using difference_type = typename base::difference_type;
  using transformer = std::function<NewType(typename base::reference)>;

public:
  TransformIterator(Wrapped Iterator, transformer F) :
    IteratorWrapper<Wrapped>(Iterator),
    F(F) {}

  reference operator*() const { return F(base::operator*()); }

  pointer operator->() const = delete;

  reference operator[](const difference_type &n) const {
    return F(base::operator[](n));
  }

private:
  transformer F;
};

namespace adaptors {

template<typename R, typename Iterator>
class Transform {
public:
  using transformer = typename TransformIterator<R, Iterator>::transformer;
  Transform(transformer Transformer) : Transformer(Transformer) {}

  template<typename I>
  Range<TransformIterator<R, I>> transform(Range<I> Input) {
    return Range<TransformIterator<R, I>>(TransformIterator<R, I>(Input.begin(),
                                                                  Transformer),
                                          TransformIterator<R, I>(Input.end(),
                                                                  Transformer));
  }

private:
  transformer Transformer;
};

} // namespace adaptors

template<typename T>
struct function_traits : public function_traits<decltype(&T::operator())> {};

template<typename ClassType, typename ReturnType, typename... Args>
struct function_traits<ReturnType (ClassType::*)(Args...) const> {
  using return_type = ReturnType;
};

template<typename R, typename I>
using TransformFunction = std::function<R(typename I::reference)>;

template<typename R, typename I>
auto operator|(Range<I> Input, R Transformer)
  -> Range<TransformIterator<typename function_traits<R>::return_type, I>> {
  using T = adaptors::Transform<typename function_traits<R>::return_type, I>;
  return T(Transformer).transform(Input);
}

template<typename R, typename C>
auto operator|(C Input, R Transformer)
  -> Range<TransformIterator<typename function_traits<R>::return_type,
                             typename C::iterator>> {
  using T = adaptors::Transform<typename function_traits<R>::return_type,
                                typename C::iterator>;
  return T(Transformer).transform(make_range(Input));
}

// template<typename C>
// Range<TransformIterator<typename C::iterator>>
// operator|(C Input, TransformFunction<typename C::iterator> Function) {

//   adaptors::Transform<typename C::iterator> Transformer(Function);
//   return Transformer.transform(make_range(Input));
// }

#endif // TRANSFORMADAPTER_H
