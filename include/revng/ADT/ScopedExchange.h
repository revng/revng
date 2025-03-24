#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <concepts>
#include <utility>

/// Temporarily assigns a new value over the specified object for the duration
/// of the lifetime of the ScopedExchange object. The old value is first moved
/// out and upon destruction is once again move-assigned over the object.
template<typename T>
class ScopedExchange {
public:
  template<std::convertible_to<T> NewValueT>
  explicit ScopedExchange(T &Object, NewValueT &&NewValue) :
    Object(Object), OldValue(std::move(Object)) {
    Object = std::forward<NewValueT>(NewValue);
  }

  ScopedExchange(const ScopedExchange &) = delete;
  ScopedExchange &operator=(const ScopedExchange &) = delete;

  ~ScopedExchange() { Object = std::move(OldValue); }

private:
  T &Object;
  T OldValue;
};
