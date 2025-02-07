#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

template<typename T>
class ScopedExchange {
public:
  template<std::convertible_to<T> NewValueT>
  ScopedExchange(T &Object, NewValueT &&NewValue) :
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
