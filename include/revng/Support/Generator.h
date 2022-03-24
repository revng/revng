#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// This files has been from the cppcoro project by Lewis Baker, which is
// licensed under MIT license.

#include <exception>
#include <experimental/coroutine>
#include <functional>
#include <iterator>
#include <type_traits>
#include <utility>

namespace cppcoro {
template<typename T>
class generator;

namespace detail {
template<typename T>
class GeneratorPromise {
public:
  using value_type = std::remove_reference_t<T>;
  using reference_type = std::conditional_t<std::is_reference_v<T>, T, T &>;
  using pointer_type = value_type *;

  GeneratorPromise() = default;

  generator<T> get_return_object() noexcept;

  constexpr std::experimental::suspend_always initial_suspend() const noexcept {
    return {};
  }
  constexpr std::experimental::suspend_always final_suspend() const noexcept {
    return {};
  }

  template<typename U = T,
           std::enable_if_t<!std::is_rvalue_reference<U>::value, int> = 0>
  std::experimental::suspend_always
  yield_value(std::remove_reference_t<T> &ValueToYield) noexcept {
    Value = std::addressof(ValueToYield);
    return {};
  }

  std::experimental::suspend_always
  yield_value(std::remove_reference_t<T> &&ValueToYieldValue) noexcept {
    Value = std::addressof(ValueToYieldValue);
    return {};
  }

  void unhandled_exception() { Exception = std::current_exception(); }

  void return_void() {}

  reference_type value() const noexcept {
    return static_cast<reference_type>(*Value);
  }

  // Don't allow any use of 'co_await' inside the generator coroutine.
  template<typename U>
  std::experimental::suspend_never await_transform(U &&) = delete;

  void rethrow_if_exception() {
    if (Exception) {
      std::rethrow_exception(Exception);
    }
  }

private:
  pointer_type Value;
  std::exception_ptr Exception;
};

struct GeneratorSentinel {};

template<typename T>
class GeneratorIterator {
  using promise = GeneratorPromise<T>;
  using coroutine_handle = std::experimental::coroutine_handle<promise>;

public:
  using iterator_category = std::input_iterator_tag;
  // What type should we use for counting elements of a potentially infinite
  // sequence?
  using difference_type = std::ptrdiff_t;
  using value_type = typename GeneratorPromise<T>::value_type;
  using reference = typename GeneratorPromise<T>::reference_type;
  using pointer = typename GeneratorPromise<T>::pointer_type;

  // Iterator needs to be default-constructible to satisfy the Range concept.
  GeneratorIterator() noexcept : Coroutine(nullptr) {}

  explicit GeneratorIterator(coroutine_handle Coroutine) noexcept :
    Coroutine(Coroutine) {}

  friend bool
  operator==(const GeneratorIterator &It, GeneratorSentinel) noexcept {
    return !It.Coroutine || It.Coroutine.done();
  }

  friend bool
  operator!=(const GeneratorIterator &It, GeneratorSentinel S) noexcept {
    return !(It == S);
  }

  friend bool
  operator==(GeneratorSentinel S, const GeneratorIterator &It) noexcept {
    return (It == S);
  }

  friend bool
  operator!=(GeneratorSentinel S, const GeneratorIterator &It) noexcept {
    return It != S;
  }

  GeneratorIterator &operator++() {
    Coroutine.resume();
    if (Coroutine.done()) {
      Coroutine.promise().rethrow_if_exception();
    }

    return *this;
  }

  // Need to provide post-increment operator to implement the 'Range' concept.
  void operator++(int) { (void) operator++(); }

  reference operator*() const noexcept { return Coroutine.promise().value(); }

  pointer operator->() const noexcept { return std::addressof(operator*()); }

private:
  coroutine_handle Coroutine;
};
} // namespace detail

template<typename T>
class [[nodiscard]] generator {
public:
  using promise_type = detail::GeneratorPromise<T>;
  using iterator = detail::GeneratorIterator<T>;

  generator() noexcept : Coroutine(nullptr) {}

  generator(generator &&Other) noexcept : Coroutine(Other.Coroutine) {
    Other.Coroutine = nullptr;
  }

  generator(const generator &Other) = delete;
  generator &operator=(const generator &Other) = delete;

  ~generator() {
    if (Coroutine) {
      revng_assert(Coroutine.done());
      Coroutine.destroy();
    }
  }

  generator &operator=(generator &&Other) noexcept {
    swap(Other);
    return *this;
  }

  iterator begin() {
    if (Coroutine) {
      Coroutine.resume();
      if (Coroutine.done()) {
        Coroutine.promise().rethrow_if_exception();
      }
    }

    return iterator{ Coroutine };
  }

  detail::GeneratorSentinel end() noexcept {
    return detail::GeneratorSentinel{};
  }

  void swap(generator &Other) noexcept {
    std::swap(Coroutine, Other.Coroutine);
  }

private:
  friend class detail::GeneratorPromise<T>;

  using coroutine_handle = std::experimental::coroutine_handle<promise_type>;

  explicit generator(coroutine_handle Coroutine) noexcept :
    Coroutine(Coroutine) {}

  std::experimental::coroutine_handle<promise_type> Coroutine;
};

template<typename T>
void swap(generator<T> &A, generator<T> &B) {
  A.swap(B);
}

namespace detail {
template<typename T>
generator<T> GeneratorPromise<T>::get_return_object() noexcept {
  using promise = GeneratorPromise<T>;
  using coroutine_handle = std::experimental::coroutine_handle<promise>;
  return generator<T>{ coroutine_handle::from_promise(*this) };
}
} // namespace detail

} // namespace cppcoro

// NOLINTEND
