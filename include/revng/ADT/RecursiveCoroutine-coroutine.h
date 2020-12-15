#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <experimental/coroutine>
#include <optional>

#include "revng/Support/Assert.h"

struct PromiseBase {

  auto initial_suspend() const { return std::experimental::suspend_always(); }
  auto final_suspend() const noexcept {
    return std::experimental::suspend_always();
  }

  [[noreturn]] void unhandled_exception() const { std::terminate(); }

  PromiseBase *Callee = nullptr;
};

template<typename ReturnT = void>
struct [[nodiscard("RecursiveCoroutine is discarded without running "
                   "it")]] RecursiveCoroutine {
public:
  struct promise_type;

public:
  using coro_handle = std::experimental::coroutine_handle<promise_type>;

public:
  RecursiveCoroutine(coro_handle H) : OwnedHandle(H) {}
  RecursiveCoroutine() : RecursiveCoroutine(coro_handle{}) {}

  // Not copyable, because we don't want OwnedHandle to be destroyed twice
  RecursiveCoroutine &operator=(const RecursiveCoroutine &);
  RecursiveCoroutine(const RecursiveCoroutine &);

  // Movable, but we clean up the OwnedHandle, because we don't want it to be
  // destroyed twice in the destructor
  RecursiveCoroutine &operator=(RecursiveCoroutine &&Other) {
    this->OwnedHandle = Other.OwnedHandle;
    Other.OwnedHandle = {};
    return *this;
  }
  RecursiveCoroutine(RecursiveCoroutine && Other) { *this = std::move(Other); }

  ~RecursiveCoroutine() {
    revng_assert(OwnedHandle and OwnedHandle.done());
    OwnedHandle.destroy();
    OwnedHandle = {};
  }

  bool await_ready() const { return false; }

  // TODO: I wanted to do something like the following
  //
  // template<typename CallerReturnT>
  // void await_suspend(std::coroutine_handle<RecursiveCoroutine<CallerReturnT>>
  //                    CallerHandle)
  //
  // but I had to fight the template parameter type inference engine and I gave
  // up. This is also likely a good spot to constraint CallerCoroHandleT with
  // C++20 concepts, but not today.
  template<typename CallerCoroHandleT>
  void await_suspend(CallerCoroHandleT CallerHandle) {
    revng_assert(CallerHandle and not CallerHandle.done());
    revng_assert(OwnedHandle and not OwnedHandle.done());
    CallerHandle.promise().Callee = &OwnedHandle.promise();
  }

  ReturnT await_resume() {
    revng_assert(OwnedHandle and OwnedHandle.done());
    if constexpr (std::is_same_v<ReturnT, void>) {
      return;
    } else {
      return OwnedHandle.promise().get();
    }
  }

  ReturnT run() {
    revng_assert(OwnedHandle);

    // If the result is already available, use it!
    if (OwnedHandle.done())
      return OwnedHandle.promise().get();

    using recursive_handle = std::experimental::coroutine_handle<PromiseBase>;
    std::vector<PromiseBase *> Stack;

    Stack.push_back(&OwnedHandle.promise());

    while (not Stack.empty()) {
      PromiseBase &CurrentPromise = *Stack.back();
      auto CurrentHandle = recursive_handle::from_promise(CurrentPromise);
      revng_assert(CurrentHandle);

      // Resume the coroutine that's on top of the Stack.
      // This will either suspend at the final suspension point of the current
      // coroutine on top of the stack, or after setting a handle to a new
      // recursive coroutine ready for execution in the promise object of the
      // current coroutine on top of the stack.
      CurrentHandle.resume();

      if (CurrentHandle.done()) {
        // The coroutine that we have just resumed has terminated its execution
        // and is suspended at its final suspension point. It has not pushed
        // anything else on top of the stack, so we can pop this.
        Stack.pop_back();

      } else {

        // The coroutine that we have just resumed has suspended its execution
        // and we can find a non-owning handle to the callee inside its promise
        // object.

        revng_assert(CurrentHandle and CurrentHandle.promise().Callee);
        PromiseBase *CalleePromise = CurrentHandle.promise().Callee;
        revng_assert(CalleePromise);

        // Push the callee handle on top of the stack for resumption.
        Stack.push_back(CalleePromise);
      }
    }

    revng_assert(OwnedHandle and OwnedHandle.done());

    return OwnedHandle.promise().get();
  }

private:
  coro_handle OwnedHandle;
};

template<>
struct RecursiveCoroutine<void>::promise_type : public PromiseBase {

  RecursiveCoroutine<void> get_return_object() {
    return RecursiveCoroutine<void>(coro_handle::from_promise(*this));
  }

  [[noreturn]] static RecursiveCoroutine<void>
  get_return_object_on_allocation_failure() {
    std::terminate();
    // TODO: if we need this not to be a hard crash we could do the following
    // return RecursiveCoroutine<void>();
  }

  void return_void() const { return; }
  void get() const {}
};

template<typename ReturnT>
struct RecursiveCoroutine<ReturnT>::promise_type : public PromiseBase {

  RecursiveCoroutine<ReturnT> get_return_object() {
    return RecursiveCoroutine<ReturnT>(coro_handle::from_promise(*this));
  }

  [[noreturn]] static RecursiveCoroutine<ReturnT>
  get_return_object_on_allocation_failure() {
    std::terminate();
    // TODO: if we need this not to be a hard crash we could do the following
    // return RecursiveCoroutine<ReturnT>();
  }

  void return_value(ReturnT R) {
    revng_assert(not CurrValue.has_value());
    CurrValue = std::move(R);
    return;
  }

  ReturnT get() const {
    revng_assert(CurrValue.has_value());
    return *CurrValue;
  }

protected:
  std::optional<ReturnT> CurrValue = std::nullopt;
};

template<typename CoroutineT, typename... Args>
auto rc_run(CoroutineT F, Args... Arguments) {
  return F(Arguments...).run();
}

#define rc_return co_return

#define rc_recur co_await
