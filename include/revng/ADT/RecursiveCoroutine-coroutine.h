#pragma once

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

#include <coroutine>
#include <optional>
#include <type_traits>
#include <utility>

#include "revng/Support/Assert.h"

template<typename>
struct RecursiveCoroutine;

namespace revng::detail {

template<typename RetT>
struct ReturnBase {

  void return_value(RetT R) {
    revng_assert(not CurrValue.has_value());
    CurrValue = std::move(R);
    return;
  }

  RetT &&take() {
    revng_assert(CurrValue.has_value());
    return std::move(CurrValue.value());
  }

protected:
  std::optional<RetT> CurrValue = std::nullopt;
};

template<>
struct ReturnBase<void> {

  void return_void() const { return; }
  void take() const {}
};

template<typename ReturnT>
struct RecursivePromise : public ReturnBase<ReturnT> {

  using promise_type = RecursivePromise<ReturnT>;
  using coro_handle = std::coroutine_handle<promise_type>;

  template<typename>
  friend struct RecursivePromise;

  RecursiveCoroutine<ReturnT> get_return_object() {
    return RecursiveCoroutine<ReturnT>(coro_handle::from_promise(*this));
  }

  [[noreturn]] static RecursiveCoroutine<ReturnT>
  get_return_object_on_allocation_failure() {
    std::terminate();
    // TODO: if we need this not to be a hard crash we could do the following
    // return RecursiveCoroutine<void>();
  }

  [[noreturn]] void unhandled_exception() const { std::terminate(); }

  auto initial_suspend() const { return std::suspend_always(); }

  auto final_suspend() noexcept {
    // In principle, we want our RecursiveCoroutine to always suspend at the
    // end of execution. Cleanup of coroutine resources is done in the
    // destructor of RecursiveCoroutine. However, we cannot simply return
    // suspend_always, because if have an associated AwaiterContinuation, it
    // means that the coroutine that is at this final suspension point was
    // being co_awaited by an awaiter, that must be resumed when the awaitee
    // reaches the final suspension point.
    //
    // So our final awaitable object should:
    // - always suspend (await_ready should always return false)
    // - return the handle to the awaiter continuation, so that if it's a
    //   valid handle it will be resumed.
    // - never be resumed, because it's the final suspension point
    //   (await_resume actually aborts)
    struct ResumeAwaiter {

      ResumeAwaiter(std::coroutine_handle<> AwaiterHandle) :
        Awaiter(AwaiterHandle) {}

      bool await_ready() const noexcept { return false; }

      std::coroutine_handle<> await_suspend(std::coroutine_handle<>) noexcept {
        std::coroutine_handle<> ToResume = std::noop_coroutine();
        if (Awaiter and not Awaiter.done()) {
          ToResume = Awaiter;
          Awaiter = {};
        }
        return ToResume;
      }

      void await_resume() const noexcept { revng_abort(); }

    private:
      std::coroutine_handle<> Awaiter;
    };

    auto ToResume = AwaiterContinuation;
    AwaiterContinuation = {};
    return ResumeAwaiter(ToResume);
  }

  RecursivePromise() : AwaiterContinuation(std::coroutine_handle<>{}) {}
  ~RecursivePromise() { revng_assert(not AwaiterContinuation); }

  // Not copyable, otherwise AwaiterContinuation could be resumed twice.
  RecursivePromise &operator=(const RecursivePromise &) = delete;
  RecursivePromise(const RecursivePromise &) = delete;

  RecursivePromise &operator=(RecursivePromise &&Other) {
    if (this != &Other) {
      this->AwaiterContinuation = Other.AwaiterContinuation;
      Other.AwaiterContinuation = {};
    }
    return *this;
  }
  RecursivePromise(RecursivePromise &&Other) { *this = std::move(Other); }

  template<typename AwaiteeReturnT>
  auto await_transform(RecursiveCoroutine<AwaiteeReturnT> &&Awaitee) {
    // This await_transform is called when the RecursiveCoroutine associated
    // with this promise object (the awaiter) calls co_await on another
    // RecursiveCoroutine (the awaitee).
    //
    // We want to inject the handle of the awaiter into the awaitee, so that
    // when the awaitee ends it can resume the awaiter.
    //
    // We also have to return an awaitable that will dictate the suspension
    // behavior of the awaiter.
    // We want it to always suspend (await_ready should return false); we want
    // it to resume the awaitee straight away after suspending (await_suspend
    // should return a coroutine_handle to the awaitee), and we want the
    // awaiter to get the result of the awaitee when resuming (await_resume
    // should get the result of the awaitee and return it to the awaiter).
    using AwaiteePromiseT = RecursivePromise<AwaiteeReturnT>;
    using AwaiteeCoroHandle = typename AwaiteePromiseT::coro_handle;
    AwaiteeCoroHandle AwaiteeHandle = Awaitee.OwnedHandle;
    revng_assert(AwaiteeHandle and not AwaiteeHandle.done());

    auto &AwaiteePromise = AwaiteeHandle.promise();
    AwaiteePromise.AwaiterContinuation = coro_handle::from_promise(*this);

    struct ResumeAwaitee {
      ResumeAwaitee(AwaiteeCoroHandle AwaiteeHandle) : Awaitee(AwaiteeHandle) {}

      bool await_ready() const { return false; }

      auto await_suspend(std::coroutine_handle<>) {
        // In principle we could clear Awaitee here, before suspending, so
        // that if for some reason `await_suspend` is called twice we don't
        // end up suspending the same coroutine twice (which is a bug).
        // However, we still need access to Awaitee in `await_resume` to take
        // the result of the Awaitee and propagate it to the awaiter.
        revng_assert(Awaitee and not Awaitee.done());
        return Awaitee;
      }

      auto await_resume() {
        revng_assert(Awaitee and Awaitee.done());

        if constexpr (std::is_void_v<AwaiteeReturnT>)
          return;
        else
          return Awaitee.promise().take();
      }

    private:
      AwaiteeCoroHandle Awaitee;
    };

    return ResumeAwaitee{ AwaiteeHandle };
  }

protected:
  std::coroutine_handle<> AwaiterContinuation;
};

} // namespace revng::detail

template<typename ReturnT = void>
struct RecursiveCoroutine {

public:
  using promise_type = revng::detail::RecursivePromise<ReturnT>;
  using coro_handle = std::coroutine_handle<promise_type>;
  template<typename T>
  friend struct revng::detail::RecursivePromise;

public:
  RecursiveCoroutine() = delete;
  RecursiveCoroutine(coro_handle H) : OwnedHandle(H) {}

  // Not copyable, because we don't want OwnedHandle to be destroyed twice
  RecursiveCoroutine &operator=(const RecursiveCoroutine &) = delete;
  RecursiveCoroutine(const RecursiveCoroutine &) = delete;

  // Movable, but we clean up the OwnedHandle, because we don't want it to be
  // destroyed twice in the destructor
  RecursiveCoroutine &operator=(RecursiveCoroutine &&Other) {
    this->OwnedHandle = Other.OwnedHandle;
    Other.OwnedHandle = {};
    return *this;
  }
  RecursiveCoroutine(RecursiveCoroutine &&Other) { *this = std::move(Other); }

  ~RecursiveCoroutine() {
    revng_assert(OwnedHandle);

    // If we reach this point, the coroutine associated to the OwnedHandle was
    // either awaited upon (and it is now suspended at its final suspension
    // point), or it was not awaited upon at all (simply called as a regular
    // function); in this last case its result has also not been taken with
    // operator ReturnT and is going to be discarded.
    // We need to run the coroutine anyway, to be consistent with function calls
    // and to trigger potential side-effects.
    if (not OwnedHandle.done())
      OwnedHandle.resume();
    revng_assert(OwnedHandle.done());

    // After we are done we can clean up
    OwnedHandle.destroy();
    OwnedHandle = {};
  }

  // This is necessary to enable calling a recursive coroutine that returns a
  // value in the same way you would call a regular function.
  operator ReturnT() {
    revng_assert(OwnedHandle);

    if (not OwnedHandle.done())
      OwnedHandle.resume();
    revng_assert(OwnedHandle.done());

    if constexpr (std::is_void_v<ReturnT>) {
      return;
    } else {
      return OwnedHandle.promise().take();
    }
  }

  auto operator*() { return *this->operator ReturnT(); }

protected:
  coro_handle OwnedHandle;
};

#define rc_return co_return

#define rc_recur co_await
