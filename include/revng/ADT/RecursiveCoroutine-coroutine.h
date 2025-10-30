#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <coroutine>
#include <optional>
#include <type_traits>
#include <utility>

#include "revng/ADT/UniqueCoroutineHandle.h"
#include "revng/Support/Assert.h"

template<typename>
class RecursiveCoroutine;

namespace revng::detail {

template<typename RetT>
struct ReturnBase {

  template<typename U>
  void return_value(const RecursiveCoroutine<U> &) = delete;

  template<std::convertible_to<RetT> U = RetT>
  void return_value(U &&R) {
    revng_assert(not CurrValue.has_value());
    CurrValue.emplace(std::forward<U>(R));
  }

  RetT take() {
    revng_assert(CurrValue.has_value());
    return std::move(*CurrValue);
  }

protected:
  std::optional<RetT> CurrValue = std::nullopt;
};

template<>
struct ReturnBase<void> {

  void return_void() {}
  void take() {}
};

template<typename ReturnT>
struct RecursivePromise : public ReturnBase<ReturnT> {

  using CoroHandle = std::coroutine_handle<RecursivePromise>;

  template<typename>
  friend struct RecursivePromise;

  RecursiveCoroutine<ReturnT> get_return_object() {
    return RecursiveCoroutine<ReturnT>(CoroHandle::from_promise(*this));
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
    // All RecursiveCoroutines are suspended at the end of their execution, in
    // order to allow the owner of the coroutine to extract the result (if any),
    // before it is destroyed. However, it is not enough to simply return
    // suspend_always, because in the case that this coroutine was awaited by
    // another RecursiveCoroutine, that awaiting coroutine must be resumed via
    // symmetric transfer from this suspension point. To that end, a special
    // awaiter is returned from this function.
    //
    // The final_suspend awaiter should:
    // - Always suspend (await_ready returns false).
    // - Resume the awaiting coroutine via symmetric transfer (await_suspend
    //   returns the continuation handle (if any; otherwise noop_coroutine is
    //   returned).
    // - Never be resumed, because it's the final suspension point (await_resume
    //   aborts).
    struct Awaiter {

      Awaiter(std::coroutine_handle<> Continuation) :
        Continuation(Continuation) {}

      bool await_ready() const noexcept { return false; }

      std::coroutine_handle<> await_suspend(std::coroutine_handle<>) noexcept {
        return Continuation ? Continuation : std::noop_coroutine();
      }

      void await_resume() const noexcept { revng_abort(); }

    private:
      std::coroutine_handle<> Continuation;
    };

    // Transfers the continuation handle to the awaiter. The promise requires
    // that no continuation is associated with it upon its destruction.
    return Awaiter(std::exchange(Continuation, nullptr));
  }

  RecursivePromise() : Continuation(nullptr) {}

  RecursivePromise(RecursivePromise &&) = delete;
  RecursivePromise &operator=(RecursivePromise &&) = delete;

  RecursivePromise(const RecursivePromise &) = delete;
  RecursivePromise &operator=(const RecursivePromise &) = delete;

  ~RecursivePromise() { revng_assert(not Continuation); }

  template<typename AwaiteeReturnT>
  auto await_transform(RecursiveCoroutine<AwaiteeReturnT> &&Awaitee) {
    revng_assert(Awaitee.OwnedHandle);

    // This await_transform is called when the RecursiveCoroutine associated
    // with this promise object (the awaiting coroutine) awaits (using co_await)
    // another RecursiveCoroutine (the awaitee).
    //
    // Ownership of the awaitee is transferred from the RecursiveCoroutine
    // referring to the awaitee and into the returned awaiter.
    //
    // When the awaiting coroutine suspends, the awaitee is given a handle to
    // the awaiting coroutine to borrow. This allows the awaitee to resume the
    // awaiting coroutine via symmetric transfer from its final_suspend.
    //
    // We also have to return an awaiter that will dictate the suspension
    // behavior of the awaiting coroutine. Ownership of the handle referring to
    // the awaitee is transferred to the returned awaiter.
    //
    // The returned awaiter should:
    // - Always suspend (await_ready returns false).
    // - Resume the awaitee via symmetric transfer (await_suspend returns a
    //   coroutine_handle referring to the awaitee).
    // - Return the result of the awaitee to the awaiting coroutine
    //   (await_resume returns the result of the awaitee).
    // - Destroy the awaitee before transferring control back to the awaiting
    //   coroutine.
    using AwaiteePromiseT = RecursivePromise<AwaiteeReturnT>;
    using AwaiteeCoroHandle = typename AwaiteePromiseT::CoroHandle;

    struct Awaiter {
      Awaiter(AwaiteeCoroHandle AwaiteeHandle) : Awaitee(AwaiteeHandle) {}

      bool await_ready() const noexcept { return false; }

      std::coroutine_handle<>
      await_suspend(std::coroutine_handle<> Continuation) noexcept {
        // It is not possible to suspend the same coroutine twice.
        revng_assert(Awaitee);
        revng_assert(not Awaitee.done());
        revng_assert(not Awaitee.promise().Continuation);
        Awaitee.promise().Continuation = Continuation;
        return Awaitee;
      }

      AwaiteeReturnT await_resume() {
        revng_assert(Awaitee.done());

        // Ownership of the coroutine is moved to a local unique handle in order
        // to destroy the coroutine after returning the result (if any), but
        // before transferring control back to the awaiter.
        UniqueCoroutineHandle<AwaiteePromiseT> Handle(Awaitee);
        Awaitee = nullptr;

        if constexpr (std::is_void_v<AwaiteeReturnT>) {
          return;
        } else {
          return Handle.promise().take();
        }
      }

    private:
      AwaiteeCoroHandle Awaitee;
    };

    // Ownership of the coroutine is transferred to the awaiter.
    return Awaiter(std::exchange(Awaitee.OwnedHandle, nullptr));
  }

private:
  std::coroutine_handle<> Continuation;
};

} // namespace revng::detail

template<typename ReturnT = void>
class RecursiveCoroutine {

public:
  using promise_type = revng::detail::RecursivePromise<ReturnT>;

private:
  using CoroHandle = std::coroutine_handle<promise_type>;

  template<typename T>
  friend struct revng::detail::RecursivePromise;

public:
  RecursiveCoroutine() = delete;
  RecursiveCoroutine(CoroHandle H) : OwnedHandle(H) {}

  // RecursiveCoroutine should be neither movable nor copyable, because the
  // coroutine implementation maintains compatibility with the non-coroutine
  // fallback implementation, and it is not possible to move a function call.
  // For this reason the interface of this class is as limited as possible.
  RecursiveCoroutine(RecursiveCoroutine &&) = delete;
  RecursiveCoroutine &operator=(RecursiveCoroutine &&) = delete;

  RecursiveCoroutine(const RecursiveCoroutine &) = delete;
  RecursiveCoroutine &operator=(const RecursiveCoroutine &) = delete;

  ~RecursiveCoroutine() {
    if (OwnedHandle) {
      // If we reach this point, the coroutine associated with the OwnedHandle
      // was neither awaited upon nor evaluated. We need to run the coroutine
      // anyway, for consistency with function calls and to trigger potential
      // side-effects.
      runCoroutine(OwnedHandle);

      // After we are done we can clean up.
      OwnedHandle.destroy();
    }
  }

  // Conversion operators are necessary to enable calling a recursive coroutine
  // that returns a value in the same way you would call a regular subroutine.
  // The RecursiveCoroutine is convertible to any type which ReturnT is also
  // convertible. This conversion is explicit if and only if the corresponding
  // conversion from ReturnT is explicit.
  //
  // ConvertedT must be a cv-unqualified non-reference type.
  template<typename ConvertedT>
    requires std::is_constructible_v<ConvertedT, ReturnT>
             and std::is_same_v<ConvertedT, std::remove_cvref_t<ConvertedT>>
  explicit(not std::is_convertible_v<ReturnT, ConvertedT>)
  operator ConvertedT() && {
    return static_cast<ConvertedT>(evaluateImpl());
  }

  // An implicit non-template conversion operator for ReturnT is necessary for
  // cases where a subsequent derived-to-base conversion is applied.
  operator ReturnT() && { return evaluateImpl(); }

  auto operator*() && { return *evaluateImpl(); }

private:
  ReturnT evaluateImpl() {
    revng_assert(OwnedHandle);

    // Ownership of the coroutine is moved to a local unique handle in order to
    // destroy the coroutine after returning the result (if any), but before
    // transferring control back to the caller.
    UniqueCoroutineHandle<promise_type> Handle(OwnedHandle);
    OwnedHandle = nullptr;

    runCoroutine(Handle.get());

    if constexpr (std::is_void_v<ReturnT>) {
      return;
    } else {
      return Handle.promise().take();
    }
  }

  static void runCoroutine(CoroHandle Handle) {
    revng_assert(not Handle.done());
    Handle.resume();
    revng_assert(Handle.done());
  }

  CoroHandle OwnedHandle;
};

#define rc_return co_return

#define rc_recur co_await
