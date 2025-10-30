#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <coroutine>

template<typename PromiseT = void>
class UniqueCoroutineHandle {
  using HandleType = std::coroutine_handle<PromiseT>;

  HandleType Handle;

public:
  constexpr UniqueCoroutineHandle() noexcept : Handle(nullptr) {}

  constexpr UniqueCoroutineHandle(decltype(nullptr)) noexcept :
    Handle(nullptr) {}

  explicit constexpr UniqueCoroutineHandle(HandleType Handle) noexcept :
    Handle(Handle) {}

  constexpr UniqueCoroutineHandle(UniqueCoroutineHandle &&Other) noexcept :
    Handle(Other.Handle) {
    Other.Handle = nullptr;
  }

  constexpr UniqueCoroutineHandle &
  operator=(UniqueCoroutineHandle &&Other) & noexcept {
    HandleType OtherHandle = Other.Handle;
    Other.Handle = nullptr;

    if (Handle)
      Handle.destroy();

    Handle = OtherHandle;
    return *this;
  }

  constexpr ~UniqueCoroutineHandle() {
    if (Handle)
      Handle.destroy();
  }

  friend constexpr void swap(UniqueCoroutineHandle &LHS,
                             UniqueCoroutineHandle &RHS) noexcept {
    HandleType LHSHandle = LHS.Handle;
    LHS.Handle = RHS.Handle;
    RHS.Handle = LHSHandle;
  }

  [[nodiscard]] HandleType get() const noexcept { return Handle; }

  [[nodiscard]] bool done() const noexcept { return Handle.done(); }

  [[nodiscard]] HandleType release() noexcept {
    auto Handle = this->Handle;
    this->Handle = nullptr;
    return Handle;
  }

  [[nodiscard]] PromiseT &promise() const noexcept { return Handle.promise(); }

  void resume() const noexcept { Handle.resume(); }

  void reset() noexcept {
    HandleType Handle = this->Handle;
    this->Handle = nullptr;

    if (Handle)
      Handle.destroy();
  }

  [[nodiscard]] explicit constexpr operator bool() const noexcept {
    return static_cast<bool>(Handle);
  }
};
