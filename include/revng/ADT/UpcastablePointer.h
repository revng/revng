#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <memory>
#include <type_traits>

#include "llvm/Support/Casting.h"

#include "revng/ADT/STLExtras.h"
#include "revng/Support/Assert.h"
#include "revng/Support/Concepts.h"

template<typename T>
concept is_pointer = std::is_pointer_v<T>;

template<typename T>
struct concrete_types_traits;

template<typename T>
using concrete_types_traits_t = typename concrete_types_traits<T>::type;

template<typename T>
concept HasConcretTypeTraits = requires {
  typename concrete_types_traits_t<T>;
};

// clang-format off
template<typename T>
concept HasLLVMRTTI = requires(T *a) {
  { a->classof(a) } -> same_as<bool>;
};
// clang-format on

template<typename T>
concept Upcastable = HasLLVMRTTI<T> and HasConcretTypeTraits<T>;

template<typename T>
using pointee = typename std::pointer_traits<T>::element_type;

template<typename T>
concept PointerLike = requires(T a) {
  { *a };
};

static_assert(PointerLike<int *>);
static_assert(not PointerLike<int>);

template<typename T>
concept UpcastablePointerLike = PointerLike<T> and Upcastable<pointee<T>>;

template<typename T>
concept NotVoid = !same_as<void, T>;

template<NotVoid ReturnT, typename L, UpcastablePointerLike P, size_t I = 0>
ReturnT upcast(P &Upcastable, const L &Callable, const ReturnT &IfNull) {
  using pointee = std::remove_reference_t<decltype(*Upcastable)>;
  using concrete_types = concrete_types_traits_t<pointee>;
  auto *Pointer = &*Upcastable;
  if (Pointer == nullptr)
    return IfNull;

  if constexpr (I < std::tuple_size_v<concrete_types>) {
    using type = std::tuple_element_t<I, concrete_types>;
    if (auto *Upcasted = llvm::dyn_cast<type>(Pointer)) {
      return Callable(*Upcasted);
    } else {
      return upcast<ReturnT, L, P, I + 1>(Upcastable, Callable, IfNull);
    }
  } else {
    revng_abort();
  }
}

template<typename L, UpcastablePointerLike P>
void upcast(P &Upcastable, const L &Callable) {
  auto Wrapper = [&](auto &Upcasted) {
    Callable(Upcasted);
    return true;
  };
  upcast(Upcastable, Wrapper, false);
}

/// A unique_ptr copiable thanks to LLVM RTTI
template<Upcastable T>
class UpcastablePointer {
private:
  template<Upcastable P>
  static P *clone(P *Pointer) {
    auto Dispatcher = [](auto &Upcasted) -> P * {
      using type = std::remove_reference_t<decltype(Upcasted)>;
      return new type(Upcasted);
    };
    return upcast(Pointer, Dispatcher, static_cast<P *>(nullptr));
  }

  template<Upcastable P>
  static void destroy(P *Pointer) {
    upcast(Pointer, [](auto &Upcasted) { delete &Upcasted; });
  }

private:
  using concrete_types = concrete_types_traits_t<T>;
  static constexpr void (*deleter)(T *) = &destroy<T>;
  using inner_pointer = std::unique_ptr<T, decltype(deleter)>;

public:
  using pointer = typename inner_pointer::pointer;
  using element_type = typename inner_pointer::element_type;

public:
  constexpr UpcastablePointer() noexcept : Pointer(nullptr, deleter) {}
  constexpr UpcastablePointer(std::nullptr_t P) noexcept :
    Pointer(P, deleter) {}
  explicit UpcastablePointer(pointer P) noexcept : Pointer(P, deleter) {}
  UpcastablePointer(UpcastablePointer &&P) noexcept :
    Pointer(std::move(P.Pointer)) {
    revng_assert(Pointer.get_deleter() == deleter);
  }

public:
  UpcastablePointer(const UpcastablePointer &Other) :
    Pointer(nullptr, deleter) {
    *this = Other;
    revng_assert(Pointer.get_deleter() == deleter);
  }

  UpcastablePointer &operator=(const UpcastablePointer &Other) {
    Pointer.reset(clone(Other.Pointer.get()));
    revng_assert(Pointer.get_deleter() == deleter);
    return *this;
  }

  UpcastablePointer &operator=(UpcastablePointer &&Other) {
    Pointer.reset(Other.Pointer.release());
    revng_assert(Pointer.get_deleter() == deleter);
    return *this;
  }

  auto get() const noexcept { return Pointer.get(); }
  auto &operator*() const { return *Pointer; }
  auto *operator->() const noexcept { return Pointer.operator->(); }

  void reset(pointer Other = pointer()) noexcept {
    Pointer.reset(Other);
    revng_assert(Pointer.get_deleter() == deleter);
  }

private:
  inner_pointer Pointer;
};

template<typename T>
concept IsUpcastablePointer = is_specialization_v<T, UpcastablePointer>;
