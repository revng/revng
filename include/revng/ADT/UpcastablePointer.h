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
concept HasLLVMRTTI = requires(T *A) {
  { A->classof(A) } -> same_as<bool>;
};
// clang-format on

template<typename T>
concept Upcastable = HasLLVMRTTI<T> and HasConcretTypeTraits<T>;

template<typename T>
using pointee = typename std::pointer_traits<std::decay_t<T>>::element_type;

template<typename T>
concept PointerLike = requires(T A) {
  { *A };
};

static_assert(PointerLike<int *>);
static_assert(not PointerLike<int>);

template<typename T>
concept UpcastablePointerLike = PointerLike<T> and Upcastable<pointee<T>>;

template<typename T>
concept NotVoid = not std::is_void_v<T>;

template<class Derived, class Base>
concept DerivesFrom = std::is_base_of_v<Base, Derived>;

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
    return ::upcast(Pointer, Dispatcher, static_cast<P *>(nullptr));
  }

  template<Upcastable P>
  static void destroy(P *Pointer) {
    ::upcast(Pointer, [](auto &Upcasted) { delete &Upcasted; });
  }

public:
  template<typename L>
  void upcast(const L &Callable) {
    ::upcast(Pointer, Callable);
  }

  template<typename L>
  void upcast(const L &Callable) const {
    ::upcast(Pointer, Callable);
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

public:
  template<DerivesFrom<T> Q, typename... Args>
  static UpcastablePointer<T> make(Args &&...TheArgs) {
    return UpcastablePointer<T>(new Q(std::forward<Args>(TheArgs)...));
  }

public:
  UpcastablePointer &operator=(const UpcastablePointer &Other) {
    if (&Other != this) {
      Pointer.reset(clone(Other.Pointer.get()));
      revng_assert(Pointer.get_deleter() == deleter);
    }
    return *this;
  }

  UpcastablePointer(const UpcastablePointer &Other) :
    UpcastablePointer(nullptr) {
    *this = Other;
  }

  UpcastablePointer &operator=(UpcastablePointer &&Other) {
    if (&Other != this) {
      Pointer.reset(Other.Pointer.release());
      revng_assert(Pointer.get_deleter() == deleter);
    }
    return *this;
  }

  UpcastablePointer(UpcastablePointer &&Other) noexcept :
    UpcastablePointer(nullptr) {
    *this = std::move(Other);
  }

  bool operator==(const UpcastablePointer &Other) const {
    bool Result = false;
    upcast([&](auto &Upcasted) {
      Other.upcast([&](auto &OtherUpcasted) {
        using ThisType = std::remove_cvref_t<decltype(Upcasted)>;
        using OtherType = std::remove_cvref_t<decltype(OtherUpcasted)>;
        if constexpr (std::is_same_v<ThisType, OtherType>) {
          Result = Upcasted == OtherUpcasted;
        }
      });
    });
    return Result;
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

template<typename T>
concept IsNotUpcastablePointer = not IsUpcastablePointer<T>;
