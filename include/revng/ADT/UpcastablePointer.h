#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <memory>
#include <type_traits>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"

#include "revng/ADT/STLExtras.h"
#include "revng/Support/Assert.h"

template<typename T>
struct concrete_types_traits;

template<typename T>
using concrete_types_traits_t = typename concrete_types_traits<T>::type;

template<typename T>
concept ConcreteTypeTraitCompatible = requires {
  typename concrete_types_traits_t<T>;
} && StrictSpecializationOf<concrete_types_traits_t<T>, std::tuple>;

template<typename T>
concept HasLLVMRTTI = requires(T *A) {
  { A->classof(A) } -> std::same_as<bool>;
};

template<typename T>
concept Upcastable = HasLLVMRTTI<T> and ConcreteTypeTraitCompatible<T>;

template<typename T>
using pointee = typename std::pointer_traits<std::decay_t<T>>::element_type;

template<typename T>
concept Dereferenceable = requires(T A) {
  { *A };
};

static_assert(Dereferenceable<int *>);
static_assert(not Dereferenceable<int>);

template<Dereferenceable T>
using Dereferenced = std::decay_t<decltype(*std::declval<T>())>;

template<typename T>
concept UpcastablePointerLike = Dereferenceable<T> and Upcastable<pointee<T>>;

template<typename T>
concept NotUpcastablePointerLike = not UpcastablePointerLike<T>;

template<typename ReturnT, typename L, UpcastablePointerLike P, size_t I = 0>
  requires(not std::is_void_v<ReturnT>)
ReturnT upcast(P &&Upcastable, const L &Callable, ReturnT &&IfNull) {
  using pointee = std::remove_reference_t<decltype(*Upcastable)>;
  using concrete_types = concrete_types_traits_t<pointee>;
  auto *Pointer = &*Upcastable;
  if (Pointer == nullptr)
    return std::forward<ReturnT>(IfNull);

  if constexpr (I < std::tuple_size_v<concrete_types>) {
    using type = std::tuple_element_t<I, concrete_types>;
    if (auto *Upcasted = llvm::dyn_cast<type>(Pointer)) {
      if constexpr (std::same_as<std::decay_t<ReturnT>, llvm::Error>)
        llvm::consumeError(std::move(IfNull));
      static_assert(not SpecializationOf<std::decay_t<ReturnT>,
                                         llvm::Expected>);

      return Callable(*Upcasted);
    } else {
      return upcast<ReturnT, L, P, I + 1>(std::forward<P>(Upcastable),
                                          Callable,
                                          std::forward<ReturnT>(IfNull));
    }
  } else {
    revng_abort();
  }
}

template<typename L, UpcastablePointerLike P>
void upcast(P &&Upcastable, L &&Callable) {
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
  void upcast(L &&Callable) {
    ::upcast(Pointer, std::forward<L>(Callable));
  }

  template<typename L>
  void upcast(L &&Callable) const {
    ::upcast(Pointer, std::forward<L>(Callable));
  }

private:
  using concrete_types = concrete_types_traits_t<T>;
  static constexpr void (*Deleter)(T *) = &destroy<T>;
  using inner_pointer = std::unique_ptr<T, decltype(Deleter)>;

public:
  using pointer = typename inner_pointer::pointer;
  using element_type = typename inner_pointer::element_type;

public:
  constexpr UpcastablePointer() noexcept : Pointer(nullptr, Deleter) {}
  explicit UpcastablePointer(pointer P) noexcept : Pointer(P, Deleter) {}

public:
  template<std::derived_from<T> Q, typename... Args>
  static UpcastablePointer<T> make(Args &&...TheArgs) {
    return UpcastablePointer<T>(new Q(std::forward<Args>(TheArgs)...));
  }

  UpcastablePointer copy() const {
    return UpcastablePointer(clone(Pointer.get()));
  }

public:
  UpcastablePointer &operator=(const UpcastablePointer &Other) {
    if (&Other != this) {
      Pointer.reset(clone(Other.Pointer.get()));
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
    }
    return *this;
  }

  UpcastablePointer(UpcastablePointer &&Other) noexcept :
    UpcastablePointer(nullptr) {
    *this = std::move(Other);
  }

  constexpr bool operator==(std::nullptr_t P) const noexcept {
    return Pointer == P;
  }

  UpcastablePointer &operator=(const T &Another) noexcept {
    Pointer.reset();
    ::upcast(&Another, [this]<typename UT>(const UT &Upcasted) {
      *this = make<UT>(Upcasted);
    });
    return *this;
  }

  UpcastablePointer(const T &Another) noexcept : UpcastablePointer(nullptr) {
    ::upcast(&Another, [this]<typename UT>(const UT &Upcasted) {
      *this = make<UT>(Upcasted);
    });
  }

public:
  template<UpcastablePointerLike PointerLike>
    requires(std::equality_comparable_with<T, Dereferenced<PointerLike>>)
  bool operator==(const PointerLike &Other) const {
    if (isEmpty() || Other.isEmpty())
      return Pointer == Other.Pointer;

    bool Result = false;
    upcast([&](auto &Upcasted) {
      Other.upcast([&](auto &OtherUpcasted) {
        using ThisType = std::remove_cvref_t<decltype(Upcasted)>;
        using OtherType = std::remove_cvref_t<decltype(OtherUpcasted)>;
        if constexpr (std::is_same_v<ThisType, OtherType>)
          Result = Upcasted == OtherUpcasted;
      });
    });
    return Result;
  }

  template<UpcastablePointerLike PointerLike>
    requires(std::three_way_comparable_with<T, Dereferenced<PointerLike>>)
  auto operator<=>(const PointerLike &Other) const {
    return *Pointer <=> *Other;
  }

  auto *get() noexcept { return Pointer.get(); }
  const auto *get() const noexcept { return Pointer.get(); }

  auto &operator*() { return *Pointer; }
  const auto &operator*() const { return *Pointer; }

  auto *operator->() noexcept { return Pointer.operator->(); }
  const auto *operator->() const noexcept { return Pointer.operator->(); }

  explicit operator bool() const noexcept { return static_cast<bool>(Pointer); }

  void reset(pointer Other = pointer()) noexcept { Pointer.reset(Other); }

  bool isEmpty() const noexcept { return Pointer == nullptr; }
  constexpr static UpcastablePointer empty() noexcept {
    return UpcastablePointer(nullptr);
  }

private:
  inner_pointer Pointer;
};
