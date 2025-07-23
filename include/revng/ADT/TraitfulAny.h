#include <set>
#include <memory>
#include <cstdlib>
#include <initializer_list>
#include <type_traits>
#include <typeinfo>
#include <utility>
#include <version>

#if 0
#define _LIBCPP_NORETURN
#define _LIBCPP_INLINE_VISIBILITY
#define _LIBCPP_SUPPRESS_DEPRECATED_PUSH
#define _LIBCPP_SUPPRESS_DEPRECATED_POP
#define _NOEXCEPT noexcept
#define _LIBCPP_TEMPLATE_VIS
#endif

template<template<typename> class Trait>
struct TraitActionImpl;

template<template<typename> class Trait>
using TraitAction = TraitActionImpl<Trait>::type;


template <class _Tp> struct __is_inplace_type_imp : std::false_type {};
template <class _Tp> struct __is_inplace_type_imp<std::in_place_type_t<_Tp>> : std::true_type {};

template <class _Tp>
using __is_inplace_type = __is_inplace_type_imp<std::remove_cvref_t<_Tp>>;

template <class _Alloc>
class __allocator_destructor
{
    typedef std::allocator_traits<_Alloc> __alloc_traits;
public:
    typedef typename __alloc_traits::pointer pointer;
    typedef typename __alloc_traits::size_type size_type;
private:
    _Alloc& __alloc_;
    std::size_t __s_;
public:
    _LIBCPP_INLINE_VISIBILITY __allocator_destructor(_Alloc& __a, std::size_t __s)
             _NOEXCEPT
        : __alloc_(__a), __s_(__s) {}
    _LIBCPP_INLINE_VISIBILITY
    void operator()(pointer __p) _NOEXCEPT
        {__alloc_traits::deallocate(__alloc_, __p, __s_);}
};


// Forward declarations
template<template<typename> class Trait>
class any;

template <class _ValueType, template<typename> class Trait>
_LIBCPP_INLINE_VISIBILITY
std::add_pointer_t<std::add_const_t<_ValueType>>
any_cast(any<Trait> const *) _NOEXCEPT;

template <class _ValueType, template<typename> class Trait>
_LIBCPP_INLINE_VISIBILITY
std::add_pointer_t<_ValueType> any_cast(any<Trait> *) _NOEXCEPT;

namespace __any_imp
{
  _LIBCPP_SUPPRESS_DEPRECATED_PUSH
  using _Buffer = std::aligned_storage_t<3*sizeof(void*), std::alignment_of<void*>::value>;
  _LIBCPP_SUPPRESS_DEPRECATED_POP

  template <class _Tp>
  using _IsSmallObject = std::integral_constant<bool
        , sizeof(_Tp) <= sizeof(_Buffer)
          && std::alignment_of<_Buffer>::value
             % std::alignment_of<_Tp>::value == 0
          && std::is_nothrow_move_constructible<_Tp>::value
        >;

  enum class _Action {
    _Destroy,
    _Copy,
    _Move,
    _Get,
    _Custom
  };

  template <class _Tp, template<typename> class Trait> struct _SmallHandler;
  template <class _Tp, template<typename> class Trait> struct _LargeHandler;

  template <class _Tp>
  struct  _LIBCPP_TEMPLATE_VIS __unique_typeinfo { static constexpr int __id = 0; };
  template <class _Tp> constexpr int __unique_typeinfo<_Tp>::__id;

  template <class _Tp>
  inline _LIBCPP_INLINE_VISIBILITY
  constexpr const void* __get_fallback_typeid() {
      return &__unique_typeinfo<std::remove_cv_t<std::remove_reference_t<_Tp>>>::__id;
  }

  template <class _Tp>
  inline _LIBCPP_INLINE_VISIBILITY
  bool __compare_typeid(const void* __fallback_id)
  {
      if (__fallback_id == __any_imp::__get_fallback_typeid<_Tp>())
          return true;
      return false;
  }

  template <class _Tp, template<typename> class Trait>
  using _Handler = std::conditional_t<
    _IsSmallObject<_Tp>::value, _SmallHandler<_Tp, Trait>, _LargeHandler<_Tp, Trait>>;

} // namespace __any_imp

template<template<typename> class Trait>
class _LIBCPP_TEMPLATE_VIS any
{
public:
  // construct/destruct
  _LIBCPP_INLINE_VISIBILITY
  constexpr any() _NOEXCEPT : __h_(nullptr) {}

  _LIBCPP_INLINE_VISIBILITY
  any(any const & __other) : __h_(nullptr)
  {
    if (__other.__h_) __other.__call(_Action::_Copy, this);
  }

  _LIBCPP_INLINE_VISIBILITY
  any(any && __other) _NOEXCEPT : __h_(nullptr)
  {
    if (__other.__h_) __other.__call(_Action::_Move, this);
  }

  template <
      class _ValueType
    , class _Tp = std::decay_t<_ValueType>
    , class = std::enable_if_t<
        !std::is_same<_Tp, any>::value &&
        !__is_inplace_type<_ValueType>::value &&
        std::is_copy_constructible<_Tp>::value>
    >
  _LIBCPP_INLINE_VISIBILITY
  any(_ValueType && __value);

  template <class _ValueType, class ..._Args,
    class _Tp = std::decay_t<_ValueType>,
    class = std::enable_if_t<
        std::is_constructible<_Tp, _Args...>::value &&
        std::is_copy_constructible<_Tp>::value
    >
  >
  _LIBCPP_INLINE_VISIBILITY
  explicit any(std::in_place_type_t<_ValueType>, _Args&&... __args);

  template <class _ValueType, class _Up, class ..._Args,
    class _Tp = std::decay_t<_ValueType>,
    class = std::enable_if_t<
        std::is_constructible<_Tp, std::initializer_list<_Up>&, _Args...>::value &&
        std::is_copy_constructible<_Tp>::value>
  >
  _LIBCPP_INLINE_VISIBILITY
  explicit any(std::in_place_type_t<_ValueType>, std::initializer_list<_Up>, _Args&&... __args);

  _LIBCPP_INLINE_VISIBILITY
  ~any() { this->reset(); }

  // assignments
  _LIBCPP_INLINE_VISIBILITY
  any & operator=(any const & __rhs) {
    any(__rhs).swap(*this);
    return *this;
  }

  _LIBCPP_INLINE_VISIBILITY
  any & operator=(any && __rhs) _NOEXCEPT {
    any(std::move(__rhs)).swap(*this);
    return *this;
  }

  template <
      class _ValueType
    , class _Tp = std::decay_t<_ValueType>
    , class = std::enable_if_t<
          !std::is_same<_Tp, any>::value
          && std::is_copy_constructible<_Tp>::value>
    >
  _LIBCPP_INLINE_VISIBILITY
  any & operator=(_ValueType && __rhs);

  template <class _ValueType, class ..._Args,
    class _Tp = std::decay_t<_ValueType>,
    class = std::enable_if_t<
        std::is_constructible<_Tp, _Args...>::value &&
        std::is_copy_constructible<_Tp>::value>
    >
  _LIBCPP_INLINE_VISIBILITY
  _Tp& emplace(_Args&&...);

  template <class _ValueType, class _Up, class ..._Args,
    class _Tp = std::decay_t<_ValueType>,
    class = std::enable_if_t<
        std::is_constructible<_Tp, std::initializer_list<_Up>&, _Args...>::value &&
        std::is_copy_constructible<_Tp>::value>
  >
  _LIBCPP_INLINE_VISIBILITY
  _Tp& emplace(std::initializer_list<_Up>, _Args&&...);

  // 6.3.3 any modifiers
  _LIBCPP_INLINE_VISIBILITY
  void reset() _NOEXCEPT { if (__h_) this->__call(_Action::_Destroy); }

  _LIBCPP_INLINE_VISIBILITY
  void swap(any & __rhs) _NOEXCEPT;

  // 6.3.4 any observers
  _LIBCPP_INLINE_VISIBILITY
  bool has_value() const _NOEXCEPT { return __h_ != nullptr; }

public:
    intptr_t type_id() const { return reinterpret_cast<intptr_t>(__h_); }

protected:
    // WIP
    using CustomActionType = TraitAction<Trait>;

    _LIBCPP_INLINE_VISIBILITY
    void * call(CustomActionType Action, any * __other = nullptr)
    {
        return __h_(static_cast<_Action>(static_cast<size_t>(Action) + static_cast<size_t>(_Action::_Custom)), this, __other, nullptr);
    }


private:
    typedef __any_imp::_Action _Action;
    using _HandleFuncPtr =  void* (*)(_Action, any const *, any *,
      const void* __fallback_info);

    union _Storage {
        constexpr _Storage() : __ptr(nullptr) {}
        void *  __ptr;
        __any_imp::_Buffer __buf;
    };

    _LIBCPP_INLINE_VISIBILITY
    void * __call(_Action __a, any * __other = nullptr,
                   const void* __fallback_info = nullptr) const
    {
        return __h_(__a, this, __other, __fallback_info);
    }

    _LIBCPP_INLINE_VISIBILITY
    void * __call(_Action __a, any * __other = nullptr,
                  const void* __fallback_info = nullptr)
    {
        return __h_(__a, this, __other, __fallback_info);
    }

    template <class, template<typename> class>
    friend struct __any_imp::_SmallHandler;
    template <class, template<typename> class>
    friend struct __any_imp::_LargeHandler;

    template <class _ValueType>
    friend std::add_pointer_t<std::add_const_t<_ValueType>>
    any_cast(any const *) _NOEXCEPT;

    template <class _ValueType>
    friend std::add_pointer_t<_ValueType>
    any_cast(any *) _NOEXCEPT;

    _HandleFuncPtr __h_ = nullptr;
    _Storage __s_;
};

namespace __any_imp
{
  template <class _Tp, template<typename> class Trait>
  struct _LIBCPP_TEMPLATE_VIS _SmallHandler
  {
     using any = any<Trait>;
     _LIBCPP_INLINE_VISIBILITY
     static void* __handle(_Action __act, any const * __this, any * __other,
                           const void* __fallback_info)
     {
        switch (__act)
        {
        case _Action::_Destroy:
          __destroy(const_cast<any &>(*__this));
          return nullptr;
        case _Action::_Copy:
            __copy(*__this, *__other);
            return nullptr;
        case _Action::_Move:
          __move(const_cast<any &>(*__this), *__other);
          return nullptr;
        case _Action::_Get:
            return __get(const_cast<any &>(*__this), __fallback_info);
        case _Action::_Custom:
            using CustomActionType = TraitAction<Trait>;
            return reinterpret_cast<void *>(Trait<_Tp>::handle(static_cast<CustomActionType>(static_cast<size_t>(__act) - static_cast<size_t>(_Action::_Custom)),
                                      const_cast<any *>(__this),
                                      __other));
        }
        abort();
    }

    template <class ..._Args>
    _LIBCPP_INLINE_VISIBILITY
    static _Tp& __create(any & __dest, _Args&&... __args) {
        typedef std::allocator<_Tp> _Alloc;
        typedef std::allocator_traits<_Alloc> _ATraits;
        _Alloc __a;
        _Tp * __ret = static_cast<_Tp*>(static_cast<void*>(&__dest.__s_.__buf));
        _ATraits::construct(__a, __ret, std::forward<_Args>(__args)...);
        __dest.__h_ = &_SmallHandler::__handle;
        return *__ret;
    }

  private:
    _LIBCPP_INLINE_VISIBILITY
    static void __destroy(any & __this) {
        typedef std::allocator<_Tp> _Alloc;
        typedef std::allocator_traits<_Alloc> _ATraits;
        _Alloc __a;
        _Tp * __p = static_cast<_Tp *>(static_cast<void*>(&__this.__s_.__buf));
        _ATraits::destroy(__a, __p);
        __this.__h_ = nullptr;
    }

    _LIBCPP_INLINE_VISIBILITY
    static void __copy(any const & __this, any & __dest) {
        _SmallHandler::__create(__dest, *static_cast<_Tp const *>(
            static_cast<void const *>(&__this.__s_.__buf)));
    }

    _LIBCPP_INLINE_VISIBILITY
    static void __move(any & __this, any & __dest) {
        _SmallHandler::__create(__dest, std::move(
            *static_cast<_Tp*>(static_cast<void*>(&__this.__s_.__buf))));
        __destroy(__this);
    }

    _LIBCPP_INLINE_VISIBILITY
    static void* __get(any & __this,
                       const void* __fallback_id)
    {
        if (__any_imp::__compare_typeid<_Tp>(__fallback_id))
            return static_cast<void*>(&__this.__s_.__buf);
        return nullptr;
    }

  };

  template <class _Tp, template<typename> class Trait>
  struct _LIBCPP_TEMPLATE_VIS _LargeHandler
  {
    using any = any<Trait>;

    _LIBCPP_INLINE_VISIBILITY
    static void* __handle(_Action __act, any const * __this,
                          any * __other, 
                          void const* __fallback_info)
    {
        switch (__act)
        {
        case _Action::_Destroy:
          __destroy(const_cast<any &>(*__this));
          return nullptr;
        case _Action::_Copy:
          __copy(*__this, *__other);
          return nullptr;
        case _Action::_Move:
          __move(const_cast<any &>(*__this), *__other);
          return nullptr;
        case _Action::_Get:
            return __get(const_cast<any &>(*__this), __fallback_info);
        case _Action::_Custom:
            abort();
        }
        abort();
    }

    template <class ..._Args>
    _LIBCPP_INLINE_VISIBILITY
    static _Tp& __create(any & __dest, _Args&&... __args) {
        typedef std::allocator<_Tp> _Alloc;
        typedef std::allocator_traits<_Alloc> _ATraits;
        typedef __allocator_destructor<_Alloc> _Dp;
        _Alloc __a;
        std::unique_ptr<_Tp, _Dp> __hold(_ATraits::allocate(__a, 1), _Dp(__a, 1));
        _Tp * __ret = __hold.get();
        _ATraits::construct(__a, __ret, std::forward<_Args>(__args)...);
        __dest.__s_.__ptr = __hold.release();
        __dest.__h_ = &_LargeHandler::__handle;
        return *__ret;
    }

  private:

    _LIBCPP_INLINE_VISIBILITY
    static void __destroy(any & __this){
        typedef std::allocator<_Tp> _Alloc;
        typedef std::allocator_traits<_Alloc> _ATraits;
        _Alloc __a;
        _Tp * __p = static_cast<_Tp *>(__this.__s_.__ptr);
        _ATraits::destroy(__a, __p);
        _ATraits::deallocate(__a, __p, 1);
        __this.__h_ = nullptr;
    }

    _LIBCPP_INLINE_VISIBILITY
    static void __copy(any const & __this, any & __dest) {
        _LargeHandler::__create(__dest, *static_cast<_Tp const *>(__this.__s_.__ptr));
    }

    _LIBCPP_INLINE_VISIBILITY
    static void __move(any & __this, any & __dest) {
      __dest.__s_.__ptr = __this.__s_.__ptr;
      __dest.__h_ = &_LargeHandler::__handle;
      __this.__h_ = nullptr;
    }

    _LIBCPP_INLINE_VISIBILITY
    static void* __get(any & __this,
                       void const* __fallback_info)
    {
        if (__any_imp::__compare_typeid<_Tp>(__fallback_info))
            return static_cast<void*>(__this.__s_.__ptr);
        return nullptr;

    }

  };

} // namespace __any_imp


template<template<typename> class Trait>
template <class _ValueType, class _Tp, class>
any<Trait>::any(_ValueType && __v) : __h_(nullptr)
{
  __any_imp::_Handler<_Tp, Trait>::__create(*this, std::forward<_ValueType>(__v));
}

template<template<typename> class Trait>
template <class _ValueType, class ..._Args, class _Tp, class>
any<Trait>::any(std::in_place_type_t<_ValueType>, _Args&&... __args) {
  __any_imp::_Handler<_Tp, Trait>::__create(*this, std::forward<_Args>(__args)...);
}

template<template<typename> class Trait>
template <class _ValueType, class _Up, class ..._Args, class _Tp, class>
any<Trait>::any(std::in_place_type_t<_ValueType>, std::initializer_list<_Up> __il, _Args&&... __args) {
  __any_imp::_Handler<_Tp, Trait>::__create(*this, __il, std::forward<_Args>(__args)...);
}

template<template<typename> class Trait>
template <class _ValueType, class, class>
inline _LIBCPP_INLINE_VISIBILITY
any<Trait> & any<Trait>::operator=(_ValueType && __v)
{
  any(std::forward<_ValueType>(__v)).swap(*this);
  return *this;
}

template<template<typename> class Trait>
template <class _ValueType, class ..._Args, class _Tp, class>
inline _LIBCPP_INLINE_VISIBILITY
_Tp& any<Trait>::emplace(_Args&&... __args) {
  reset();
  return __any_imp::_Handler<_Tp, Trait>::__create(*this, std::forward<_Args>(__args)...);
}

template<template<typename> class Trait>
template <class _ValueType, class _Up, class ..._Args, class _Tp, class>
inline _LIBCPP_INLINE_VISIBILITY
_Tp& any<Trait>::emplace(std::initializer_list<_Up> __il, _Args&&... __args) {
  reset();
  return __any_imp::_Handler<_Tp, Trait>::__create(*this, __il, std::forward<_Args>(__args)...);
}

template<template<typename> class Trait>
inline _LIBCPP_INLINE_VISIBILITY
void any<Trait>::swap(any & __rhs) _NOEXCEPT
{
    if (this == &__rhs)
      return;
    if (__h_ && __rhs.__h_) {
        any __tmp;
        __rhs.__call(_Action::_Move, &__tmp);
        this->__call(_Action::_Move, &__rhs);
        __tmp.__call(_Action::_Move, this);
    }
    else if (__h_) {
        this->__call(_Action::_Move, &__rhs);
    }
    else if (__rhs.__h_) {
        __rhs.__call(_Action::_Move, this);
    }
}

// 6.4 Non-member functions

template<template<typename> class Trait>
inline _LIBCPP_INLINE_VISIBILITY
void swap(any<Trait> & __lhs, any<Trait> & __rhs) _NOEXCEPT
{
    __lhs.swap(__rhs);
}

template <class _Tp, template<typename> class Trait, class ..._Args>
inline _LIBCPP_INLINE_VISIBILITY
any<Trait> make_any(_Args&&... __args) {
    return any(std::in_place_type<_Tp>, std::forward<_Args>(__args)...);
}

template <class _Tp, template<typename> class Trait, class _Up, class ..._Args>
inline _LIBCPP_INLINE_VISIBILITY
any<Trait> make_any(std::initializer_list<_Up> __il, _Args&&... __args) {
    return any(std::in_place_type<_Tp>, __il, std::forward<_Args>(__args)...);
}

template <class _ValueType, template<typename> class Trait>
inline _LIBCPP_INLINE_VISIBILITY
_ValueType any_cast(any<Trait> const & __v)
{
    using _RawValueType = std::remove_cvref_t<_ValueType>;
    static_assert(std::is_constructible<_ValueType, _RawValueType const &>::value,
                  "ValueType is required to be a const lvalue reference "
                  "or a CopyConstructible type");
    auto __tmp = any_cast<std::add_const_t<_RawValueType>>(&__v);
    if (__tmp == nullptr)
        abort();
    return static_cast<_ValueType>(*__tmp);
}

template <class _ValueType, template<typename> class Trait>
inline _LIBCPP_INLINE_VISIBILITY
_ValueType any_cast(any<Trait> & __v)
{
    using _RawValueType = std::remove_cvref_t<_ValueType>;
    static_assert(std::is_constructible<_ValueType, _RawValueType &>::value,
                  "ValueType is required to be an lvalue reference "
                  "or a CopyConstructible type");
    auto __tmp = any_cast<_RawValueType>(&__v);
    if (__tmp == nullptr)
        abort();
    return static_cast<_ValueType>(*__tmp);
}

template <class _ValueType, template<typename> class Trait>
inline _LIBCPP_INLINE_VISIBILITY
_ValueType any_cast(any<Trait> && __v)
{
    using _RawValueType = std::remove_cvref_t<_ValueType>;
    static_assert(std::is_constructible<_ValueType, _RawValueType>::value,
                  "ValueType is required to be an rvalue reference "
                  "or a CopyConstructible type");
    auto __tmp = any_cast<_RawValueType>(&__v);
    if (__tmp == nullptr)
        abort();
    return static_cast<_ValueType>(std::move(*__tmp));
}

template <class _ValueType,template<typename> class Trait>
inline _LIBCPP_INLINE_VISIBILITY
std::add_pointer_t<std::add_const_t<_ValueType>>
any_cast(any<Trait> const * __any) _NOEXCEPT
{
    static_assert(!std::is_reference<_ValueType>::value,
                  "_ValueType may not be a reference.");
    return any_cast<_ValueType>(const_cast<any<Trait> *>(__any));
}

template <class _RetType>
inline _LIBCPP_INLINE_VISIBILITY
_RetType __pointer_or_func_cast(void* __p, /*IsFunction*/std::false_type) noexcept {
  return static_cast<_RetType>(__p);
}

template <class _RetType>
inline _LIBCPP_INLINE_VISIBILITY
_RetType __pointer_or_func_cast(void*, /*IsFunction*/std::true_type) noexcept {
  return nullptr;
}

template <class _ValueType, template<typename> class Trait>
std::add_pointer_t<_ValueType>
any_cast(any<Trait> * __any) _NOEXCEPT
{
    using __any_imp::_Action;
    static_assert(!std::is_reference<_ValueType>::value,
                  "_ValueType may not be a reference.");
    typedef std::add_pointer_t<_ValueType> _ReturnType;
    if (__any && __any->__h_) {
      void *__p = __any->__call(_Action::_Get, nullptr,
                          nullptr,
                          __any_imp::__get_fallback_typeid<_ValueType>());
        return __pointer_or_func_cast<_ReturnType>(
            __p, std::is_function<_ValueType>{});
    }
    return nullptr;
}

enum class MyTraitAction {
    Compare
};

template<typename T>
struct MyTrait;

template<>
struct TraitActionImpl<MyTrait> { using type = MyTraitAction; };

template<typename T>
struct MyTrait {
    static intptr_t handle(MyTraitAction Action, any<MyTrait> *First, any<MyTrait> *Second) {
        switch (Action) {
        case MyTraitAction::Compare:
            if (First->type_id() == Second->type_id()) {
                return *any_cast<T>(First) < *any_cast<T>(Second);
            } else {
                return First->type_id() < Second->type_id();
            }
            break;
        default:
            abort();
        }
    }
};

class MyAny : public any<MyTrait> {
    bool operator<(const MyAny &Other) const {
        return const_cast<MyAny *>(this)->call(MyTraitAction::Compare, const_cast<MyAny *>(&Other));
    }
};

MyAny A(2);
std::set<MyAny> X;
