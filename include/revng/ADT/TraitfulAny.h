#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Note: this file is a copy of LLVM 16's libcxx/include/any, with some changes.
// It is designed to be diff-able so we can merge changes.
// This is also why the formatting is not the official one and why we do weird
// macro tricks.
// Please change carefully and make sure you minimize the changes w.r.t. the
// original file.

#include <memory>


namespace revng {

using std::add_const_t;
using std::add_pointer_t;
using std::aligned_storage_t;
using std::alignment_of;
using std::allocator_traits;
using std::allocator;
using std::conditional_t;
using std::decay_t;
using std::enable_if_t;
using std::false_type;
using std::forward;
using std::in_place_type_t;
using std::in_place_type;
using std::initializer_list;
using std::integral_constant;
using std::is_constructible;
using std::is_copy_constructible;
using std::is_function;
using std::is_nothrow_move_constructible;
using std::is_reference;
using std::is_same;
using std::move;
using std::remove_cv_t;
using std::remove_cvref_t;
using std::remove_reference_t;
using std::size_t;
using std::true_type;
using std::unique_ptr;

using type_info = void;

#pragma push_macro("_VSTD")

#ifdef _VSTD
#undef _VSTD
#endif
#define _VSTD std
#define __remove_cvref_t remove_cvref_t
#define __throw_bad_any_cast abort
#define __libcpp_unreachable abort
#define _LIBCPP_HAS_NO_RTTI

template <class _Tp> struct __is_inplace_type_imp : false_type {};
template <class _Tp> struct __is_inplace_type_imp<in_place_type_t<_Tp>> : true_type {};

template <class _Tp>
using __is_inplace_type = __is_inplace_type_imp<remove_cvref_t<_Tp>>;

template <class _Alloc>
class __allocator_destructor
{
    typedef allocator_traits<_Alloc> __alloc_traits;
public:
    typedef typename __alloc_traits::pointer pointer;
    typedef typename __alloc_traits::size_type size_type;
private:
    _Alloc& __alloc_;
    size_t __s_;
public:
    _LIBCPP_INLINE_VISIBILITY __allocator_destructor(_Alloc& __a, size_t __s)
             _NOEXCEPT
        : __alloc_(__a), __s_(__s) {}
    _LIBCPP_INLINE_VISIBILITY
    void operator()(pointer __p) _NOEXCEPT
        {__alloc_traits::deallocate(__alloc_, __p, __s_);}
};


// Forward declarations
template<typename Trait>
class any;

template <class _ValueType, typename Trait>
_LIBCPP_INLINE_VISIBILITY
add_pointer_t<add_const_t<_ValueType>>
any_cast(any<Trait> const *) _NOEXCEPT;

template <class _ValueType, typename Trait>
_LIBCPP_INLINE_VISIBILITY
add_pointer_t<_ValueType> any_cast(any<Trait> *) _NOEXCEPT;

namespace __any_imp
{
  _LIBCPP_SUPPRESS_DEPRECATED_PUSH
  using _Buffer = aligned_storage_t<3*sizeof(void*), alignment_of<void*>::value>;
  _LIBCPP_SUPPRESS_DEPRECATED_POP

  template <class _Tp>
  using _IsSmallObject = integral_constant<bool
        , sizeof(_Tp) <= sizeof(_Buffer)
          && alignment_of<_Buffer>::value
             % alignment_of<_Tp>::value == 0
          && is_nothrow_move_constructible<_Tp>::value
        >;

  enum class _Action {
    _Destroy,
    _Copy,
    _Move,
    _Get,
    _TypeInfo,
    _Custom
  };

  template <class _Tp, typename Trait> struct _SmallHandler;
  template <class _Tp, typename Trait> struct _LargeHandler;

  template <class _Tp>
  struct  _LIBCPP_TEMPLATE_VIS __unique_typeinfo { static constexpr int __id = 0; };
  template <class _Tp> constexpr int __unique_typeinfo<_Tp>::__id;

  template <class _Tp>
  inline _LIBCPP_INLINE_VISIBILITY
  constexpr const void* __get_fallback_typeid() {
      return &__unique_typeinfo<remove_cv_t<remove_reference_t<_Tp>>>::__id;
  }

  template <class _Tp>
  inline _LIBCPP_INLINE_VISIBILITY
  bool __compare_typeid(type_info const* __id, const void* __fallback_id)
  {
#if !defined(_LIBCPP_HAS_NO_RTTI)
      if (__id && *__id == typeid(_Tp))
          return true;
#endif
      if (!__id && __fallback_id == __any_imp::__get_fallback_typeid<_Tp>())
          return true;
      return false;
  }

  template <class _Tp, typename Trait>
  using _Handler = conditional_t<
    _IsSmallObject<_Tp>::value, _SmallHandler<_Tp, Trait>, _LargeHandler<_Tp, Trait>>;

} // namespace __any_imp

template<typename Trait>
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
    , class _Tp = decay_t<_ValueType>
    , class = enable_if_t<
        !is_same<_Tp, any>::value &&
        !__is_inplace_type<_ValueType>::value &&
        is_copy_constructible<_Tp>::value>
    >
  _LIBCPP_INLINE_VISIBILITY
  any(_ValueType && __value);

  template <class _ValueType, class ..._Args,
    class _Tp = decay_t<_ValueType>,
    class = enable_if_t<
        is_constructible<_Tp, _Args...>::value &&
        is_copy_constructible<_Tp>::value
    >
  >
  _LIBCPP_INLINE_VISIBILITY
  explicit any(in_place_type_t<_ValueType>, _Args&&... __args);

  template <class _ValueType, class _Up, class ..._Args,
    class _Tp = decay_t<_ValueType>,
    class = enable_if_t<
        is_constructible<_Tp, initializer_list<_Up>&, _Args...>::value &&
        is_copy_constructible<_Tp>::value>
  >
  _LIBCPP_INLINE_VISIBILITY
  explicit any(in_place_type_t<_ValueType>, initializer_list<_Up>, _Args&&... __args);

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
    any(_VSTD::move(__rhs)).swap(*this);
    return *this;
  }

  template <
      class _ValueType
    , class _Tp = decay_t<_ValueType>
    , class = enable_if_t<
          !is_same<_Tp, any>::value
          && is_copy_constructible<_Tp>::value>
    >
  _LIBCPP_INLINE_VISIBILITY
  any & operator=(_ValueType && __rhs);

  template <class _ValueType, class ..._Args,
    class _Tp = decay_t<_ValueType>,
    class = enable_if_t<
        is_constructible<_Tp, _Args...>::value &&
        is_copy_constructible<_Tp>::value>
    >
  _LIBCPP_INLINE_VISIBILITY
  _Tp& emplace(_Args&&...);

  template <class _ValueType, class _Up, class ..._Args,
    class _Tp = decay_t<_ValueType>,
    class = enable_if_t<
        is_constructible<_Tp, initializer_list<_Up>&, _Args...>::value &&
        is_copy_constructible<_Tp>::value>
  >
  _LIBCPP_INLINE_VISIBILITY
  _Tp& emplace(initializer_list<_Up>, _Args&&...);

  // 6.3.3 any modifiers
  _LIBCPP_INLINE_VISIBILITY
  void reset() _NOEXCEPT { if (__h_) this->__call(_Action::_Destroy); }

  _LIBCPP_INLINE_VISIBILITY
  void swap(any & __rhs) _NOEXCEPT;

  // 6.3.4 any observers
  _LIBCPP_INLINE_VISIBILITY
  bool has_value() const _NOEXCEPT { return __h_ != nullptr; }

#if !defined(_LIBCPP_HAS_NO_RTTI)
  _LIBCPP_INLINE_VISIBILITY
  const type_info & type() const _NOEXCEPT {
    if (__h_) {
        return *static_cast<type_info const *>(this->__call(_Action::_TypeInfo));
    } else {
        return typeid(void);
    }
  }
#endif
  void *type_id() const _NOEXCEPT {
    return __call(_Action::_TypeInfo);
  }

protected:
  _LIBCPP_INLINE_VISIBILITY
  void *call(Trait::TraitAction Action, any * __other = nullptr) {
      return __h_(static_cast<_Action>(static_cast<size_t>(Action) + static_cast<size_t>(_Action::_Custom)), this, __other, nullptr, nullptr);
  }


private:
    typedef __any_imp::_Action _Action;
    using _HandleFuncPtr =  void* (*)(_Action, any const *, any *, const type_info *,
      const void* __fallback_info);

    union _Storage {
        constexpr _Storage() : __ptr(nullptr) {}
        void *  __ptr;
        __any_imp::_Buffer __buf;
    };

    _LIBCPP_INLINE_VISIBILITY
    void * __call(_Action __a, any * __other = nullptr,
                  type_info const * __info = nullptr,
                   const void* __fallback_info = nullptr) const
    {
        return __h_(__a, this, __other, __info, __fallback_info);
    }

    _LIBCPP_INLINE_VISIBILITY
    void * __call(_Action __a, any * __other = nullptr,
                  type_info const * __info = nullptr,
                  const void* __fallback_info = nullptr)
    {
        return __h_(__a, this, __other, __info, __fallback_info);
    }

    template <class, typename>
    friend struct __any_imp::_SmallHandler;
    template <class, typename>
    friend struct __any_imp::_LargeHandler;

    template <class _ValueType, typename T>
    friend add_pointer_t<add_const_t<_ValueType>>
    any_cast(any<T> const *) _NOEXCEPT;

    template <class _ValueType, typename T>
    friend add_pointer_t<_ValueType>
    any_cast(any<T> *) _NOEXCEPT;

    _HandleFuncPtr __h_ = nullptr;
    _Storage __s_;
};

namespace __any_imp
{
  template <class _Tp, typename Trait>
  struct _LIBCPP_TEMPLATE_VIS _SmallHandler
  {
     using any = any<Trait>;
     _LIBCPP_INLINE_VISIBILITY
     static void* __handle(_Action __act, any const * __this, any * __other,
                           type_info const * __info, const void* __fallback_info)
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
            return __get(const_cast<any &>(*__this), __info, __fallback_info);
        case _Action::_TypeInfo:
            return const_cast<void *>(__any_imp::__get_fallback_typeid<_Tp>());
        case _Action::_Custom:
        default:
            return reinterpret_cast<void *>(Trait::template handle<_Tp>(static_cast<Trait::TraitAction>(static_cast<size_t>(__act) - static_cast<size_t>(_Action::_Custom)),
                                      const_cast<any *>(__this),
                                      __other));
        }
        __libcpp_unreachable();
    }

    template <class ..._Args>
    _LIBCPP_INLINE_VISIBILITY
    static _Tp& __create(any & __dest, _Args&&... __args) {
        typedef allocator<_Tp> _Alloc;
        typedef allocator_traits<_Alloc> _ATraits;
        _Alloc __a;
        _Tp * __ret = static_cast<_Tp*>(static_cast<void*>(&__dest.__s_.__buf));
        _ATraits::construct(__a, __ret, _VSTD::forward<_Args>(__args)...);
        __dest.__h_ = &_SmallHandler::__handle;
        return *__ret;
    }

  private:
    _LIBCPP_INLINE_VISIBILITY
    static void __destroy(any & __this) {
        typedef allocator<_Tp> _Alloc;
        typedef allocator_traits<_Alloc> _ATraits;
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
        _SmallHandler::__create(__dest, _VSTD::move(
            *static_cast<_Tp*>(static_cast<void*>(&__this.__s_.__buf))));
        __destroy(__this);
    }

    _LIBCPP_INLINE_VISIBILITY
    static void* __get(any & __this,
                       type_info const * __info,
                       const void* __fallback_id)
    {
        if (__any_imp::__compare_typeid<_Tp>(__info, __fallback_id))
            return static_cast<void*>(&__this.__s_.__buf);
        return nullptr;
    }

    _LIBCPP_INLINE_VISIBILITY
    static void* __type_info()
    {
#if !defined(_LIBCPP_HAS_NO_RTTI)
        return const_cast<void*>(static_cast<void const *>(&typeid(_Tp)));
#else
        return nullptr;
#endif
    }
  };

  template <class _Tp, typename Trait>
  struct _LIBCPP_TEMPLATE_VIS _LargeHandler
  {
    using any = any<Trait>;

    _LIBCPP_INLINE_VISIBILITY
    static void* __handle(_Action __act, any const * __this,
                          any * __other, type_info const * __info,
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
          return __get(const_cast<any &>(*__this), __info, __fallback_info);
        case _Action::_TypeInfo:
          return const_cast<void *>(__any_imp::__get_fallback_typeid<_Tp>());
        case _Action::_Custom:
            __libcpp_unreachable();
        }
        __libcpp_unreachable();
    }

    template <class ..._Args>
    _LIBCPP_INLINE_VISIBILITY
    static _Tp& __create(any & __dest, _Args&&... __args) {
        typedef allocator<_Tp> _Alloc;
        typedef allocator_traits<_Alloc> _ATraits;
        typedef __allocator_destructor<_Alloc> _Dp;
        _Alloc __a;
        unique_ptr<_Tp, _Dp> __hold(_ATraits::allocate(__a, 1), _Dp(__a, 1));
        _Tp * __ret = __hold.get();
        _ATraits::construct(__a, __ret, _VSTD::forward<_Args>(__args)...);
        __dest.__s_.__ptr = __hold.release();
        __dest.__h_ = &_LargeHandler::__handle;
        return *__ret;
    }

  private:

    _LIBCPP_INLINE_VISIBILITY
    static void __destroy(any & __this){
        typedef allocator<_Tp> _Alloc;
        typedef allocator_traits<_Alloc> _ATraits;
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
    static void* __get(any & __this, type_info const * __info,
                       void const* __fallback_info)
    {
        if (__any_imp::__compare_typeid<_Tp>(__info, __fallback_info))
            return static_cast<void*>(__this.__s_.__ptr);
        return nullptr;

    }

  };

  _LIBCPP_INLINE_VISIBILITY
  static void* __type_info() __attribute__((used));

  _LIBCPP_INLINE_VISIBILITY
  static void* __type_info()
{
#if !defined(_LIBCPP_HAS_NO_RTTI)
        return const_cast<void*>(static_cast<void const *>(&typeid(_Tp)));
#else
        return nullptr;
#endif
    }
} // namespace __any_imp


template<typename Trait>
template <class _ValueType, class _Tp, class>
any<Trait>::any(_ValueType && __v) : __h_(nullptr)
{
  __any_imp::_Handler<_Tp, Trait>::__create(*this, _VSTD::forward<_ValueType>(__v));
}

template<typename Trait>
template <class _ValueType, class ..._Args, class _Tp, class>
any<Trait>::any(in_place_type_t<_ValueType>, _Args&&... __args) {
  __any_imp::_Handler<_Tp, Trait>::__create(*this, _VSTD::forward<_Args>(__args)...);
}

template<typename Trait>
template <class _ValueType, class _Up, class ..._Args, class _Tp, class>
any<Trait>::any(in_place_type_t<_ValueType>, initializer_list<_Up> __il, _Args&&... __args) {
  __any_imp::_Handler<_Tp, Trait>::__create(*this, __il, _VSTD::forward<_Args>(__args)...);
}

template<typename Trait>
template <class _ValueType, class, class>
inline _LIBCPP_INLINE_VISIBILITY
any<Trait> & any<Trait>::operator=(_ValueType && __v)
{
  any(_VSTD::forward<_ValueType>(__v)).swap(*this);
  return *this;
}

template<typename Trait>
template <class _ValueType, class ..._Args, class _Tp, class>
inline _LIBCPP_INLINE_VISIBILITY
_Tp& any<Trait>::emplace(_Args&&... __args) {
  reset();
  return __any_imp::_Handler<_Tp, Trait>::__create(*this, _VSTD::forward<_Args>(__args)...);
}

template<typename Trait>
template <class _ValueType, class _Up, class ..._Args, class _Tp, class>
inline _LIBCPP_INLINE_VISIBILITY
_Tp& any<Trait>::emplace(initializer_list<_Up> __il, _Args&&... __args) {
  reset();
  return __any_imp::_Handler<_Tp, Trait>::__create(*this, __il, _VSTD::forward<_Args>(__args)...);
}

template<typename Trait>
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

template<typename Trait>
inline _LIBCPP_INLINE_VISIBILITY
void swap(any<Trait> & __lhs, any<Trait> & __rhs) _NOEXCEPT
{
    __lhs.swap(__rhs);
}

template <class _Tp, typename Trait, class ..._Args>
inline _LIBCPP_INLINE_VISIBILITY
any<Trait> make_any(_Args&&... __args) {
    return any(in_place_type<_Tp>, _VSTD::forward<_Args>(__args)...);
}

template <class _Tp, typename Trait, class _Up, class ..._Args>
inline _LIBCPP_INLINE_VISIBILITY
any<Trait> make_any(initializer_list<_Up> __il, _Args&&... __args) {
    return any(in_place_type<_Tp>, __il, _VSTD::forward<_Args>(__args)...);
}

template <class _ValueType, typename Trait>
inline _LIBCPP_INLINE_VISIBILITY
_LIBCPP_AVAILABILITY_THROW_BAD_ANY_CAST
_ValueType any_cast(any<Trait> const & __v)
{
    using _RawValueType = __remove_cvref_t<_ValueType>;
    static_assert(is_constructible<_ValueType, _RawValueType const &>::value,
                  "ValueType is required to be a const lvalue reference "
                  "or a CopyConstructible type");
    auto __tmp = any_cast<add_const_t<_RawValueType>>(&__v);
    if (__tmp == nullptr)
        __throw_bad_any_cast();
    return static_cast<_ValueType>(*__tmp);
}

template <class _ValueType, typename Trait>
inline _LIBCPP_INLINE_VISIBILITY
_LIBCPP_AVAILABILITY_THROW_BAD_ANY_CAST
_ValueType any_cast(any<Trait> & __v)
{
    using _RawValueType = __remove_cvref_t<_ValueType>;
    static_assert(is_constructible<_ValueType, _RawValueType &>::value,
                  "ValueType is required to be an lvalue reference "
                  "or a CopyConstructible type");
    auto __tmp = any_cast<_RawValueType>(&__v);
    if (__tmp == nullptr)
        __throw_bad_any_cast();
    return static_cast<_ValueType>(*__tmp);
}

template <class _ValueType, typename Trait>
inline _LIBCPP_INLINE_VISIBILITY
_LIBCPP_AVAILABILITY_THROW_BAD_ANY_CAST
_ValueType any_cast(any<Trait> && __v)
{
    using _RawValueType = __remove_cvref_t<_ValueType>;
    static_assert(is_constructible<_ValueType, _RawValueType>::value,
                  "ValueType is required to be an rvalue reference "
                  "or a CopyConstructible type");
    auto __tmp = any_cast<_RawValueType>(&__v);
    if (__tmp == nullptr)
        __throw_bad_any_cast();
    return static_cast<_ValueType>(_VSTD::move(*__tmp));
}

template <class _ValueType, typename Trait>
inline _LIBCPP_INLINE_VISIBILITY
add_pointer_t<add_const_t<_ValueType>>
any_cast(any<Trait> const * __any) _NOEXCEPT
{
    static_assert(!is_reference<_ValueType>::value,
                  "_ValueType may not be a reference.");
    return any_cast<_ValueType>(const_cast<any<Trait> *>(__any));
}

template <class _RetType>
inline _LIBCPP_INLINE_VISIBILITY
_RetType __pointer_or_func_cast(void* __p, /*IsFunction*/false_type) noexcept {
  return static_cast<_RetType>(__p);
}

template <class _RetType>
inline _LIBCPP_INLINE_VISIBILITY
_RetType __pointer_or_func_cast(void*, /*IsFunction*/true_type) noexcept {
  return nullptr;
}

template <class _ValueType, typename Trait>
_LIBCPP_HIDE_FROM_ABI
add_pointer_t<_ValueType>
any_cast(any<Trait> * __any) _NOEXCEPT
{
    using __any_imp::_Action;
    static_assert(!is_reference<_ValueType>::value,
                  "_ValueType may not be a reference.");
    typedef add_pointer_t<_ValueType> _ReturnType;
    if (__any && __any->__h_) {
      void *__p = __any->__call(_Action::_Get, nullptr,
#if !defined(_LIBCPP_HAS_NO_RTTI)
                          &typeid(_ValueType),
#else
                          nullptr,
#endif
                          __any_imp::__get_fallback_typeid<_ValueType>());
        return revng::__pointer_or_func_cast<_ReturnType>(
            __p, is_function<_ValueType>{});
    }
    return nullptr;
}

template<typename Trait>
using TraitfulAny = any<Trait>;

#pragma pop_macro("_VSTD")
#undef __remove_cvref_t
#undef __throw_bad_any_cast
#undef __libcpp_unreachable

} // namespace revng
