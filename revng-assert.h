#ifndef REVNG_ASSERT_H
#define REVNG_ASSERT_H

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#ifdef __cplusplus
extern "C" {
#endif

// Standard includes
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef __has_builtin
#define __has_builtin(x) 0
#endif

#ifndef __has_attribute
#define __has_attribute(x) 0
#endif

#if !__has_builtin(__builtin_assume)
#define __builtin_assume(x)
#endif

#if !__has_builtin(__builtin_unreachable) && !defined(__GNUC__)
#define __builtin_unreachable(x)
#endif

#if defined(__clang__)

#define SILENCE_ASSUME_HEADER                                          \
  _Pragma("clang diagnostic push") _Pragma("clang diagnostic ignored " \
                                           "\"-Wassume\"")
#define SILENCE_ASSUME_FOOTER _Pragma("clang diagnostic pop")

#else

#define SILENCE_ASSUME_HEADER
#define SILENCE_ASSUME_FOOTER

#endif

/// \brief Same as __builtin_assume but (temporarily) suppresses warnings about
///        ignored side-effects
#define silent_assume(what) \
  do {                      \
    SILENCE_ASSUME_HEADER   \
    __builtin_assume(what); \
    SILENCE_ASSUME_FOOTER   \
  } while (0)

// We support C++11, C99 (with GNU attributes) or C11
#ifdef __cplusplus

#define boolcast(what) static_cast<bool>(what)
#define noret [[noreturn]]

#else

#define boolcast(what) (bool) (what)

#if __STDC_VERSION__ >= 201112L
#include <stdnoreturn.h>
#define noret noreturn
#elif __has_attribute(noreturn) || defined(__GNUC__)
#define noret __attribute__((noreturn))
#else
#warning "Can't mark functions as noreturn"
#endif

#endif

noret void revng_assert_fail(const char *AssertionBody,
                             const char *Message,
                             const char *File,
                             unsigned Line);
noret void revng_check_fail(const char *CheckBody,
                            const char *Message,
                            const char *File,
                            unsigned Line);
noret void revng_do_abort(const char *Message, const char *File, unsigned Line);

#undef noret

/// \brief Aborts program execution with a message, in release mode too.
///
/// Use this macro to ensure program termination in case of an unexpected
/// situation.
#define revng_abort(message)                     \
  do {                                           \
    revng_do_abort(message, __FILE__, __LINE__); \
  } while (0)

/// \brief Asserts \a what or aborts with \a message, in release mode too.
///
/// Use this macro to ensure program termination in case of an unexpected
/// situation.
#define revng_check(what, message)                          \
  do {                                                      \
    bool Condition = boolcast(what);                        \
    if (!Condition) {                                       \
      revng_check_fail(#what, message, __FILE__, __LINE__); \
    }                                                       \
    silent_assume(Condition);                               \
  } while (0)

#ifndef NDEBUG

/// \brief Marks a program path as unreachable. In debug mode, aborts with \a
///        message.
///
/// Use this macro to catch bugs during development an tell the compiler that a
/// certain situation will never happen at run-time, which might open up to new
/// optimization opportunities.
#define revng_unreachable(message) revng_abort(message)

/// \brief Asserts \a what or, in debug mode, aborts with \a message.
///
/// Use this macro to catch bugs during development an tell the compiler that a
/// certain situation will never happen at run-time, which might open up to new
/// optimization opportunities.
#define revng_assert(what, message)                          \
  do {                                                       \
    bool Condition = boolcast(what);                         \
    if (!Condition) {                                        \
      revng_assert_fail(#what, message, __FILE__, __LINE__); \
    }                                                        \
    silent_assume(Condition);                                \
  } while (0)

#else

#define revng_unreachable(message) __builtin_unreachable()

#define revng_assert(what, message) \
  do {                              \
    (void) sizeof((what));          \
    silent_assume(what);            \
  } while (0)

#endif

#ifdef __cplusplus
}
#endif

#endif // REVNG_ASSERT_H
