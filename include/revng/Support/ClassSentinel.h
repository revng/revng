#ifndef CLASSSENTINEL_H
#define CLASSSENTINEL_H

#ifdef SENTINEL_STACKTRACES
extern "C" {
#include <execinfo.h>
}
#endif

// Local libraries includes
#include "revng/Support/Assert.h"
#include "revng/Support/Debug.h"

#ifdef SENTINEL_STACKTRACES
class StackTrace {
public:
  void take() { Count = backtrace(Buffer, BufferCapacity); }

  void print() const {
    char **Strings = backtrace_symbols(Buffer, Count);
    revng_assert(Strings != nullptr);

    std::string Result;
    for (int J = 0; J < Count; J++)
      Result += std::string(Strings[J]) + "\n";

    free(Strings);

    // It would be nice to be able to use the nice stack traces that LLVM
    // provides, but the PrintStackTrace method is private.
    //
    // llvm::raw_string_ostream Stream(Result);
    // llvm::sys::printSymbolizedStackTrace(Buffer, Count, Stream);

    dbg << Result << "\n";
  }

private:
  static const size_t BufferCapacity = 10;
  int Count;
  void *Buffer[BufferCapacity];
};
#endif

/// \brief Helper class to detect object used after being freed or moved
///
/// To use the ClassSentinel Simply put an instance of the ClassSentinel in
/// your class, it will be destroyed and moved with your parent object. Then
/// add some class to check where you think something wrong might be going on.
/// The check methods checks if the current object has been destroyed or moved.
///
/// The ClassSentinel can also collect backtraces on destruction/move, that will
/// be printed in case a call to check fails. This makes it easy to detect the
/// source of the problem, but creates a sensible overhead.
/// To enable this feature, define SENTINEL_STACKTRACES.
///
/// \note Using a moved object is not always bad. For instance, asking the size
///       of a moved vector might be perfectly fine. However, in many cases,
///       using a moved object is unwanted.
///
/// \note While detecting if the current object has been moved is easy and
///       correct, the same is not true for destroyed objects. In fact, this
///       class, in its destructor, writes a field. However this field might be
///       clobbered by another object using the same memory area where the
///       sentil was allocated, possibly creating false negatives.
class ClassSentinel {
public:
  ClassSentinel() : Moved(false), Destroyed(false) {}
  ClassSentinel(const ClassSentinel &) = default;
  ClassSentinel &operator=(const ClassSentinel &) = default;
  ClassSentinel(ClassSentinel &&Other) : Moved(false), Destroyed(false) {
    Other.check();
    Other.Moved = true;
#ifdef SENTINEL_STACKTRACES
    Other.MoveBacktrace.take();
#endif
  }

  ClassSentinel &operator=(ClassSentinel &&Other) {
    checkDestroy();
    Other.check();
    Other.Moved = true;
#ifdef SENTINEL_STACKTRACES
    Other.MoveBacktrace.take();
#endif
    Moved = false;
    Destroyed = false;
    return *this;
  }

  ~ClassSentinel() {
    Destroyed = true;
#ifdef SENTINEL_STACKTRACES
    DestroyBacktrace.take();
#endif
  }

  void checkDestroy() const {
    if (Destroyed) {
      dbg << "Trying to use a destroyed object.\n";
#ifdef SENTINEL_STACKTRACES
      dbg << "It was moved with the following backtrace:\n\n";
      DestroyBacktrace.print();
#endif
      revng_abort();
    }
  }

  void check() const {
    checkDestroy();
    if (Moved) {
      dbg << "Trying to use a moved object.\n";
#ifdef SENTINEL_STACKTRACES
      dbg << "It was moved with the following backtrace:\n\n";
      MoveBacktrace.print();
#endif
      revng_abort();
    }
  }

  bool isMoved() const { return Moved; }
  bool isDestroyed() const { return Destroyed; }

private:
  bool Moved;
  bool Destroyed;
#ifdef SENTINEL_STACKTRACES
  StackTrace DestroyBacktrace;
  StackTrace MoveBacktrace;
#endif
};

#endif // CLASSSENTINEL_H
