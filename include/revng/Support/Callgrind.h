#ifndef VALGRINDHELPERS_H
#define VALGRINDHELPERS_H

#ifdef HAVE_VALGRIND_CALLGRIND_H

#include "valgrind/callgrind.h"

#else

#define CALLGRIND_START_INSTRUMENTATION \
  do {                                  \
  } while (0)
#define CALLGRIND_STOP_INSTRUMENTATION \
  do {                                 \
  } while (0)

#endif

class Callgrind {
public:
  Callgrind(bool Enable) : Enabled(Enable) {
    if (Enabled) {
      CALLGRIND_START_INSTRUMENTATION;
    } else {
      CALLGRIND_STOP_INSTRUMENTATION;
    }
  }

  ~Callgrind() {
    if (Enabled) {
      CALLGRIND_STOP_INSTRUMENTATION;
    } else {
      CALLGRIND_START_INSTRUMENTATION;
    }
  }

private:
  bool Enabled;
};

#endif // VALGRINDHELPERS_H
