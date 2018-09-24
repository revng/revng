#ifndef VALGRINDHELPERS_H
#define VALGRINDHELPERS_H

#include "valgrind/callgrind.h"

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
