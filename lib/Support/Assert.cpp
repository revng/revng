/// \file Assert.cpp
/// Implementation of the various functions to assert and abort.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <cassert>
#include <iostream>

#include "revng/Support/Assert.h"

[[noreturn]] static void terminate(void) {
  abort();
}

static void
report(const char *Type, const char *File, unsigned Line, const char *What) {
  fprintf(stderr, "%s at %s:%d", Type, File, Line);
  if (What != nullptr)
    fprintf(stderr, ":\n\n%s", What);
  fprintf(stderr, "\n");
}

void revng_assert_fail(const char *AssertionBody,
                       const char *Message,
                       const char *File,
                       unsigned Line) {
  report("Assertion failed", File, Line, Message);
  fprintf(stderr, "\n%s\n", AssertionBody);
  terminate();
}

void revng_check_fail(const char *CheckBody,
                      const char *Message,
                      const char *File,
                      unsigned Line) {
  report("Check failed", File, Line, Message);
  fprintf(stderr, "\n%s\n", CheckBody);
  terminate();
}

void revng_do_abort(const char *Message, const char *File, unsigned Line) {
  report("Abort", File, Line, Message);
  terminate();
}
