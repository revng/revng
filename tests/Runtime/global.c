/*
 * This file is distributed under the MIT License. See LICENSE.md for details.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>

int global = 1;
const int const_global = 2;

int root(char *buffer, size_t size) {
  global++;

  return global + const_global;
}

int main(int argc, char *argv[]) {
  printf("%d\n", root(argv[1], strlen(argv[1])));
  return EXIT_SUCCESS;
}
