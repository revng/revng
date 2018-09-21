/*
 * This file is distributed under the MIT License. See LICENSE.md for details.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>

int root(char *buffer, size_t size) {
  int result;

#ifdef TARGET_mips
  __asm__("li $v0,0xfb8\n\tsyscall\n\tmove %0, $v0" : "=r"(result) : : "%v0");
#else
  result = syscall(SYS_getuid);
#endif

  return result;
}

int main(int argc, char *argv[]) {
  printf("%d\n", root(argv[1], strlen(argv[1])));
  return EXIT_SUCCESS;
}
