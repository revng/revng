#include <stdlib.h>
#include <stdio.h>
#include <string.h>

int root(char *buffer, size_t size) {
  union { float f; int i; } number;
  number.i = (127 + size) << 23;
  return (int) (number.f * number.f);
}

int main(int argc, char *argv[]) {
  printf("%d\n", root(argv[1], strlen(argv[1])));
  return EXIT_SUCCESS;
}
