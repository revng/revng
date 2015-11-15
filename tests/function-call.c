#include <stdlib.h>
#include <stdio.h>
#include <string.h>

int half(int parameter) {
  return parameter / 2;
}

int root(char *buffer, size_t size) {
  int result = 412;
  result += half(result);
  result += 81;
  return result;
}

int main(int argc, char *argv[]) {
  printf("%d\n", root(argv[1], strlen(argv[1])));
  return EXIT_SUCCESS;
}
