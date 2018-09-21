/*
 * This file is distributed under the MIT License. See LICENSE.md for details.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FAILURE 666
#define MAX_OPERATIONS 10
#define MAX_ARGUMENTS 3

typedef int value_t;

typedef enum { none, positive, negative } literal_status;

typedef struct {
  uint8_t operator;
  uint16_t argument_count;
  uint16_t current_argument;
  value_t arguments[MAX_ARGUMENTS];
} operation_t;

int root(char *buffer, size_t size) {
  int depth = -1;
  int position = 0;
  value_t current_literal = 0;
  literal_status literal = none;

  operation_t stack[MAX_OPERATIONS];

  while (position < size) {
    if (buffer[position] == '(') {
      if (literal != none)
        return FAILURE;
      current_literal = 0;

      if (++depth == MAX_OPERATIONS)
        return FAILURE;

      // bzero
      for (int i = 0; i < sizeof(operation_t); i++)
        ((char *) &stack[depth])[i] = 0;

      // Move to operator
      if (++position == size)
        return FAILURE;

      // Check the operator is supported
      if (buffer[position] == '+' || buffer[position] == '-' ||
          /* buffer[position] == '%' || */
          /* buffer[position] == '/' || */
          buffer[position] == '*' || buffer[position] == '&'
          || buffer[position] == '|' || buffer[position] == '^') {
        stack[depth].argument_count = 2;
      } else if (buffer[position] == '~' || buffer[position] == '!') {
        stack[depth].argument_count = 1;
      } else if (buffer[position] == '?') {
        stack[depth].argument_count = 3;
      } else {
        // Unsupported operator
        return FAILURE;
      }

      stack[depth].operator= buffer[position];
      stack[depth].current_argument = 0;

      // Move to space
      if (++position == size || buffer[position] != ' ')
        return FAILURE;

    } else if (buffer[position] == '-') {
      if (literal != none)
        return FAILURE;

      literal = negative;

    } else if (buffer[position] >= '0' && buffer[position] <= '9') {
      // We got a number literal
      current_literal = current_literal * 10 + (buffer[position] - '0');

      if (literal == negative)
        current_literal = -current_literal;

      literal = positive;
    } else if (buffer[position] == ' ') {
      if (depth == -1)
        return FAILURE;

      stack[depth].arguments[stack[depth].current_argument] = current_literal;
      literal = none;
      current_literal = 0;

      // Check we're not exceeding the arguments we're supposed to have
      if (++stack[depth].current_argument == stack[depth].argument_count)
        return FAILURE;
    } else if (buffer[position] == ')') {
      if (depth == -1)
        return FAILURE;

      stack[depth].arguments[stack[depth].current_argument] = current_literal;
      literal = none;
      current_literal = 0;

      // Ensure all the required arguments have been provided
      if (stack[depth].current_argument + 1 != stack[depth].argument_count)
        return FAILURE;

      // Operation implementation
      if (stack[depth].operator== '+') {
        current_literal = stack[depth].arguments[0] + stack[depth].arguments[1];
      } else if (stack[depth].operator== '-') {
        current_literal = stack[depth].arguments[0] - stack[depth].arguments[1];
      } else if (stack[depth].operator== '*') {
        current_literal = stack[depth].arguments[0] * stack[depth].arguments[1];
      } else if (stack[depth].operator== '&') {
        current_literal = stack[depth].arguments[0] & stack[depth].arguments[1];
      } else if (stack[depth].operator== '|') {
        current_literal = stack[depth].arguments[0] | stack[depth].arguments[1];
      } else if (stack[depth].operator== '^') {
        current_literal = stack[depth].arguments[0] ^ stack[depth].arguments[1];
      } else if (stack[depth].operator== '?') {
        current_literal = stack[depth].arguments[0] ?
                            stack[depth].arguments[1] :
                            stack[depth].arguments[2];
      } else if (stack[depth].operator== '~') {
        current_literal = ~stack[depth].arguments[0];
      } else if (stack[depth].operator== '!') {
        current_literal = !stack[depth].arguments[0];
      } else {
        return FAILURE;
      }

      depth--;
      if (depth < -1)
        return FAILURE;
    }

    position++;
  }

  return current_literal;
}

int main(int argc, char *argv[]) {
  printf("%d\n", root(argv[1], strlen(argv[1])));
  return EXIT_SUCCESS;
}
