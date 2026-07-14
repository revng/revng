//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

struct _PACKED my_struct {
  // This statement will be ignored because it conflicts with the next one
  uint8_t padding_at_0[64];

  _STARTS_AT(16)
  uint64_t *field;
};
