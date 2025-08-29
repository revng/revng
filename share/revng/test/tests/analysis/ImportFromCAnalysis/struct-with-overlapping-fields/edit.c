//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

struct _PACKED my_struct {
  uint8_t padding_at_0[8];
  uint64_t *normal_field;

  _STARTS_AT(10)
  uint64_t *overlapping_field;
};
