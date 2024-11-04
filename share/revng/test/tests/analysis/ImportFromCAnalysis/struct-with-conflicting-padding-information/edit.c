//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

struct _PACKED _SIZE(24) my_struct {
  _START_AT(16)
  uint8_t padding_at_0[8];
  uint64_t normal_field;
};
