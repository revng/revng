//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

struct _PACKED _SIZE(24) my_struct {
  _START_AT(16)
  uint64_t normal_field;

  _START_AT(1024)
  uint64_t broken_field;
};
