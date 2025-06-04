//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngpipe emit-c %S/model.yml %s <(tar -czT /dev/null) /dev/stdout | tar -zxO

!void = !clift.primitive<void 0>
!int32_t = !clift.primitive<signed 4>
!int32_t$p = !clift.ptr<8 to !int32_t>

!f = !clift.func<
  "/type-definition/1001-CABIFunctionDefinition" : !void()
>

module attributes {clift.module} {
  // CHECK: void fun_0x40001001(void) {
  clift.func @f<!f>() attributes {
    handle = "/function/0x40001001:Code_x86_64"
  } {
    // CHECK: int32_t _var_0;
    %x = clift.local !int32_t

    // CHECK: int32_t *_var_1;
    %p = clift.local !int32_t$p

    // CHECK: -_var_0;
    clift.expr {
      %r = clift.neg %x : !int32_t
      clift.yield %r : !int32_t
    }

    // CHECK: ~_var_0;
    clift.expr {
      %r = clift.bitnot %x : !int32_t
      clift.yield %r : !int32_t
    }

    // CHECK: !_var_0;
    clift.expr {
      %r = clift.not %x : !int32_t -> !int32_t
      clift.yield %r : !int32_t
    }

    // CHECK: ++_var_0;
    clift.expr {
      %r = clift.inc %x : !int32_t
      clift.yield %r : !int32_t
    }

    // CHECK: --_var_0;
    clift.expr {
      %r = clift.dec %x : !int32_t
      clift.yield %r : !int32_t
    }

    // CHECK: _var_0++;
    clift.expr {
      %r = clift.post_inc %x : !int32_t
      clift.yield %r : !int32_t
    }

    // CHECK: _var_0--;
    clift.expr {
      %r = clift.post_dec %x : !int32_t
      clift.yield %r : !int32_t
    }

    // CHECK: &_var_0;
    clift.expr {
      %r = clift.addressof %x : !int32_t$p
      clift.yield %r : !int32_t$p
    }

    // CHECK: *_var_1;
    clift.expr {
      %r = clift.indirection %p : !int32_t$p
      clift.yield %r : !int32_t
    }

    // CHECK: _var_0 + _var_0;
    clift.expr {
      %r = clift.add %x, %x : !int32_t
      clift.yield %r : !int32_t
    }

    // CHECK: _var_0 - _var_0;
    clift.expr {
      %r = clift.sub %x, %x : !int32_t
      clift.yield %r : !int32_t
    }

    // CHECK: _var_0 * _var_0;
    clift.expr {
      %r = clift.mul %x, %x : !int32_t
      clift.yield %r : !int32_t
    }

    // CHECK: _var_0 / _var_0;
    clift.expr {
      %r = clift.div %x, %x : !int32_t
      clift.yield %r : !int32_t
    }

    // CHECK: _var_0 % _var_0;
    clift.expr {
      %r = clift.rem %x, %x : !int32_t
      clift.yield %r : !int32_t
    }

    // CHECK: _var_0 && _var_0;
    clift.expr {
      %r = clift.and %x, %x : !int32_t -> !int32_t
      clift.yield %r : !int32_t
    }

    // CHECK: _var_0 || _var_0;
    clift.expr {
      %r = clift.or %x, %x : !int32_t -> !int32_t
      clift.yield %r : !int32_t
    }

    // CHECK: _var_0 & _var_0;
    clift.expr {
      %r = clift.bitand %x, %x : !int32_t
      clift.yield %r : !int32_t
    }

    // CHECK: _var_0 | _var_0;
    clift.expr {
      %r = clift.bitor %x, %x : !int32_t
      clift.yield %r : !int32_t
    }

    // CHECK: _var_0 ^ _var_0;
    clift.expr {
      %r = clift.bitxor %x, %x : !int32_t
      clift.yield %r : !int32_t
    }

    // CHECK: _var_0 << _var_0;
    clift.expr {
      %r = clift.shl %x, %x : !int32_t
      clift.yield %r : !int32_t
    }

    // CHECK: _var_0 >> _var_0;
    clift.expr {
      %r = clift.shr %x, %x : !int32_t
      clift.yield %r : !int32_t
    }

    // CHECK: _var_0 == _var_0;
    clift.expr {
      %r = clift.eq %x, %x : !int32_t -> !int32_t
      clift.yield %r : !int32_t
    }

    // CHECK: _var_0 != _var_0;
    clift.expr {
      %r = clift.ne %x, %x : !int32_t -> !int32_t
      clift.yield %r : !int32_t
    }

    // CHECK: _var_0 < _var_0;
    clift.expr {
      %r = clift.lt %x, %x : !int32_t -> !int32_t
      clift.yield %r : !int32_t
    }

    // CHECK: _var_0 > _var_0;
    clift.expr {
      %r = clift.gt %x, %x : !int32_t -> !int32_t
      clift.yield %r : !int32_t
    }

    // CHECK: _var_0 <= _var_0;
    clift.expr {
      %r = clift.le %x, %x : !int32_t -> !int32_t
      clift.yield %r : !int32_t
    }

    // CHECK: _var_0 >= _var_0;
    clift.expr {
      %r = clift.ge %x, %x : !int32_t -> !int32_t
      clift.yield %r : !int32_t
    }

    // CHECK: _var_0, _var_0;
    clift.expr {
      %r = clift.comma %x, %x : !int32_t, !int32_t
      clift.yield %r : !int32_t
    }
  }
  // CHECK: }
}
