//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt %s --optimize-expressions | FileCheck %s

!void = !clift.void

!int16_t = !clift.int<signed 2>
!uint16_t = !clift.int<unsigned 2>
!generic16_t = !clift.int<generic 2>

!int32_t = !clift.int<signed 4>
!uint32_t = !clift.int<unsigned 4>
!generic32_t = !clift.int<generic 4>

!int64_t = !clift.int<signed 8>
!uint64_t = !clift.int<unsigned 8>
!generic64_t = !clift.int<generic 8>

!f = !clift.func<"/model-type/1001" : !void()>

module attributes {clift.module} {
  clift.func @f<!f>() -> !void {
    // CHECK: %0 = clift.local : !uint16_t
    %0 = clift.local : !uint16_t

    // CHECK: %1 = clift.local : !uint32_t
    %1 = clift.local : !uint32_t

    // CHECK: %2 = clift.local : !uint64_t
    %2 = clift.local : !uint64_t

    // bitcast(bitcast(x)) -> bitcast(x)
    // CHECK: clift.expr {
    clift.expr {
      // CHECK: %3 = clift.bitcast %2 : !uint64_t -> !int64_t
      %3 = clift.bitcast %2 : !uint64_t -> !generic64_t
      %4 = clift.bitcast %3 : !generic64_t -> !int64_t
      // CHECK: clift.yield %3 : !int64_t
      clift.yield %4 : !int64_t
    }
    // CHECK: }

    // bitcast(zext(x)) -> zext(x)
    // CHECK: clift.expr {
    clift.expr {
      // CHECK: %3 = clift.zext %1 : !uint32_t -> !int64_t
      %3 = clift.zext %1 : !uint32_t -> !generic64_t
      %4 = clift.bitcast %3 : !generic64_t -> !int64_t
      // CHECK: clift.yield %3 : !int64_t
      clift.yield %4 : !int64_t
    }
    // CHECK: }

    // bitcast(truncate(x)) -> truncate(x)
    // CHECK: clift.expr {
    clift.expr {
      // CHECK: %3 = clift.truncate %2 : !uint64_t -> !int32_t
      %3 = clift.truncate %2 : !uint64_t -> !generic32_t
      %4 = clift.bitcast %3 : !generic32_t -> !int32_t
      // CHECK: clift.yield %3 : !int32_t
      clift.yield %4 : !int32_t
    }
    // CHECK: }

    // zext(bitcast(x)) -> zext(x)
    // CHECK: clift.expr {
    clift.expr {
      // CHECK: %3 = clift.zext %1 : !uint32_t -> !int64_t
      %3 = clift.bitcast %1 : !uint32_t -> !generic32_t
      %4 = clift.zext %3 : !generic32_t -> !int64_t
      // CHECK: clift.yield %3 : !int64_t
      clift.yield %4 : !int64_t
    }
    // CHECK: }

    // sext(bitcast(x))
    // CHECK: clift.expr {
    clift.expr {
      // CHECK: %3 = clift.sext %1 : !uint32_t -> !int64_t
      %3 = clift.bitcast %1 : !uint32_t -> !int32_t
      %4 = clift.sext %3 : !int32_t -> !int64_t
      // CHECK: clift.yield %3 : !int64_t
      clift.yield %4 : !int64_t
    }
    // CHECK: }

    // zext(zext(x)) -> zext(x)
    // CHECK: clift.expr {
    clift.expr {
      // CHECK: %3 = clift.zext %0 : !uint16_t -> !int64_t
      %3 = clift.zext %0 : !uint16_t -> !uint32_t
      %4 = clift.zext %3 : !uint32_t -> !int64_t
      // CHECK: clift.yield %3 : !int64_t
      clift.yield %4 : !int64_t
    }
    // CHECK: }

    // sext(zext(x)) -> zext(x)
    // CHECK: clift.expr {
    clift.expr {
      // CHECK: %3 = clift.zext %0 : !uint16_t -> !int64_t
      %3 = clift.zext %0 : !uint16_t -> !int32_t
      %4 = clift.sext %3 : !int32_t -> !int64_t
      // CHECK: clift.yield %3 : !int64_t
      clift.yield %4 : !int64_t
    }
    // CHECK: }

    // zext(truncate(x)) -> zext(truncate(x))
    // CHECK: clift.expr {
    clift.expr {
      // CHECK: %3 = clift.truncate %1 : !uint32_t -> !uint16_t
      %3 = clift.truncate %1 : !uint32_t -> !uint16_t
      // CHECK: %4 = clift.zext %3 : !uint16_t -> !uint32_t
      %4 = clift.zext %3 : !uint16_t -> !uint32_t
      // CHECK: clift.yield %4 : !uint32_t
      clift.yield %4 : !uint32_t
    }
    // CHECK: }

    // sext(truncate(x)) -> sext(truncate(x))
    // CHECK: clift.expr {
    clift.expr {
      // CHECK: %3 = clift.truncate %1 : !uint32_t -> !uint16_t
      %3 = clift.truncate %1 : !uint32_t -> !uint16_t
      // CHECK: %4 = clift.sext %3 : !uint16_t -> !uint32_t
      %4 = clift.sext %3 : !uint16_t -> !uint32_t
      // CHECK: clift.yield %4 : !uint32_t
      clift.yield %4 : !uint32_t
    }
    // CHECK: }

    // zext(sext(x)) -> zext(sext(x))
    // CHECK: clift.expr {
    clift.expr {
      // CHECK: %3 = clift.sext %0 : !uint16_t -> !uint32_t
      %3 = clift.sext %0 : !uint16_t -> !uint32_t
      // CHECK: %4 = clift.zext %3 : !uint32_t -> !uint64_t
      %4 = clift.zext %3 : !uint32_t -> !uint64_t
      // CHECK: clift.yield %4 : !uint64_t
      clift.yield %4 : !uint64_t
    }
    // CHECK: }

    // truncate(bitcast(x)) -> truncate(x)
    // CHECK: clift.expr {
    clift.expr {
      // CHECK: %3 = clift.truncate %2 : !uint64_t -> !uint32_t
      %3 = clift.bitcast %2 : !uint64_t -> !int64_t
      %4 = clift.truncate %3 : !int64_t -> !uint32_t
      // CHECK: clift.yield %3 : !uint32_t
      clift.yield %4 : !uint32_t
    }
    // CHECK: }

    // truncate(zext(x)) -> bitcast(x)
    // CHECK: clift.expr {
    clift.expr {
      // CHECK: %3 = clift.bitcast %1 : !uint32_t -> !int32_t
      %3 = clift.zext %1 : !uint32_t -> !int64_t
      %4 = clift.truncate %3 : !int64_t -> !int32_t
      // CHECK: clift.yield %3 : !int32_t
      clift.yield %4 : !int32_t
    }
    // CHECK: }

    // truncate(zext(x)) -> truncate(x)
    // CHECK: clift.expr {
    clift.expr {
      // CHECK: %3 = clift.zext %0 : !uint16_t -> !int32_t
      %3 = clift.zext %0 : !uint16_t -> !int64_t
      %4 = clift.truncate %3 : !int64_t -> !int32_t
      // CHECK: clift.yield %3 : !int32_t
      clift.yield %4 : !int32_t
    }
    // CHECK: }

    // truncate(zext(x)) -> truncate(x)
    // CHECK: clift.expr {
    clift.expr {
      // CHECK: %3 = clift.truncate %1 : !uint32_t -> !int16_t
      %3 = clift.zext %1 : !uint32_t -> !int64_t
      %4 = clift.truncate %3 : !int64_t -> !int16_t
      // CHECK: clift.yield %3 : !int16_t
      clift.yield %4 : !int16_t
    }
    // CHECK: }

    // truncate(truncate(x)) -> truncate(x)
    // CHECK: clift.expr {
    clift.expr {
      // CHECK: %3 = clift.truncate %2 : !uint64_t -> !uint16_t
      %3 = clift.truncate %2 : !uint64_t -> !int32_t
      %4 = clift.truncate %3 : !int32_t -> !uint16_t
      // CHECK: clift.yield %3 : !uint16_t
      clift.yield %4 : !uint16_t
    }
    // CHECK: }
  }
}
