;
; This file is distributed under the MIT License. See LICENSE.md for details.
;

; RUN: %root/bin/revng opt %s -split-struct-phis -verify -S | FileCheck %s
; Test that a predecessor reaching the phi over several edges, such as a switch
; with more than one case targeting the same block, gets one incoming value per
; predecessor instead of one per edge, which would produce an invalid phi.

%struct = type <{ i64, i64 }>

declare %struct @producer()

define i64 @f(i64 %x) {
entry:
  %s = call %struct @producer()
  switch i64 %x, label %join [
    i64 0, label %join
    i64 1, label %join
  ]

join:
  %p = phi %struct [ %s, %entry ], [ %s, %entry ], [ %s, %entry ]
  %r = extractvalue %struct %p, 0
  ret i64 %r
}

; A single extractvalue per field is materialized in the predecessor.
; CHECK:      [[F0:%[0-9]+]] = call i64 @OpaqueExtractvalue(%struct %s, i64 0)
; CHECK-NEXT: [[F1:%[0-9]+]] = call i64 @OpaqueExtractvalue(%struct %s, i64 1)
; CHECK-NOT:  call i64 @OpaqueExtractvalue

; Every entry for the repeated predecessor carries the same value.
; CHECK:      phi i64 [ [[F0]], %entry ], [ [[F0]], %entry ], [ [[F0]], %entry ]
; CHECK-NEXT: phi i64 [ [[F1]], %entry ], [ [[F1]], %entry ], [ [[F1]], %entry ]
