;
; This file is distributed under the MIT License. See LICENSE.md for details.
;

; RUN: %revngopt %s -S -o - | FileCheck %s
; CHECK: define i1 @f
define i1 @f () {
  ret i1 true
}
