;
; This file is distributed under the MIT License. See LICENSE.md for details.
;

; RUN: %revngopt %s -ternary-reduction -S -o - | FileCheck %s
;
; Ensures that `ternary-reduction` did what was expected of it

define i1 @ternary_reduction_no_const (i1 %condition, i1 %if_true, i1 %if_false) {
  ; CHECK: %1 = select i1 %condition, i1 %if_true, i1 %if_false
  %1 = select i1 %condition, i1 %if_true, i1 %if_false
  ret i1 %1
}

define i1 @ternary_reduction_second_true (i1 %condition, i1 %if_true, i1 %if_false) {
  ; CHECK: %1 = or i1 %condition, %if_false
  %1 = select i1 %condition, i1 true, i1 %if_false
  ret i1 %1
}

define i1 @ternary_reduction_second_false (i1 %condition, i1 %if_true, i1 %if_false) {
  ; CHECK: %1 = call i1 @boolean_not(i1 %condition)
  ; CHECK-NEXT: %2 = and i1 %1, %if_false
  %1 = select i1 %condition, i1 false, i1 %if_false
  ret i1 %1
}

define i1 @ternary_reduction_third_true (i1 %condition, i1 %if_true, i1 %if_false) {
  ; CHECK: %1 = call i1 @boolean_not(i1 %condition)
  ; CHECK-NEXT: %2 = or i1 %1, %if_true
  %1 = select i1 %condition, i1 %if_true, i1 true
  ret i1 %1
}

define i1 @ternary_reduction_third_false (i1 %condition, i1 %if_true, i1 %if_false) {
  ; CHECK: %1 = and i1 %condition, %if_true
  %1 = select i1 %condition, i1 %if_true, i1 false
  ret i1 %1
}
