; RUN: %revngopt %s --exit-ssa -S -o - | FileCheck %s

; CHECK: define i1 @basicphi
define i1 @basicphi (i1 %x) !revng.tags !0 {
  ; The phi node below becomes a single alloca
  ; CHECK: %1 = alloca i1
  ; CHECK-NOT: alloca
  ; The ExitSSAPass.cpp decides that it's safe to push the store up to here.
  ; CHECK: store i1 true, i1* %1
  ; CHECK-NEXT: br i1 %x, label %then, label %else
  br i1 %x, label %then, label %else

then:
  ; This block stays empty, because the associated store has been pushed above.
  ; CHECK: then:
  ; CHECK-NEXT: br label %tail
  br label %tail

else:
  ; This block now contains a store
  ; CHECK: else:
  ; CHECK-NEXT: store i1 false, i1* %1
  ; CHECK-NEXT: br label %tail
  br label %tail

tail:
  ; The following phi has become an alloca, so now here we don't have any phis
  ; anymore, we just have a load from the alloca and the ret.
  ; CHECK: tail:
  ; CHECK-NEXT: %2 = load i1, i1* %1
  ; CHECK-NEXT: ret i1 %2
  %res = phi i1 [true, %then], [false, %else]
  ret i1 %res
}

; CHECK: define i1 @nestedifwithglobals
define i1 @nestedifwithglobals (i1 %x) !revng.tags !0 {
  ; CHECK: %1 = alloca i1
  ; CHECK-NEXT: %2 = alloca i1
  ; CHECK-NEXT: store i1 true, i1* %2
  ; CHECK-NEXT: store i1 true, i1* %1
  ; CHECK-NEXT: br i1 %x, label %then, label %else
  br i1 %x, label %then, label %else

then:
  ; This block stays empty, because the associated store has been pushed above.
  ; CHECK: then:
  ; CHECK-NEXT: br label %tail
  br label %tail

else:
  ; This block stays empty, because the associated store has been pushed above.
  ; CHECK: else:
  ; CHECK-NEXT: br i1 %x, label %then2, label %else2
  br i1 %x, label %then2, label %else2

then2:
  ; This block stays empty, because the associated store has been pushed above.
  ; CHECK: then2:
  ; CHECK-NEXT: br label %tail2
  br label %tail2

else2:
  ; This block now contains a store
  ; CHECK: else2:
  ; CHECK-NEXT: store i1 false, i1* %2
  ; CHECK-NEXT: br label %tail2
  br label %tail2

tail2:
  ; The following phi has become an alloca, so now here we don't have any phis
  ; anymore, we just have a load from the alloca. Then we have a store in the
  ; other alloca, associated with the second phi.
  ; Finally we branch to tail
  ; CHECK: tail2:
  ; CHECK-NEXT: %3 = load i1, i1* %2
  ; CHECK-NEXT: store i1 %3, i1* %1
  ; CHECK-NEXT: br label %tail
  %a = phi i1 [true, %then2], [false, %else2]
  br label %tail

tail:
  ; The following phi has become an alloca, so now here we don't have any phis
  ; anymore, we just have a load from the alloca and the ret.
  ; CHECK: tail:
  ; CHECK-NEXT: %4 = load i1, i1* %1
  ; CHECK-NEXT: ret i1 %4
  %res = phi i1 [true, %then], [%a, %tail2]
  ret i1 %res
}

; CHECK: define i1 @nestedifwithlocals
define i1 @nestedifwithlocals (i1 %x) !revng.tags !0 {
  ; CHECK: %1 = alloca i1
  ; CHECK-NEXT: store i1 true, i1* %1
  ; CHECK-NEXT: br i1 %x, label %then, label %else
  br i1 %x, label %then, label %else

then:
  ; This block stays empty, because the associated store has been pushed above.
  ; CHECK: then:
  ; CHECK-NEXT: br label %tail
  br label %tail

else:
  ; This block stays empty, because the associated store has been pushed above.
  ; CHECK: else:
  ; CHECK-NEXT: %2 = alloca i1
  ; CHECK-NEXT: %local1 = and i1 %x, %x
  ; CHECK-NEXT: store i1 %local1, i1* %2
  ; CHECK-NEXT: br i1 %x, label %then2, label %else2
  %local1 = and i1 %x, %x
  br i1 %x, label %then2, label %else2

then2:
  ; This block stays empty, because the associated store has been pushed above.
  ; CHECK: then2:
  ; CHECK-NEXT: br label %tail2
  br label %tail2

else2:
  ; This block now contains a store
  ; CHECK: else2:
  ; CHECK-NEXT: %local2 = xor i1 %x, %x
  ; CHECK-NEXT: store i1 %local2, i1* %2
  ; CHECK-NEXT: br label %tail2
  %local2 = xor i1 %x, %x
  br label %tail2

tail2:
  ; The following phi has become an alloca, so now here we don't have any phis
  ; anymore, we just have a load from the alloca. Then we have a store in the
  ; other alloca, associated with the second phi.
  ; Finally we branch to tail
  ; CHECK: tail2:
  ; CHECK-NEXT: %3 = load i1, i1* %2
  ; CHECK-NEXT: store i1 %3, i1* %1
  ; CHECK-NEXT: br label %tail
  %a = phi i1 [%local1, %then2], [%local2, %else2]
  br label %tail

tail:
  ; The following phi has become an alloca, so now here we don't have any phis
  ; anymore, we just have a load from the alloca and the ret.
  ; CHECK: tail:
  ; CHECK-NEXT: %4 = load i1, i1* %1
  ; CHECK-NEXT: ret i1 %4
  %res = phi i1 [true, %then], [%a, %tail2]
  ret i1 %res
}

!0 = !{!"Isolated"}
