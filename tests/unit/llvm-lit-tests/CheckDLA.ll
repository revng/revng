;
; Copyright rev.ng Labs Srl. See LICENSE.md for details.
;

; RUN: %revngopt %s -S --dla -o - | revng model dump | revng model compare %s.yml -
; This file is meant to be used to test that the DLA is able to merge the
; recovered type information with an already existing model.
; In particular, in this file all functions and indirect calls share the same
; prototype in the model, so all the information recovered by the DLA from each
; call should ultimately end into the same model entity, i.e. the prototype.
target triple = "x86_64-unknown-linux-gnu"


define void @local_function_0x401200_Code_x86_64(i64 %rax, i64 %rdx) !revng.tags !58898 !revng.function.metadata !59051 !revng.function.entry !59052 {
newFuncRoot:
  %0 = inttoptr i64 %rdx to i64*
  store i64 %rax, i64* %0, align 8
  %1 = add i64 %rdx, 8
  %2 = inttoptr i64 %1 to i64*
  store i64 0, i64* %2, align 8
  ret void
}

define void @local_function_0x401060_Code_x86_64(i64 %rax, i64 %rdx) !revng.tags !58898 !revng.function.metadata !58935 !revng.function.entry !58936 {
newFuncRoot:
  %0 = add i64 %rax, 0                ; rax
  %1 = add i64 %0, 8                  ; rax + 8
  %2 = add i64 %1, 16                 ; rax + 16
  %3 = add i64 %2, -8                 ; rax + 8
  %4 = inttoptr i64 %3 to i64*
  store i64 %rax, i64* %4, align 8    ; store rax into rax+8
  %5 = add i64 %2, -16                ; rax
  %6 = inttoptr i64 %5 to i64*
  store i64 %3, i64* %6, align 16     ; store rax+8 into rax
  %7 = load i64, i64* inttoptr (i64 4210656 to i64*), align 32 ; load from const
  %8 = add i64 %2, 0                  ; rax + 0
  %9 = inttoptr i64 %8 to i64*
  store i64 4198542, i64* %9, align 8 ; store 8-byte const into rax-8
  %10 = inttoptr i64 %7 to void (i64, i64)*
  call void %10(i64 %rax, i64 %rdx), !revng.callerblock.start !58936
  ret void
}

; General Metadata
!revng.input.architecture = !{!5870}
!revng.model = !{!5872}

!58898 = !{!"Isolated"}
!5870 = !{!"x86_64", i32 1, i32 0, !"pc", !"rsp", !5871}
!5871 = !{!"rax", !"rbx", !"rcx", !"rdx", !"rbp", !"rsp", !"rsi", !"rdi", !"r8", !"r9", !"r10", !"r11", !"r12", !"r13", !"r14", !"r15", !"state_0x8558", !"state_0x8598", !"state_0x85d8", !"state_0x8618", !"state_0x8658", !"state_0x8698", !"state_0x86d8", !"state_0x8718"}

@a1 = internal constant [21 x i8] c"0x401200:Code_x86_64\00"
!59052 = !{i8* getelementptr inbounds ([21 x i8], [21 x i8]* @a1, i32 0, i32 0)}
@a2 = internal constant [21 x i8] c"0x401060:Code_x86_64\00"
!58936 = !{i8* getelementptr inbounds ([21 x i8], [21 x i8]* @a2, i32 0, i32 0)}

; Function metadata
!59051 = !{!"---
Entry:           \220x401200:Code_x86_64\22
ControlFlowGraph:
  - Start:           \220x401200:Code_x86_64\22
    End:             \220x401205:Code_x86_64\22
    Successors:
      - Kind: FunctionEdge
        Destination:     \22:Invalid\22
        Type:            Return
...
"}

!58935 = !{!"---
Entry:           \220x401060:Code_x86_64\22
ControlFlowGraph:
  - Start:           \220x401060:Code_x86_64\22
    End:             \220x40108e:Code_x86_64\22
    Successors:
      - Kind: CallEdge
        Destination:     \22:Invalid\22
        Type:            FunctionCall
  - Start:           \220x40108e:Code_x86_64\22
    End:             \220x40108f:Code_x86_64\22
    Successors:
    - Kind: FunctionEdge
      Destination:     \22:Invalid\22
      Type:            Return
...
"}

; Model
!5872 = !{!"---
EntryPoint: \220x401200:Code_x86_64\22
Functions:
  - Entry:           \220x401200:Code_x86_64\22
    Prototype:       \22/Types/RawFunctionType-6646220838590018230\22
  - Entry:           \220x401060:Code_x86_64\22
    Prototype:       \22/Types/RawFunctionType-6646220838590018230\22
ImportedDynamicFunctions: []
Types:
  - Kind:            PrimitiveType
    ID:              520
    PrimitiveKind:   Generic
    Size:            8
  - Kind:            PrimitiveType
    ID:              776
    PrimitiveKind:   PointerOrNumber
    Size:            8
  - Kind:            RawFunctionType
    ID:              6646220838590018230
    Arguments:
      - Location:        rax_x86_64
        Type:
          UnqualifiedType: \22/Types/PrimitiveType-520\22
      - Location:        rdx_x86_64
        Type:
          UnqualifiedType: \22/Types/PrimitiveType-520\22
    ReturnValues:    []
    PreservedRegisters: []
    FinalStackOffset: 0
Architecture:    x86_64
Segments:
  - StartAddress:    \220x400000:Generic64\22
    VirtualSize:     1480
    StartOffset:     0
    FileSize:        1480
    IsReadable:      true
    IsWriteable:     false
    IsExecutable:    false
...
"}
