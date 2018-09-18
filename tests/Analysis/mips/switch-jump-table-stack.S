#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

    .globl  __start
    .set    noreorder
    .set    nomacro
jumptable:
    .word one
    .word two
    .word three
__start:
    lw      $v0,696($s8)
    sltiu   $v1,$v0,3
    beqz    $v1,end
    nop
    lui     $gp,%hi(jumptable)
    addiu   $gp,$gp,%lo(jumptable)
    lw      $v0,696($s8)
    sll     $v0,$v0,2
    add     $gp,$gp,$v0
    lw      $t9,0($gp)
    nop
    beqz    $t9,nullptr
    nop
    jr      $t9
    nop
end:
    jr      $ra
    nop
nullptr:
    jr      $ra
    nop
one:
    jr      $ra
    nop
two:
    jr      $ra
    nop
three:
    jr      $ra
    nop
