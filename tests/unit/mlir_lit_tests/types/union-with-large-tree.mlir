//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: diff -q <(%revngcliftopt %s) <(%revngcliftopt %s --emit-bytecode | %revngcliftopt)

!U1 = !clift.union<"/type-definition/1-UnionDefinition" : {
  !clift.ptr<8 to !clift.union<"/type-definition/10-UnionDefinition">>
}>

!U2 = !clift.union<"/type-definition/2-UnionDefinition" : {
  !clift.ptr<8 to !U1>,
  !clift.ptr<8 to !U1>,
  !clift.ptr<8 to !U1>,
  !clift.ptr<8 to !U1>,
  !clift.ptr<8 to !U1>,
  !clift.ptr<8 to !U1>,
  !clift.ptr<8 to !U1>,
  !clift.ptr<8 to !U1>
}>

!U3 = !clift.union<"/type-definition/3-UnionDefinition" : {
  !clift.ptr<8 to !U2>,
  !clift.ptr<8 to !U2>,
  !clift.ptr<8 to !U2>,
  !clift.ptr<8 to !U2>,
  !clift.ptr<8 to !U2>,
  !clift.ptr<8 to !U2>,
  !clift.ptr<8 to !U2>,
  !clift.ptr<8 to !U2>
}>

!U4 = !clift.union<"/type-definition/4-UnionDefinition" : {
  !clift.ptr<8 to !U3>,
  !clift.ptr<8 to !U3>,
  !clift.ptr<8 to !U3>,
  !clift.ptr<8 to !U3>,
  !clift.ptr<8 to !U3>,
  !clift.ptr<8 to !U3>,
  !clift.ptr<8 to !U3>,
  !clift.ptr<8 to !U3>
}>

!U5 = !clift.union<"/type-definition/5-UnionDefinition" : {
  !clift.ptr<8 to !U4>,
  !clift.ptr<8 to !U4>,
  !clift.ptr<8 to !U4>,
  !clift.ptr<8 to !U4>,
  !clift.ptr<8 to !U4>,
  !clift.ptr<8 to !U4>,
  !clift.ptr<8 to !U4>,
  !clift.ptr<8 to !U4>
}>

!U6 = !clift.union<"/type-definition/6-UnionDefinition" : {
  !clift.ptr<8 to !U5>,
  !clift.ptr<8 to !U5>,
  !clift.ptr<8 to !U5>,
  !clift.ptr<8 to !U5>,
  !clift.ptr<8 to !U5>,
  !clift.ptr<8 to !U5>,
  !clift.ptr<8 to !U5>,
  !clift.ptr<8 to !U5>
}>

!U7 = !clift.union<"/type-definition/7-UnionDefinition" : {
  !clift.ptr<8 to !U6>,
  !clift.ptr<8 to !U6>,
  !clift.ptr<8 to !U6>,
  !clift.ptr<8 to !U6>,
  !clift.ptr<8 to !U6>,
  !clift.ptr<8 to !U6>,
  !clift.ptr<8 to !U6>,
  !clift.ptr<8 to !U6>
}>

!U8 = !clift.union<"/type-definition/8-UnionDefinition" : {
  !clift.ptr<8 to !U7>,
  !clift.ptr<8 to !U7>,
  !clift.ptr<8 to !U7>,
  !clift.ptr<8 to !U7>,
  !clift.ptr<8 to !U7>,
  !clift.ptr<8 to !U7>,
  !clift.ptr<8 to !U7>,
  !clift.ptr<8 to !U7>
}>

!U9 = !clift.union<"/type-definition/9-UnionDefinition" : {
  !clift.ptr<8 to !U8>,
  !clift.ptr<8 to !U8>,
  !clift.ptr<8 to !U8>,
  !clift.ptr<8 to !U8>,
  !clift.ptr<8 to !U8>,
  !clift.ptr<8 to !U8>,
  !clift.ptr<8 to !U8>,
  !clift.ptr<8 to !U8>
}>

!U10 = !clift.union<"/type-definition/10-UnionDefinition" : {
  !clift.ptr<8 to !U9>,
  !clift.ptr<8 to !U9>,
  !clift.ptr<8 to !U9>,
  !clift.ptr<8 to !U9>,
  !clift.ptr<8 to !U9>,
  !clift.ptr<8 to !U9>,
  !clift.ptr<8 to !U9>,
  !clift.ptr<8 to !U9>
}>

clift.undef : !U10
