//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: diff -q <(%revngcliftopt %s) <(%revngcliftopt %s --emit-bytecode | %revngcliftopt)

!U1 = !clift.union<"/type-definition/1-UnionDefinition" : {
  "/union-field/1-UnionDefinition/0" : !clift.ptr<8 to !clift.union<"/type-definition/10-UnionDefinition">>
}>

!U2 = !clift.union<"/type-definition/2-UnionDefinition" : {
  "/union-field/2-UnionDefinition/0" : !clift.ptr<8 to !U1>,
  "/union-field/2-UnionDefinition/1" : !clift.ptr<8 to !U1>,
  "/union-field/2-UnionDefinition/2" : !clift.ptr<8 to !U1>,
  "/union-field/2-UnionDefinition/3" : !clift.ptr<8 to !U1>,
  "/union-field/2-UnionDefinition/4" : !clift.ptr<8 to !U1>,
  "/union-field/2-UnionDefinition/5" : !clift.ptr<8 to !U1>,
  "/union-field/2-UnionDefinition/6" : !clift.ptr<8 to !U1>,
  "/union-field/2-UnionDefinition/7" : !clift.ptr<8 to !U1>
}>

!U3 = !clift.union<"/type-definition/3-UnionDefinition" : {
  "/union-field/3-UnionDefinition/0" : !clift.ptr<8 to !U2>,
  "/union-field/3-UnionDefinition/1" : !clift.ptr<8 to !U2>,
  "/union-field/3-UnionDefinition/2" : !clift.ptr<8 to !U2>,
  "/union-field/3-UnionDefinition/3" : !clift.ptr<8 to !U2>,
  "/union-field/3-UnionDefinition/4" : !clift.ptr<8 to !U2>,
  "/union-field/3-UnionDefinition/5" : !clift.ptr<8 to !U2>,
  "/union-field/3-UnionDefinition/6" : !clift.ptr<8 to !U2>,
  "/union-field/3-UnionDefinition/7" : !clift.ptr<8 to !U2>
}>

!U4 = !clift.union<"/type-definition/4-UnionDefinition" : {
  "/union-field/4-UnionDefinition/0" : !clift.ptr<8 to !U3>,
  "/union-field/4-UnionDefinition/1" : !clift.ptr<8 to !U3>,
  "/union-field/4-UnionDefinition/2" : !clift.ptr<8 to !U3>,
  "/union-field/4-UnionDefinition/3" : !clift.ptr<8 to !U3>,
  "/union-field/4-UnionDefinition/4" : !clift.ptr<8 to !U3>,
  "/union-field/4-UnionDefinition/5" : !clift.ptr<8 to !U3>,
  "/union-field/4-UnionDefinition/6" : !clift.ptr<8 to !U3>,
  "/union-field/4-UnionDefinition/7" : !clift.ptr<8 to !U3>
}>

!U5 = !clift.union<"/type-definition/5-UnionDefinition" : {
  "/union-field/5-UnionDefinition/0" : !clift.ptr<8 to !U4>,
  "/union-field/5-UnionDefinition/1" : !clift.ptr<8 to !U4>,
  "/union-field/5-UnionDefinition/2" : !clift.ptr<8 to !U4>,
  "/union-field/5-UnionDefinition/3" : !clift.ptr<8 to !U4>,
  "/union-field/5-UnionDefinition/4" : !clift.ptr<8 to !U4>,
  "/union-field/5-UnionDefinition/5" : !clift.ptr<8 to !U4>,
  "/union-field/5-UnionDefinition/6" : !clift.ptr<8 to !U4>,
  "/union-field/5-UnionDefinition/7" : !clift.ptr<8 to !U4>
}>

!U6 = !clift.union<"/type-definition/6-UnionDefinition" : {
  "/union-field/6-UnionDefinition/0" : !clift.ptr<8 to !U5>,
  "/union-field/6-UnionDefinition/1" : !clift.ptr<8 to !U5>,
  "/union-field/6-UnionDefinition/2" : !clift.ptr<8 to !U5>,
  "/union-field/6-UnionDefinition/3" : !clift.ptr<8 to !U5>,
  "/union-field/6-UnionDefinition/4" : !clift.ptr<8 to !U5>,
  "/union-field/6-UnionDefinition/5" : !clift.ptr<8 to !U5>,
  "/union-field/6-UnionDefinition/6" : !clift.ptr<8 to !U5>,
  "/union-field/6-UnionDefinition/7" : !clift.ptr<8 to !U5>
}>

!U7 = !clift.union<"/type-definition/7-UnionDefinition" : {
  "/union-field/7-UnionDefinition/0" : !clift.ptr<8 to !U6>,
  "/union-field/7-UnionDefinition/1" : !clift.ptr<8 to !U6>,
  "/union-field/7-UnionDefinition/2" : !clift.ptr<8 to !U6>,
  "/union-field/7-UnionDefinition/3" : !clift.ptr<8 to !U6>,
  "/union-field/7-UnionDefinition/4" : !clift.ptr<8 to !U6>,
  "/union-field/7-UnionDefinition/5" : !clift.ptr<8 to !U6>,
  "/union-field/7-UnionDefinition/6" : !clift.ptr<8 to !U6>,
  "/union-field/7-UnionDefinition/7" : !clift.ptr<8 to !U6>
}>

!U8 = !clift.union<"/type-definition/8-UnionDefinition" : {
  "/union-field/8-UnionDefinition/0" : !clift.ptr<8 to !U7>,
  "/union-field/8-UnionDefinition/1" : !clift.ptr<8 to !U7>,
  "/union-field/8-UnionDefinition/2" : !clift.ptr<8 to !U7>,
  "/union-field/8-UnionDefinition/3" : !clift.ptr<8 to !U7>,
  "/union-field/8-UnionDefinition/4" : !clift.ptr<8 to !U7>,
  "/union-field/8-UnionDefinition/5" : !clift.ptr<8 to !U7>,
  "/union-field/8-UnionDefinition/6" : !clift.ptr<8 to !U7>,
  "/union-field/8-UnionDefinition/7" : !clift.ptr<8 to !U7>
}>

!U9 = !clift.union<"/type-definition/9-UnionDefinition" : {
  "/union-field/9-UnionDefinition/0" : !clift.ptr<8 to !U8>,
  "/union-field/9-UnionDefinition/1" : !clift.ptr<8 to !U8>,
  "/union-field/9-UnionDefinition/2" : !clift.ptr<8 to !U8>,
  "/union-field/9-UnionDefinition/3" : !clift.ptr<8 to !U8>,
  "/union-field/9-UnionDefinition/4" : !clift.ptr<8 to !U8>,
  "/union-field/9-UnionDefinition/5" : !clift.ptr<8 to !U8>,
  "/union-field/9-UnionDefinition/6" : !clift.ptr<8 to !U8>,
  "/union-field/9-UnionDefinition/7" : !clift.ptr<8 to !U8>
}>

!U10 = !clift.union<"/type-definition/10-UnionDefinition" : {
  "/union-field/10-UnionDefinition/0" : !clift.ptr<8 to !U9>,
  "/union-field/10-UnionDefinition/1" : !clift.ptr<8 to !U9>,
  "/union-field/10-UnionDefinition/2" : !clift.ptr<8 to !U9>,
  "/union-field/10-UnionDefinition/3" : !clift.ptr<8 to !U9>,
  "/union-field/10-UnionDefinition/4" : !clift.ptr<8 to !U9>,
  "/union-field/10-UnionDefinition/5" : !clift.ptr<8 to !U9>,
  "/union-field/10-UnionDefinition/6" : !clift.ptr<8 to !U9>,
  "/union-field/10-UnionDefinition/7" : !clift.ptr<8 to !U9>
}>

clift.undef : !U10
