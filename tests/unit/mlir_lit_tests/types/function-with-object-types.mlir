//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s

!int32_t = !clift.primitive<signed 4>

!f = !clift.defined<#clift.func<"/model-type/1000" as "f" : !int32_t(!int32_t)>>
