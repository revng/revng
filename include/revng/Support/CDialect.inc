//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#ifndef C_DIALECT_OPTION
#error C_DIALECT_OPTION is not defined.
#endif

/// Conversions between incompatible pointer types (e.g. int and float) are
/// considered implicit.
C_DIALECT_OPTION(ImplicitIncompatiblePointerConversions)

/// Conversions between pointers to integer types with different signedness
/// (e.g. signed int and unsigned int) are considered implicit
C_DIALECT_OPTION(ImplicitPointerSignConversions)

/// Conversions between pointer types where qualifiers are discarded are
/// considered implicit.
C_DIALECT_OPTION(ImplicitQualifierDiscardingConversions)

/// Conversions from pointers-to-void to pointers-to-object-type are
/// considered implicit.
C_DIALECT_OPTION(ImplicitVoidPointerConversions)

#undef C_DIALECT_OPTION
