#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/StringRef.h"

struct TypeEmitterConfiguration {
  /// We don't always want the complete type system. For example, when one
  /// of the types is being edited, we cannot include the type in question!
  /// As well as every type that depends on its definition.
  ///
  /// Note: the type is identified by its handle.
  llvm::StringRef TypeToOmit = "";

  /// Because we are emitting C11, we cannot specify underlying enum type
  /// (the feature was only backported from C++ in C23), which means that
  /// when we need to preserve enum size across the compilation boundary
  /// (for example, when editing a type), we need to resolve to a trick:
  /// we emit the maximum value possible - and then use it to figure out
  /// the original size.
  /// Setting this flag to `true` enables emitting of such entry.
  bool EmitMaximumEnumValue = true;

  /// When editing types, it would be extremely annoying to have to do every
  /// change twice. As such, we need a way to disable emission of the explicit
  /// padding fields.
  ///
  /// Note that setting this leads to changes in the struct layout were they
  /// recompiled with a normal compiler (*our* wrapper used when editing types
  /// is not affected because it explicitly handles `_STARTS_AT` attribute).
  bool ExplicitPadding = true;

  /// When emit the stack variable, emit the *definition* of its struct instead
  /// of a declaration.
  ///
  /// Note that the inlined definition is emitted regardless of the presence of
  /// the global definition for the same type, which means the produced C is
  /// intentionally not syntactically valid as a single translation unit.
  bool InlineStackFrameType = false;
};
