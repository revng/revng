#pragma once

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

#include <string>

#include "revng/Model/Helpers.h"
#include "revng/PTML/Tag.h"

namespace ptml {

/// Emits PTML containing a free-form comment passed in a string reformatted
/// by appending a specified indicator as well maintaining a specific offset.
std::string freeFormComment(const ::ptml::PTMLBuilder &ThePTMLBuilder,
                            llvm::StringRef Text,
                            llvm::StringRef CommentIndicator,
                            size_t Indentation,
                            size_t WrapAt);

/// Emits PTML containing the comment for the specified entity with a comment
/// field.
template<model::EntityWithComment Type>
std::string comment(const ::ptml::PTMLBuilder &Builder,
                    const Type &V,
                    llvm::StringRef Indicator,
                    size_t Indent,
                    size_t WrapAt) {
  return freeFormComment(Builder, V.Comment(), Indicator, Indent, WrapAt);
}

/// Emits PTML containing a comment for a function constructed based on
/// the model representation of both the function and its type.
///
/// \param Function the function to gather comment information from.
/// \param Binary the model of the binary the function belongs to.
/// \param CommentIndicator the indicator used at the start of commented lines.
///        If no custom value is specified, the value obtained from
///        `model::Architecture::getAssemblyCommentIndicator` is used instead,
///        which is the preferred option for the assembly (see the inline
///        overload calling this function).
///        For C code, the recommended indicator is either `'///'` or `'  *'`.
///        With second option, the opening (`'/**'`) and closing (`' */'`)
///        brackets are not provided by this function and need to be added
///        by the caller.
/// \param WrapAt the expected width of a line within the emitted comment.
/// \param Indentation the number of spaces appended **before** the comment as
///        an indentation token (see `ptml::tokens::Indentation`).
///
/// \returns a serialized PTML string containing the comment.
std::string functionComment(const ::ptml::PTMLBuilder &ThePTMLBuilder,
                            const model::Function &Function,
                            const model::Binary &Binary,
                            llvm::StringRef CommentIndicator,
                            size_t Indentation,
                            size_t WrapAt);

} // namespace ptml
