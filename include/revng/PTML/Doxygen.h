#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
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
                            std::size_t Indentation,
                            std::size_t WrapAt);

/// Emits PTML containing the comment for the specified entity with a comment
/// field.
template<model::EntityWithComment Type>
std::string comment(const ::ptml::PTMLBuilder &Builder,
                    const Type &V,
                    llvm::StringRef Indicator,
                    std::size_t Indent,
                    std::size_t WrapAt) {
  return freeFormComment(Builder, V.Comment(), Indicator, Indent, WrapAt);
}

} // namespace ptml
