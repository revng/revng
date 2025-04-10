#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <string>

#include "revng/Model/CABIFunctionDefinition.h"
#include "revng/Model/Function.h"
#include "revng/Model/Helpers.h"
#include "revng/Model/RawFunctionDefinition.h"
#include "revng/Model/StatementComment.h"
#include "revng/Model/StructDefinition.h"
#include "revng/PTML/Tag.h"

namespace model {
template<typename T>
class NameBuilder;
}

namespace ptml {

/// Emits PTML containing a free-form comment passed in a string reformatted
/// by appending a specified indicator as well maintaining a specific offset.
std::string freeFormComment(const ::ptml::MarkupBuilder &B,
                            llvm::StringRef Text,
                            llvm::StringRef CommentIndicator,
                            size_t Indentation,
                            size_t WrapAt,
                            bool LeadingNewline);

/// Emits PTML containing the comment for the specified entity with a comment
/// field.
template<model::EntityWithComment Type>
std::string comment(const ::ptml::MarkupBuilder &Builder,
                    const Type &V,
                    llvm::StringRef Indicator,
                    size_t Indent,
                    size_t WrapAt) {
  return freeFormComment(Builder, V.Comment(), Indicator, Indent, WrapAt, true);
}

template<model::EntityWithComment Type>
std::string commentWithoutLeadingNewline(const ::ptml::MarkupBuilder &Builder,
                                         const Type &V,
                                         llvm::StringRef Indicator,
                                         size_t Indent,
                                         size_t WrapAt) {
  return freeFormComment(Builder,
                         V.Comment(),
                         Indicator,
                         Indent,
                         WrapAt,
                         false);
}

namespace detail {
std::string functionCommentImpl(const ::ptml::MarkupBuilder &B,
                                const model::Function &Function,
                                const model::Binary &Binary,
                                llvm::StringRef CommentIndicator,
                                size_t Indentation,
                                size_t WrapAt,
                                std::vector<std::string> &&ArgumentNames);
}

/// Emits PTML containing a comment for a function constructed based on
/// the model representation of both the function and its type.
///
/// \param Function the function to gather comment information from.
/// \param Binary the model of the binary the function belongs to.
/// \param CommentIndicator the indicator used at the start of commented lines.
///        For C code, the recommended indicator is either `'///'` or `'  *'`.
///        With second option, the opening (`'/**'`) and closing (`' */'`)
///        brackets are not provided by this function and need to be added
///        by the caller.
/// \param WrapAt the expected width of a line within the emitted comment.
/// \param Indentation the number of spaces appended **before** the comment as
///        an indentation token (see `ptml::tokens::Indentation`).
/// \param NameBuilder an optional naming helper, when not specified, a new one
///        is made using the \ref Binary.
///
/// \returns a serialized PTML string containing the comment.
template<typename T>
std::string functionComment(const ::ptml::MarkupBuilder &B,
                            const model::Function &Function,
                            const model::Binary &Binary,
                            llvm::StringRef CommentIndicator,
                            size_t Indentation,
                            size_t WrapAt,
                            const model::NameBuilder<T> &NameBuilder) {
  std::vector<std::string> ArgumentNames;
  if (const auto *FT = Function.cabiPrototype()) {
    for (const model::Argument &Argument : FT->Arguments())
      ArgumentNames.emplace_back(NameBuilder.name(*FT, Argument));

  } else if (auto *FT = Function.rawPrototype()) {
    for (const model::NamedTypedRegister &Argument : FT->Arguments())
      ArgumentNames.emplace_back(NameBuilder.name(*FT, Argument));

    if (const model::StructDefinition *Stack = FT->stackArgumentsType())
      for (const model::StructField &Field : Stack->Fields())
        ArgumentNames.emplace_back(NameBuilder.name(*Stack, Field));

  } else {
    // If there is no prototype, just act as if there are no arguments.
  }

  return detail::functionCommentImpl(B,
                                     Function,
                                     Binary,
                                     CommentIndicator,
                                     Indentation,
                                     WrapAt,
                                     std::move(ArgumentNames));
}

/// Emits PTML containing a statement comment.
///
/// \param Comment the contents of the comment.
/// \param IsBeingEmittedAt the point where this comment is being emitted at.
///
/// \note for the remaining arguments see \ref functionComment documentation.
///
/// \returns a serialized PTML string containing the comment.
std::string statementComment(const ::ptml::MarkupBuilder &B,
                             const model::StatementComment &Comment,
                             const std::string &CommentLocation,
                             llvm::StringRef IsBeingEmittedAt,
                             llvm::StringRef CommentIndicator,
                             size_t Indentation,
                             size_t WrapAt);

} // namespace ptml
