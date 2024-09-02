//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/PTML/Constants.h"
#include "revng/PTML/IndentedOstream.h"
#include "revng/PTML/Tag.h"

namespace ptml {

ptml::Tag MarkupBuilder::getTag(llvm::StringRef Tag) const {
  if (not IsInTaglessMode)
    return ptml::Tag(Tag);

  ptml::Tag EmptyTag;
  return EmptyTag;
}

ptml::Tag MarkupBuilder::getTag(llvm::StringRef Tag,
                                llvm::StringRef Content) const {
  if (not IsInTaglessMode)
    return ptml::Tag(Tag, Content);

  ptml::Tag EmptyTagWithContent;
  EmptyTagWithContent.setContent(Content);
  return EmptyTagWithContent;
}

ptml::Tag MarkupBuilder::scopeTag(const llvm::StringRef AttributeName) const {
  if (not IsInTaglessMode)
    return ptml::Tag(ptml::tags::Div)
      .addAttribute(ptml::attributes::Scope, AttributeName);

  ptml::Tag EmptyTag;
  return EmptyTag;
}

ptml::Tag MarkupBuilder::tokenTag(const llvm::StringRef Str,
                                  const llvm::StringRef Token) const {
  if (not IsInTaglessMode)
    return ptml::Tag(ptml::tags::Span, Str)
      .addAttribute(ptml::attributes::Token, Token);

  ptml::Tag EmptyTagWithContent;
  EmptyTagWithContent.setContent(Str);
  return EmptyTagWithContent;
}

} // namespace ptml
