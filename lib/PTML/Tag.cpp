//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/PTML/Constants.h"
#include "revng/PTML/IndentedOstream.h"
#include "revng/PTML/Tag.h"

namespace ptml {
PTMLBuilder::PTMLBuilder(bool GenerateTagLessPTML) :
  GenerateTagLessPTML(GenerateTagLessPTML) {
}

bool PTMLBuilder::isGenerateTagLessPTML() const {
  return GenerateTagLessPTML;
}

ptml::Tag PTMLBuilder::getTag(llvm::StringRef Tag) const {
  if (!GenerateTagLessPTML)
    return ptml::Tag(Tag);

  ptml::Tag EmptyTag;
  return EmptyTag;
}

ptml::Tag
PTMLBuilder::getTag(llvm::StringRef Tag, llvm::StringRef Content) const {
  if (!GenerateTagLessPTML)
    return ptml::Tag(Tag, Content);

  ptml::Tag EmptyTagWithContent;
  EmptyTagWithContent.setContent(Content);
  return EmptyTagWithContent;
}

ptml::Tag PTMLBuilder::scopeTag(const llvm::StringRef AttributeName) const {
  if (!GenerateTagLessPTML)
    return ptml::Tag(ptml::tags::Div)
      .addAttribute(ptml::attributes::Scope, AttributeName);

  ptml::Tag EmptyTag;
  return EmptyTag;
}

ptml::Tag PTMLBuilder::tokenTag(const llvm::StringRef Str,
                                const llvm::StringRef Token) const {
  if (!GenerateTagLessPTML)
    return ptml::Tag(ptml::tags::Span, Str)
      .addAttribute(ptml::attributes::Token, Token);

  ptml::Tag EmptyTagWithContent;
  EmptyTagWithContent.setContent(Str);
  return EmptyTagWithContent;
}
} // namespace ptml
