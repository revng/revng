#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Yield/ControlFlow/Configuration.h"
#include "revng/Yield/ControlFlow/Graph.h"

namespace model {
class Binary;
}
namespace yield {
class Function;
}

namespace yield::cfg {

constexpr inline yield::layout::Size pureTextSize(std::string_view Text) {
  size_t LineCount = 0;
  size_t MaximumLineLength = 0;

  size_t PreviousPosition = 0;
  size_t CurrentPosition = Text.find('\n');
  while (CurrentPosition != std::string_view::npos) {
    size_t CurrentLineLength = CurrentPosition - PreviousPosition;
    if (CurrentLineLength > MaximumLineLength)
      MaximumLineLength = CurrentLineLength;
    ++LineCount;

    PreviousPosition = CurrentPosition;
    CurrentPosition = Text.find('\n', CurrentPosition + 1);
  }

  size_t LastLineLength = Text.size() - PreviousPosition;
  if (LastLineLength > MaximumLineLength)
    MaximumLineLength = LastLineLength;
  if (LastLineLength != 0)
    ++LineCount;

  // Ignore empty lines at the end
  size_t TrailingEmptyLineCount = Text.size() - Text.find_last_not_of('\n');
  if (TrailingEmptyLineCount > 1)
    LineCount -= TrailingEmptyLineCount - 1;

  return yield::layout::Size(MaximumLineLength, LineCount);
}

inline yield::layout::Size
fontSize(yield::layout::Size &&Input,
         const yield::cfg::Configuration &Configuration) {
  auto ExtraPadding = Input.H * Configuration.PaddingBetweenLines;

  Input.W *= Configuration.FontSize * Configuration.HorizontalFontFactor;
  Input.H *= Configuration.FontSize * Configuration.VerticalFontFactor;

  Input.H += ExtraPadding;

  return std::move(Input);
}

inline yield::layout::Size
textSize(std::string_view Text,
         const yield::cfg::Configuration &Configuration) {
  return fontSize(pureTextSize(Text), Configuration);
}

inline yield::layout::Size
nodeSize(std::string_view Text,
         const yield::cfg::Configuration &Configuration) {
  yield::layout::Size Result = textSize(Text, Configuration);

  Result.W += Configuration.InternalNodeMarginSize * 2;
  Result.H += Configuration.InternalNodeMarginSize * 2;

  return Result;
}

inline yield::layout::Size
nodeSize(size_t HorizontalCharacterCount,
         size_t VerticalCharacterCount,
         const yield::cfg::Configuration &Configuration) {
  auto Result = fontSize(yield::layout::Size(HorizontalCharacterCount,
                                             VerticalCharacterCount),
                         Configuration);

  Result.W += Configuration.InternalNodeMarginSize * 2;
  Result.H += Configuration.InternalNodeMarginSize * 2;

  return Result;
}

inline yield::layout::Size
emptyNodeSize(const yield::cfg::Configuration &Configuration) {
  return yield::layout::Size{
    .W = Configuration.EmptyNodeSize + Configuration.InternalNodeMarginSize * 2,
    .H = Configuration.EmptyNodeSize + Configuration.InternalNodeMarginSize * 2
  };
}

} // namespace yield::cfg
