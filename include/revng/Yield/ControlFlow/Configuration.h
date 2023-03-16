#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/GraphLayout/Traits.h"

namespace yield::cfg {

struct Configuration {
public:
  /// Specifies the minimum possible distance between two edges.
  layout::Dimension EdgeMarginSize;

  /// Specifies the distance between the node margin and its contents.
  layout::Dimension InternalNodeMarginSize;

  /// Specifies the minimum possible distance between two nodes.
  layout::Dimension ExternalNodeMarginSize;

  /// Specifies the minimum possible horizontal distance between
  /// an instruction element and the border of its node.
  layout::Dimension HorizontalInstructionMarginSize;

  /// Specifies the minimum possible distance between two instructions.
  layout::Dimension VerticalInstructionMarginSize;

  /// Specifies the relation between the horizontal size of a single character
  /// to the size of the font.
  ///
  /// \note: Only monospace fonts should be used. For other fonts, it's possible
  /// to select a value for this factor that look decent for most cases, but
  /// even then corner cases are unavoidable (e.g. "WWWW" is wider than "IIII").
  ///
  /// \note: the easiest way to measure this value is to use a browser to render
  /// a long line of text (the longer the line, the more accurate the value),
  /// let it report the final length of the said line.
  /// This factor is equal to the size of the line divided by the number of
  /// characters in it and the size of the font used.
  layout::Dimension HorizontalFontFactor;

  /// Specifies the relation between the vertical size of a single character
  /// to the size of the font.
  ///
  /// \note: the easiest way to measure this value is to use a browser to render
  /// may lines of text (this value highly depends on the `line-height` css
  /// property, so it's recommended to set a consistent value for both the value
  /// measuring case, and the final result. Note that different browser use
  /// different defaults, so it's recommended to set this value explicitly in
  /// all cases) let it report the final size of that paragraph.
  /// This factor is equal to the height of the paragraph divided by the number
  /// of lines in it and the size of the font used.
  layout::Dimension VerticalFontFactor;

  /// Specifies the size of the font used for instruction rendering.
  ///
  /// It's used to approximate node sizes.
  layout::Dimension InstructionFontSize;

  /// Specifies the size of the font used for instruction address and raw bytes
  /// rendering.
  ///
  /// It's used to approximate node sizes.
  ///
  /// \note this value should be set to zero if no bytes are displayed.
  layout::Dimension AnnotationFontSize;

  /// Specifies the size of the font used for comment rendering.
  ///
  /// It's used to approximate node sizes.
  layout::Dimension CommentFontSize;

  /// Specifies the size of the font used for label rendering.
  ///
  /// It's used to approximate node sizes.
  layout::Dimension LabelFontSize;

  /// Specifies the degree to which node corners should be rounded.
  size_t NodeCornerRoundingFactor;

  /// Specifies whether orthogonal bends should be used.
  bool UseOrthogonalBends;

  /// Specifies whether an "empty" entry node should be added to denote entry
  /// points
  bool AddEntryNode;

  /// Specifies whether an "empty" entry node should be added to denote exit
  /// points
  bool AddExitNode;

  /// Specifies whether linear segments should be preserved (for more
  /// information see layouter documentation).
  bool PreserveLinearSegments;

  /// Specifies the relative weight of virtual nodes (for more
  /// information see layouter documentation).
  float VirtualNodeWeight;

public:
  constexpr static Configuration getDefault() {
    return Configuration{ .EdgeMarginSize = 20.f,
                          .InternalNodeMarginSize = 12.f,
                          .ExternalNodeMarginSize = 40.f,
                          .HorizontalInstructionMarginSize = 20.f,
                          .VerticalInstructionMarginSize = 2.f,

                          .HorizontalFontFactor = 0.6f,
                          .VerticalFontFactor = 1.25f,

                          .InstructionFontSize = 18.f,
                          .AnnotationFontSize = 0.f,
                          .CommentFontSize = 15.f,
                          .LabelFontSize = 18.f,

                          .NodeCornerRoundingFactor = 5,

                          .UseOrthogonalBends = true,
                          .AddEntryNode = true,
                          .AddExitNode = false,

                          .PreserveLinearSegments = true,
                          .VirtualNodeWeight = 0.1f };
  }
};

} // namespace yield::cfg
