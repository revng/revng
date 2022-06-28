#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Yield/Graph.h"

namespace yield::cfg {

struct Configuration {
public:
  Graph::Dimension EdgeMarginSize;
  Graph::Dimension InternalNodeMarginSize;
  Graph::Dimension ExternalNodeMarginSize;
  Graph::Dimension HorizontalInstructionMarginSize;
  Graph::Dimension VerticalInstructionMarginSize;

  Graph::Dimension HorizontalFontFactor;
  Graph::Dimension VerticalFontFactor;

  Graph::Dimension InstructionFontSize;
  Graph::Dimension InstructionAddressFontSize;
  Graph::Dimension InstructionBytesFontSize;
  Graph::Dimension CommentFontSize;
  Graph::Dimension LabelFontSize;

  size_t NodeCornerRoundingFactor;

  bool UseOrthogonalBends;
  bool AddEntryNode;
  bool AddExitNode;

  bool PreserveLinearSegments;
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
                          .InstructionAddressFontSize = 12.f,
                          .InstructionBytesFontSize = 0.f,
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
