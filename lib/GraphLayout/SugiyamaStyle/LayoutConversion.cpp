/// \file LayoutConversion.cpp
/// \brief

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "InternalCompute.h"

LayoutContainer convertToLayout(const LayerContainer &Layers) {
  LayoutContainer Result;
  for (size_t Index = 0; Index < Layers.size(); ++Index)
    for (size_t NodeIndex = 0; NodeIndex < Layers[Index].size(); ++NodeIndex)
      Result[Layers[Index][NodeIndex]] = { Index, NodeIndex };
  return Result;
}
