#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <cstdlib>

namespace revng::pypeline::helpers {

template<typename C, size_t I, typename ListType>
struct ExtractContainerFromList {
  static C &get(ListType &Containers);
};

} // namespace revng::pypeline::helpers
