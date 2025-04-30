#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <string>

#include "revng/Yield/Generated/Early/TagAttribute.h"

namespace yield {

class TagAttribute : public generated::TagAttribute {
public:
  using generated::TagAttribute::TagAttribute;
};

} // namespace yield

#include "revng/Yield/Generated/Late/TagAttribute.h"
