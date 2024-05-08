#pragma once

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

#include <string>

/* TUPLE-TREE-YAML

name: TagAttribute
type: struct
fields:
  - name: Name
    type: string

  - name: Value
    type: string

key:
  - Name

TUPLE-TREE-YAML */

#include "revng/Yield/Generated/Early/TagAttribute.h"

namespace yield {

class TagAttribute : public generated::TagAttribute {
public:
  using generated::TagAttribute::TagAttribute;
};

} // namespace yield

#include "revng/Yield/Generated/Late/TagAttribute.h"
