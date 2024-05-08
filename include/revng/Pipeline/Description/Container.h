#pragma once

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

/* TUPLE-TREE-YAML
name: Container
doc: Data structure representing a Container
type: struct
fields:
  - name: Name
    doc: Container's name
    type: string
  - name: MIMEType
    doc: Container's mime type
    type: string
key:
  - Name
TUPLE-TREE-YAML */

#include "revng/Pipeline/Description/Generated/Early/Container.h"

class pipeline::description::Container
  : public pipeline::description::generated::Container {
public:
  using generated::Container::Container;
};

#include "revng/Pipeline/Description/Generated/Late/Container.h"
