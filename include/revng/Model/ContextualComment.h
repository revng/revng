#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Support/MetaAddress.h"

/* TUPLE-TREE-YAML

name: ContextualComment
type: struct

fields:
  - name: Location
    doc: |
      The point this comment is attached to
    type: MetaAddress
  - name: Text
    type: string

key:
  - Location

TUPLE-TREE-YAML */

#include "revng/Model/Generated/Early/ContextualComment.h"

class model::ContextualComment : public model::generated::ContextualComment {
public:
  using generated::ContextualComment::ContextualComment;

};

#include "revng/Model/Generated/Late/ContextualComment.h"
