#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Support/MetaAddress.h"

/* TUPLE-TREE-YAML

name: StatementComment
type: struct

fields:
  - name: Index
    doc: The index of the comment
    type: uint64_t

  - name: Location
    doc: |
      The point this comment is attached to, encoded as a set of addresses.

      When emitted artifact contains a statement with a set of addresses exactly
      matching the set provided here, the comment is emitted before that
      statement.

      If there's no such statement, one is chosen based on how similar its
      address set is to the address set of the comment.

      Note that the same comment can be emitted multiple times if there are
      multiple statements (that do *not* dominate each other) with the same
      address set.

    sequence:
      type: SortedVector
      elementType: MetaAddress

  - name: Body
    type: string

key:
  - Index

TUPLE-TREE-YAML */

#include "revng/Model/Generated/Early/StatementComment.h"

class model::StatementComment : public model::generated::StatementComment {
public:
  using generated::StatementComment::StatementComment;

public:
  bool verify() const debug_function;
  bool verify(bool Assert) const debug_function;
  bool verify(VerifyHelper &VH) const;
};

#include "revng/Model/Generated/Late/StatementComment.h"
