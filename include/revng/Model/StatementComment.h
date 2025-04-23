#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Support/MetaAddress.h"

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
