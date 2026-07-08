#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Support/MetaAddress.h"

#include "revng/Model/Generated/Early/GotoLabel.h"

class model::GotoLabel : public model::generated::GotoLabel {
public:
  using generated::GotoLabel::GotoLabel;

public:
  bool verify() const debug_function;
  bool verify(bool Assert) const debug_function;
  bool verify(VerifyHelper &VH) const;
};

#include "revng/Model/Generated/Late/GotoLabel.h"
