#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/Type.h"

#include "revng/Model/Generated/Early/StackFrame.h"

class model::StackFrame : public model::generated::StackFrame {
public:
  using generated::StackFrame::StackFrame;
};

#include "revng/Model/Generated/Late/StackFrame.h"
