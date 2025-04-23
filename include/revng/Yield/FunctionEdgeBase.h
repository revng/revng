#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Support/BasicBlockID/YAMLTraits.h"
#include "revng/Support/MetaAddress.h"
#include "revng/Support/MetaAddress/YAMLTraits.h"
#include "revng/Yield/FunctionEdgeType.h"

#include "revng/Yield/Generated/Early/FunctionEdgeBase.h"

namespace model {
class VerifyHelper;
}

namespace yield {
class CallEdge;
}

class yield::FunctionEdgeBase : public yield::generated::FunctionEdgeBase {
public:
  using generated::FunctionEdgeBase::FunctionEdgeBase;

public:
  bool isDirect() const { return Destination().isValid(); }
  bool isIndirect() const { return not isDirect(); }

  // Returns nullptr if this is not a call edge.
  const yield::CallEdge *getCallEdge() const;

public:
  bool verify() const debug_function;
  bool verify(bool Assert) const debug_function;
  bool verify(model::VerifyHelper &VH) const;
};

#include "revng/Yield/Generated/Late/FunctionEdgeBase.h"
