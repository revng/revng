#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/EarlyFunctionAnalysis/FunctionEdgeType.h"
#include "revng/Support/BasicBlockID/YAMLTraits.h"
#include "revng/Support/MetaAddress.h"
#include "revng/Support/MetaAddress/YAMLTraits.h"

#include "revng/EarlyFunctionAnalysis/Generated/Early/FunctionEdgeBase.h"

namespace model {
class VerifyHelper;
}

namespace efa {
class CallEdge;
}

class efa::FunctionEdgeBase : public efa::generated::FunctionEdgeBase {
public:
  using generated::FunctionEdgeBase::FunctionEdgeBase;

public:
  bool isDirect() const { return Destination().isValid(); }
  bool isIndirect() const { return not isDirect(); }

  /// \return nullptr if this is not a call edge.
  const efa::CallEdge *getCallEdge() const;

public:
  bool verify() const debug_function;
  bool verify(bool Assert) const debug_function;
  bool verify(model::VerifyHelper &VH) const;
};

#include "revng/EarlyFunctionAnalysis/Generated/Late/FunctionEdgeBase.h"
