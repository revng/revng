/// \file ConvertFromEFA.cpp

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

#include "revng/EarlyFunctionAnalysis/CallEdge.h"
#include "revng/EarlyFunctionAnalysis/FunctionEdge.h"
#include "revng/EarlyFunctionAnalysis/FunctionEdgeBase.h"
#include "revng/Yield/CallEdge.h"
#include "revng/Yield/FunctionEdge.h"
#include "revng/Yield/FunctionEdgeBase.h"

yield::CallEdge::CallEdge(const efa::CallEdge &Source) {
  Kind() = yield::FunctionEdgeBaseKind::CallEdge;
  Destination() = Source.Destination();
  Type() = yield::FunctionEdgeType::from(Source.Type());
  DynamicFunction() = Source.DynamicFunction();
  IsTailCall() = Source.IsTailCall();
  Attributes() = Source.Attributes();
}

yield::FunctionEdge::FunctionEdge(const efa::FunctionEdge &Source) {
  Kind() = yield::FunctionEdgeBaseKind::FunctionEdge;
  Destination() = Source.Destination();
  Type() = yield::FunctionEdgeType::from(Source.Type());
}
