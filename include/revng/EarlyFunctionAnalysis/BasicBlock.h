#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/ABI/FunctionType/Layout.h"
#include "revng/ADT/SortedVector.h"
#include "revng/EarlyFunctionAnalysis/CallEdge.h"
#include "revng/EarlyFunctionAnalysis/FunctionEdge.h"
#include "revng/EarlyFunctionAnalysis/FunctionEdgeBase.h"
#include "revng/Model/CallSitePrototype.h"
#include "revng/Support/BasicBlockID.h"
#include "revng/Support/BasicBlockID/YAMLTraits.h"
#include "revng/Support/MetaAddress.h"

#include "revng/EarlyFunctionAnalysis/Generated/Early/BasicBlock.h"

namespace model {
class VerifyHelper;
}

class efa::BasicBlock : public efa::generated::BasicBlock {
public:
  using generated::BasicBlock::BasicBlock;

public:
  std::string name() const;
  BasicBlockID nextBlock() const {
    return BasicBlockID(End(), ID().inliningIndex());
  }

  [[nodiscard]] bool contains(const BasicBlockID &Other) const {
    if (Other.inliningIndex() != ID().inliningIndex())
      return false;

    MetaAddress Target = Other.start();
    MetaAddress Start = ID().start();
    MetaAddress End = this->End();
    return Start <= Target and Target < End;
  }

public:
  bool verify() const debug_function;
  bool verify(bool Assert) const debug_function;
  bool verify(model::VerifyHelper &VH) const;
};

inline auto getPrototype(ConstOrNot<model::Binary> auto &Binary,
                         MetaAddress CallerFunctionAddress,
                         const efa::BasicBlock &CallerBlock,
                         const efa::CallEdge &Edge) {
  auto &CallSitePrototypes = Binary.Functions()
                               .at(CallerFunctionAddress)
                               .CallSitePrototypes();
  auto It = CallSitePrototypes.find(CallerBlock.ID().start());
  if (It != CallSitePrototypes.end())
    return It->prototype();

  if (Edge.Type() == efa::FunctionEdgeType::FunctionCall) {
    if (not Edge.DynamicFunction().empty()) {
      auto &DFs = Binary.ImportedDynamicFunctions();
      if (auto *Prototype = DFs.at(Edge.DynamicFunction()).prototype())
        return Prototype;
    } else if (Edge.Destination().isValid()) {
      auto &LFs = Binary.Functions();
      if (auto *Prototype = LFs.at(Edge.Destination().start()).prototype())
        return Prototype;
    }
  }

  return Binary.defaultPrototype();
}

#include "revng/EarlyFunctionAnalysis/Generated/Late/BasicBlock.h"
