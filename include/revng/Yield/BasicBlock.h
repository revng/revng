#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <limits>
#include <string>

#include "revng/ADT/SortedVector.h"
#include "revng/Support/BasicBlockID.h"
#include "revng/Support/BasicBlockID/YAMLTraits.h"
#include "revng/Yield/CallEdge.h"
#include "revng/Yield/FunctionEdge.h"
#include "revng/Yield/FunctionEdgeBase.h"
#include "revng/Yield/Instruction.h"

#include "revng/Yield/Generated/Early/BasicBlock.h"

namespace model {
class VerifyHelper;
}

namespace yield {

class BasicBlock : public generated::BasicBlock {
public:
  using generated::BasicBlock::BasicBlock;

public:
  void setLabel(const yield::Function &Function,
                const model::Binary &Binary,
                const model::AssemblyNameBuilder &NameBuilder);

public:
  BasicBlockID nextBlock() const {
    return BasicBlockID(End(), ID().inliningIndex());
  }

public:
  bool verify() const debug_function;
  bool verify(bool Assert) const debug_function;
  bool verify(model::VerifyHelper &VH) const;
};

template<typename T>
concept MetaAddressOrBasicBlockID = std::is_same_v<T, MetaAddress>
                                    || std::is_same_v<T, BasicBlockID>;

template<MetaAddressOrBasicBlockID T>
std::string sanitizedAddress(const T &Target, const model::Binary &Binary);

} // namespace yield

#include "revng/Yield/Generated/Late/BasicBlock.h"
