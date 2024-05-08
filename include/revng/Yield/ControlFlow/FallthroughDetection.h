#pragma once

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

#include "llvm/ADT/SmallVector.h"

namespace model {
class Binary;
}
namespace yield {
class BasicBlock;
class Function;
} // namespace yield

namespace yield::cfg {

const yield::BasicBlock *detectFallthrough(const yield::BasicBlock &BasicBlock,
                                           const yield::Function &Function,
                                           const model::Binary &Binary);

llvm::SmallVector<const yield::BasicBlock *, 8>
labeledBlock(const yield::BasicBlock &BasicBlock,
             const yield::Function &Function,
             const model::Binary &Binary);

} // namespace yield::cfg
