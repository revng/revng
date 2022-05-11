#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

namespace model {
class Binary;
}
namespace yield {
class BasicBlock;
class Function;
} // namespace yield

namespace yield::cfg {

template<bool ShouldMergeFallthroughTargets>
const yield::BasicBlock *detectFallthrough(const yield::BasicBlock &BasicBlock,
                                           const yield::Function &Function,
                                           const model::Binary &Binary);

} // namespace yield::cfg
