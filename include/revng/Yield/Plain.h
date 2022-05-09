#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <string>

class MetaAddress;
namespace model {
class Binary;
}
namespace yield {
class Function;
}

namespace yield {

namespace plain {

std::string functionAssembly(const yield::Function &InternalFunction,
                             const model::Binary &Binary);
std::string controlFlowNode(const MetaAddress &BasicBlockAddress,
                            const yield::Function &Function,
                            const model::Binary &Binary);

} // namespace plain

} // namespace yield
