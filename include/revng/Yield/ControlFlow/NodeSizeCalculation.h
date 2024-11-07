#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Yield/ControlFlow/Graph.h"

namespace model {
class Binary;
}
namespace yield {
class Function;
}

namespace yield::cfg {

struct Configuration;

void calculateNodeSizes(PreLayoutGraph &Graph,
                        const yield::Function &Function,
                        const model::Binary &Binary,
                        model::NameBuilder &NameBuilder,
                        const Configuration &Configuration);

} // namespace yield::cfg
