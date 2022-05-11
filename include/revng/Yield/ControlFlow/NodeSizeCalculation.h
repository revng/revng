#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

namespace model {
class Binary;
}
namespace yield {
class Function;
}
namespace yield {
class Graph;
}

namespace yield::cfg {

struct Configuration;

void calculateNodeSizes(Graph &Graph,
                        const yield::Function &Function,
                        const model::Binary &Binary,
                        const Configuration &Configuration);

} // namespace yield::cfg
