#pragma once

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
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
PreLayoutGraph extractFromInternal(const yield::Function &Function,
                                   const model::Binary &Binary,
                                   const Configuration &Configuration);

} // namespace yield::cfg
