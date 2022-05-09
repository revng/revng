#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <string>

namespace model {
class Binary;
}
namespace efa {
class FunctionMetadata;
}
namespace yield {
class BasicBlock;
class Function;
} // namespace yield

namespace yield {

namespace plain {

std::string assembly(const yield::BasicBlock &BasicBlock,
                     const efa::FunctionMetadata &Metadata,
                     const model::Binary &Binary);
std::string assembly(const yield::Function &Function,
                     const efa::FunctionMetadata &Metadata,
                     const model::Binary &Binary);

} // namespace plain

} // namespace yield
