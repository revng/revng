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
namespace assembly {
struct BasicBlock;
struct Function;
} // namespace assembly

namespace yield {

namespace plain {

std::string assembly(const assembly::BasicBlock &BasicBlock,
                     const efa::FunctionMetadata &Metadata,
                     const model::Binary &Binary);
std::string assembly(const assembly::Function &Function,
                     const efa::FunctionMetadata &Metadata,
                     const model::Binary &Binary);

} // namespace plain

} // namespace yield
