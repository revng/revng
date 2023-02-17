#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <string>

class BasicBlockID;
namespace model {
class Binary;
}

namespace yield {

class Function;

namespace crossrelations {

class CrossRelations;

} // namespace crossrelations

namespace svg {

namespace detail {

using CrossRelations = yield::crossrelations::CrossRelations;

} // namespace detail

std::string controlFlowGraph(const yield::Function &InternalFunction,
                             const model::Binary &Binary);
std::string callGraph(const detail::CrossRelations &CrossRelationTree,
                      const model::Binary &Binary);
std::string callGraphSlice(const BasicBlockID &SlicePoint,
                           const detail::CrossRelations &CrossRelationTree,
                           const model::Binary &Binary);

} // namespace svg

} // namespace yield
