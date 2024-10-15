#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <string>

#include "revng/PTML/Tag.h"

class BasicBlockID;
namespace model {
class Binary;
}

namespace ptml {
class MarkupBuilder;
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

std::string controlFlowGraph(const ::ptml::MarkupBuilder &B,
                             const yield::Function &InternalFunction,
                             const model::Binary &Binary);
std::string callGraph(const ::ptml::MarkupBuilder &B,
                      const detail::CrossRelations &CrossRelationTree,
                      const model::Binary &Binary);
std::string callGraphSlice(const ::ptml::MarkupBuilder &B,
                           std::string_view SlicePoint,
                           const detail::CrossRelations &CrossRelationTree,
                           const model::Binary &Binary);

} // namespace svg

} // namespace yield
