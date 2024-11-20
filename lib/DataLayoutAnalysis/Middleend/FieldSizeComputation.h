#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <cstdint>

namespace dla {

struct LayoutTypeSystemNode;
class TypeLinkTag;

} // end namespace dla

extern uint64_t getFieldSize(const dla::LayoutTypeSystemNode *Child,
                             const dla::TypeLinkTag *EdgeTag);

extern uint64_t getFieldUpperMember(const dla::LayoutTypeSystemNode *Child,
                                    const dla::TypeLinkTag *EdgeTag);
