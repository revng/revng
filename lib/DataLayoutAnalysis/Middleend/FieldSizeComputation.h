#pragma once

//
// Copyright (c) rev.ng Labs Srl. See LICENSE.md for details.
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
