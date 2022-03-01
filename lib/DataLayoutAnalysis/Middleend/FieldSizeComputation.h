#pragma once

//
// Copyright (c) rev.ng Srls. See LICENSE.md for details.
//

namespace dla {

struct LayoutTypeSystemNode;
class TypeLinkTag;

} // end namespace dla

extern uint64_t getFieldSize(const dla::LayoutTypeSystemNode *Child,
                             const dla::TypeLinkTag *EdgeTag);

extern uint64_t getFieldUpperMember(const dla::LayoutTypeSystemNode *Child,
                                    const dla::TypeLinkTag *EdgeTag);
