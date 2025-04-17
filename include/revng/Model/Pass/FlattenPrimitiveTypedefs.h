#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

namespace model {

class Binary;

/// This looks for typedefs of common primitive types and replaces them with
/// their type.
void flattenPrimitiveTypedefs(TupleTree<model::Binary> &Binary);

} // namespace model
