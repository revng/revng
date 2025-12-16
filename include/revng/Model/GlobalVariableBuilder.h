#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/Type.h"
#include "revng/Model/TypeDefinition.h"
#include "revng/Support/MetaAddress.h"
#include "revng/Support/MetaAddress/MetaAddressRange.h"

namespace model {
class Binary;
class StructDefinition;

/// Creates a global variable (i.e., a field in the Type of model::Segment), but
/// only if there's a hole at the requested address.
///
/// \note If a Segment does not a have a type, we bail out.
class GlobalVariableBuilder {
private:
  model::Binary &Binary;
  std::map<const model::TypeDefinition *, uint64_t> Instances;

public:
  GlobalVariableBuilder(model::Binary &Binary);

public:
  [[nodiscard]] bool insert(const MetaAddress &Address,
                            model::UpcastableType &&Type);
};

} // namespace model
