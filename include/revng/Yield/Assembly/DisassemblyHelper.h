#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <map>
#include <memory>

#include "revng/Yield/Assembly/Assembly.h"

namespace detail {
class DissassemblyHelperImpl;
}
namespace model {
class Function;
}
namespace efa {
class FunctionMetadata;
}
class LLVMDisassemblerInterface;
class RawBinaryView;

class DissassemblyHelper {
private:
  std::unique_ptr<detail::DissassemblyHelperImpl> Internal;

public:
  explicit DissassemblyHelper();
  ~DissassemblyHelper();

  assembly::Function disassemble(const model::Function &Function,
                                 const efa::FunctionMetadata &Metadata,
                                 const RawBinaryView &BinaryView);

private:
  LLVMDisassemblerInterface &
  getDisassemblerFor(MetaAddressType::Values AddressType);
};
