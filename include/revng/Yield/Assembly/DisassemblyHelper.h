#pragma once

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

#include <map>
#include <memory>

#include "revng/Yield/Function.h"

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

  yield::Function disassemble(const model::Function &Function,
                              const efa::FunctionMetadata &Metadata,
                              const RawBinaryView &BinaryView,
                              const model::Binary &Binary);

private:
  LLVMDisassemblerInterface &
  getDisassemblerFor(MetaAddressType::Values AddressType,
                     const model::DisassemblyConfiguration &);
};
