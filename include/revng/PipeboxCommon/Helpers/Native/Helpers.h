#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/PipeboxCommon/Helpers/Helpers.h"
#include "revng/PipeboxCommon/Helpers/Native/Container.h"

namespace revng::pypeline::helpers {

// Helper struct to unpack containers from an ArrayRef.
// To be used in conjunction with containerVectorToTuple.
template<typename C, size_t I>
struct ExtractContainerFromList<C, I, std::vector<native::Container *>> {
  static C &get(std::vector<native::Container *> &Containers) {
    return *static_cast<C *>(Containers[I]->get());
  }
};

namespace native {

struct ContainerArgument {
  llvm::StringRef Name;
  llvm::StringRef ContainerTypeName;
  Access Access;
};

} // namespace native

} // namespace revng::pypeline::helpers
