//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/HeadersGeneration/Options.h"

using namespace llvm::cl;

namespace revng::options {

opt<bool> DisableTypeInlining("disable-type-inlining",
                              desc("Disable printing struct, union and enum "
                                   "types inline in their parent types if "
                                   "they're only used once. This also disables "
                                   "printing stack types definitions inline "
                                   "in the function body."),
                              init(false));

opt<bool> DisableStackFrameInlining("disable-stack-frame-inlining",
                                    desc("Disable printing the definition "
                                         "of the function stack type inside "
                                         "its body."),
                                    init(false));

} // namespace revng::options
