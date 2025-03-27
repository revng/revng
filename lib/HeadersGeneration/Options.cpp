//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/HeadersGeneration/Options.h"

using namespace llvm::cl;

namespace revng::options {

opt<bool> EnableStackFrameInlining("enable-stack-frame-inlining",
                                   desc("Enable printing the definition "
                                        "of a function's stack type inside "
                                        "the function's body."),
                                   init(false));

} // namespace revng::options
