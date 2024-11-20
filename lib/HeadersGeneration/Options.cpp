//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/HeadersGeneration/Options.h"

using namespace llvm::cl;

namespace revng::options {

/// TODO: type inlining is currently broken in rare cases involving recursive
///       array struct fields. That's why it's disabled by default.
///       When the bugs a fixed, turn this option into `disable-type-inlining`
///       and reverse it's using.
opt<bool> EnableTypeInlining("enable-type-inlining",
                             desc("Enable printing struct, union and enum "
                                  "types inline in their parent types if "
                                  "they're only used once. This also enables "
                                  "printing stack types definitions inline "
                                  "in "
                                  "the function body."),
                             init(false));

opt<bool> DisableStackFrameInlining("disable-stack-frame-inlining",
                                    desc("Disable printing the definition "
                                         "of the function stack type inside "
                                         "its body."),
                                    init(false));

} // namespace revng::options
