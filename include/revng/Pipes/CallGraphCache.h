#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <string>

#include "revng/Pipeline/YamlizableGlobal.h"
#include "revng/Support/YAMLTraits.h"
#include "revng/TupleTree/TupleLikeTraits.h"

/* TUPLE-TREE-YAML
name: CallGraphCache
doc: don't know
type: struct
fields:
  - name: Name
    type: std::string
key:
  - Name
TUPLE-TREE-YAML */

namespace pipeline {

class Context;
}

#include "revng/Pipes/Generated/Early/CallGraphCache.h"

class revng::pipes::CallGraphCache
  : public revng::pipes::generated::CallGraphCache {
public:
  static constexpr const char *GlobalName = "CallGraphCache.yml";
  using revng::pipes::generated::CallGraphCache::CallGraphCache;

  static CallGraphCache &fromContext(pipeline::Context &Ctx);
  static const CallGraphCache &fromContext(const pipeline::Context &Ctx);
};

#include "revng/Pipes/Generated/Late/CallGraphCache.h"

namespace revng::pipes {
using CallGraphGlobal = pipeline::YamlizableGlobal<CallGraphCache>;

}
