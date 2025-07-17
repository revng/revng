#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/PipeboxCommon/Model.h"
#include "revng/PipeboxCommon/TraceRunner/Savepoint.h"
#include "revng/PipeboxCommon/TraceRunner/TraceFile.h"

namespace revng::pypeline::tracerunner {

class Runner {
public:
  Runner() = default;
  ~Runner() = default;
  Runner(const Runner &) = delete;
  Runner &operator=(const Runner &) = delete;
  Runner(Runner &&) = delete;
  Runner &operator=(Runner &&) = delete;

public:
  void run(Model &TheModel, const TraceFile &Trace, SavePoint &SP);
};

} // namespace revng::pypeline::tracerunner
