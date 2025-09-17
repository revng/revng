#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/PipeboxCommon/Helpers/Native/Registry.h"
#include "revng/Support/Sqlite3.h"

namespace revng::pypeline::tracerunner {

class SavePoint {
private:
  Sqlite3Db DB;

public:
  SavePoint(llvm::StringRef StringPath) : DB(StringPath){};

  void save(const revng::pypeline::helpers::native::Container &Cont,
            llvm::StringRef ContainerName,
            uint64_t SavepointID,
            llvm::StringRef Hash,
            llvm::ArrayRef<const ObjectID> Objects);

  void load(revng::pypeline::helpers::native::Container &Cont,
            llvm::StringRef ContainerName,
            uint64_t SavepointID,
            llvm::StringRef Hash,
            llvm::ArrayRef<const ObjectID> Objects);
};

} // namespace revng::pypeline::tracerunner
