//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/PipeboxCommon/Helpers/Native/Registry.h"
#include "revng/PipeboxCommon/TraceRunner/Savepoint.h"

using namespace revng::pypeline::helpers::native;

static std::string InsertSQL = "REPLACE INTO objects(savepoint_id, "
                               "container_id, configuration_hash, object_id, "
                               "content) VALUES (?, ?, ?, ?, ?)";

static std::string SelectSQL = "SELECT object_id, content FROM objects WHERE "
                               "savepoint_id = ? AND container_id = ? AND "
                               "configuration_hash = ? AND object_id IN ";

namespace revng::pypeline::tracerunner {

void SavePoint::save(const Container &TheContainer,
                     llvm::StringRef ContainerName,
                     uint64_t SavepointID,
                     llvm::StringRef Hash,
                     llvm::ArrayRef<const ObjectID> Objects) {
  auto Serialized = TheContainer.serialize(Objects);
  Sqlite3Statement Statement = DB.makeStatement(InsertSQL);
  for (const auto &[Obj, Buf] : Serialized) {
    std::string ObjSerialized = Obj.serialize();
    Statement.reset();
    Statement.bind(1, SavepointID);
    Statement.bind(2, ContainerName);
    Statement.bind(3, Hash);
    Statement.bind(4, ObjSerialized);
    Statement.bind(5, Buf.data());
    Statement.execute();
  }
}

void SavePoint::load(Container &TheContainer,
                     llvm::StringRef ContainerName,
                     uint64_t SavepointID,
                     llvm::StringRef Hash,
                     llvm::ArrayRef<const ObjectID> Objects) {
  std::vector<std::string> ObjectIDLiterals;
  for (const ObjectID &Obj : Objects)
    ObjectIDLiterals.push_back("'" + Obj.serialize() + "'");

  std::string InsertQuery = SelectSQL + "(" + llvm::join(ObjectIDLiterals, ",")
                            + ")";
  Sqlite3Statement Statement = DB.makeStatement(InsertQuery);
  Statement.bind(1, SavepointID);
  Statement.bind(2, ContainerName);
  Statement.bind(3, Hash);

  std::map<const ObjectID *, llvm::ArrayRef<char>> Input;
  for (auto &[Key, BufferRef] :
       Statement.execute<llvm::StringRef, llvm::ArrayRef<char>>()) {
    ObjectID Obj = llvm::cantFail(ObjectID::deserialize(Key));

    Input.clear();
    Input[&Obj] = BufferRef;
    TheContainer.deserialize(Input);
  }
}

} // namespace revng::pypeline::tracerunner
