//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Pypeline/Helpers/Native/Registry.h"
#include "revng/Pypeline/TraceRunner/Savepoint.h"

using namespace revng::pypeline::helpers::native;

static std::string InsertSQL = (llvm::Twine{}
                                + "REPLACE INTO objects(savepoint_id, "
                                + "container_id, configuration_hash, "
                                + "object_id, content)"
                                + " VALUES (?, ?, ?, ?, ?)")
                                 .str();

static std::string SelectSQL = (llvm::Twine{}
                                + "SELECT object_id, content FROM objects "
                                + "WHERE savepoint_id = ? "
                                + "AND container_id = ? "
                                + "AND configuration_hash = ? "
                                + "AND object_id IN ")
                                 .str();

namespace revng::pypeline::tracerunner {

void SavePoint::save(const Container &Cont,
                     llvm::StringRef ContainerName,
                     uint64_t SavepointID,
                     llvm::StringRef Hash,
                     llvm::ArrayRef<const ObjectID> Objects) {
  auto Serialized = Cont.serialize(Objects);
  Sqlite3Statement St = DB.makeStatement(InsertSQL);
  for (const auto &[Obj, Buf] : Serialized) {
    std::string ObjSerialized = Obj.serialize();
    St.reset();
    St.bind(1, SavepointID);
    St.bind(2, ContainerName);
    St.bind(3, Hash);
    St.bind(4, ObjSerialized);
    St.bind(5, Buf.ref());
    St.execute();
  }
}

void SavePoint::load(Container &Cont,
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

  std::map<const ObjectID *, llvm::ArrayRef<const char>> Input;
  for (auto &[Key, BufferRef] :
       Statement.execute<llvm::StringRef, llvm::ArrayRef<const char>>()) {
    ObjectID Obj;
    llvm::cantFail(Obj.deserialize(Key));

    Input.clear();
    Input[&Obj] = BufferRef;
    Cont.deserialize(Input);
  }
}

} // namespace revng::pypeline::tracerunner
