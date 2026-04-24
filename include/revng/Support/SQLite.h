#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <memory>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include "revng/ADT/CUniquePtr.h"
#include "revng/ADT/CompilationTime.h"
#include "revng/Support/Assert.h"
#include "revng/Support/Generator.h"

#include "sqlite3.h"

namespace sqlite {

namespace detail {
inline void destroyStatement(sqlite3_stmt *Ptr) {
  int Result;
  Result = sqlite3_clear_bindings(Ptr);
  revng_assert(Result == SQLITE_OK);
  Result = sqlite3_reset(Ptr);
  revng_assert(Result == SQLITE_OK);
  Result = sqlite3_finalize(Ptr);
  revng_assert(Result == SQLITE_OK);
}
} // namespace detail

class Statement {
private:
  CUniquePtr<detail::destroyStatement> SQLiteStatement;

public:
  Statement(sqlite3 *Db, llvm::StringRef SQLStatement) {
    sqlite3_stmt *Ptr = nullptr;
    int Result = sqlite3_prepare_v3(Db,
                                    SQLStatement.data(),
                                    SQLStatement.size() + 1,
                                    0,
                                    &Ptr,
                                    NULL);
    revng_assert(Result == SQLITE_OK);
    SQLiteStatement.reset(Ptr);
  }

public:
  // Bind BLOB type
  void bind(int Position, llvm::ArrayRef<char> Array) {
    int Result = sqlite3_bind_blob64(SQLiteStatement.get(),
                                     Position,
                                     Array.data(),
                                     Array.size(),
                                     SQLITE_STATIC);
    revng_assert(Result == SQLITE_OK);
  }

  // Bind string
  void bind(int Position, llvm::StringRef Str) {
    int Result = sqlite3_bind_text64(SQLiteStatement.get(),
                                     Position,
                                     Str.data(),
                                     Str.size(),
                                     SQLITE_STATIC,
                                     SQLITE_UTF8);
    revng_assert(Result == SQLITE_OK);
  }

  // Bind integer
  template<typename T>
    requires(std::same_as<T, uint64_t> or std::same_as<T, int>)
  void bind(int Position, T Int) {
    int Result = sqlite3_bind_int64(SQLiteStatement.get(), Position, Int);
    revng_assert(Result == SQLITE_OK);
  }

  // Bind a vector of strings at consecutive positions starting from 1
  void bind(const std::vector<std::string> &Values) {
    for (size_t Index = 0; Index < Values.size(); ++Index)
      bind(static_cast<int>(Index + 1), llvm::StringRef(Values[Index]));
  }

  // Execute the query, the query does not return any data
  void execute() {
    int Result = sqlite3_step(SQLiteStatement.get());
    revng_assert(Result == SQLITE_OK or Result == SQLITE_DONE);
  }

  // Execute the query, the query is expected to return data
  template<typename... T>
    requires(sizeof...(T) > 0)
  cppcoro::generator<std::tuple<T...>> execute() {
    while (true) {
      int Result = sqlite3_step(SQLiteStatement.get());
      revng_assert(Result == SQLITE_ROW or Result == SQLITE_DONE);

      if (Result == SQLITE_DONE)
        co_return;

      revng_assert(sqlite3_column_count(SQLiteStatement.get()) == sizeof...(T));
      co_yield compile_time::callWithIndexSequence<
        sizeof...(T)>([this]<size_t... I>() {
        return std::make_tuple(this->unpackRow<T, I>()...);
      });
    }
  }

  // Reset the statement, can be used to bind other parameters and re-run the
  // statement
  void reset() {
    sqlite3_clear_bindings(SQLiteStatement.get());
    sqlite3_reset(SQLiteStatement.get());
  }

private:
  void assertType(int I, int TargetType) {
    int Type = sqlite3_column_type(SQLiteStatement.get(), I);
    revng_assert(Type == TargetType);
  }

  template<std::same_as<llvm::ArrayRef<char>> T, int I>
  llvm::ArrayRef<char> unpackRow() {
    assertType(I, SQLITE_BLOB);
    const void *Ptr = sqlite3_column_blob(SQLiteStatement.get(), I);
    int Size = sqlite3_column_bytes(SQLiteStatement.get(), I);
    return llvm::ArrayRef<char>{ static_cast<const char *>(Ptr),
                                 static_cast<size_t>(Size) };
  }

  template<std::same_as<llvm::StringRef> T, int I>
  llvm::StringRef unpackRow() {
    assertType(I, SQLITE_TEXT);
    const void *Ptr = sqlite3_column_text(SQLiteStatement.get(), I);
    int Size = sqlite3_column_bytes(SQLiteStatement.get(), I);
    return llvm::StringRef{ static_cast<const char *>(Ptr),
                            static_cast<size_t>(Size) };
  }

  template<std::same_as<int64_t> T, int I>
  int64_t unpackRow() {
    assertType(I, SQLITE_INTEGER);
    return sqlite3_column_int64(SQLiteStatement.get(), I);
  }
};

class Database {
private:
  CUniquePtr<sqlite3_close_v2, SQLITE_OK> Connection;

public:
  Database(llvm::StringRef Path) {
    sqlite3 *Ptr = nullptr;
    int Result;

    if (Path == ":memory:")
      Result = sqlite3_open_v2("",
                               &Ptr,
                               SQLITE_OPEN_MEMORY | SQLITE_OPEN_PRIVATECACHE,
                               NULL);
    else
      Result = sqlite3_open_v2(Path.str().c_str(),
                               &Ptr,
                               SQLITE_OPEN_READWRITE,
                               NULL);
    revng_assert(Result == SQLITE_OK);

    Connection.reset(Ptr);
  }

  Statement makeStatement(llvm::StringRef SQL) {
    return Statement(Connection.get(), SQL);
  }
};

/// Build a SQL IN clause with N positional placeholders starting
/// at position 1: "?1,?2,...,?N".
inline std::string buildInClause(size_t Count) {
  std::string Result;
  for (size_t Index = 0; Index < Count; ++Index) {
    if (Index > 0)
      Result += ",";
    Result += "?" + std::to_string(Index + 1);
  }
  return Result;
}

} // namespace sqlite
