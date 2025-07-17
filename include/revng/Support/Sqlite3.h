#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <memory>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include "revng/Support/Assert.h"
#include "revng/Support/Generator.h"

#include "sqlite3.h"

class Sqlite3Db;

class Sqlite3Statement {
private:
  std::unique_ptr<sqlite3_stmt, void (*)(sqlite3_stmt *)> Statement;
  friend Sqlite3Db;

  Sqlite3Statement(sqlite3 *Db, llvm::StringRef SQLStatement) :
    Statement(nullptr, &close) {
    sqlite3_stmt *Ptr = nullptr;
    int Result = sqlite3_prepare_v3(Db,
                                    SQLStatement.data(),
                                    SQLStatement.size() + 1,
                                    0,
                                    &Ptr,
                                    NULL);
    revng_assert(Result == SQLITE_OK);
    Statement = decltype(Statement)(Ptr, &close);
  }

public:
  // Bind BLOB type
  void bind(int Position, llvm::ArrayRef<char> Array) {
    int Result = sqlite3_bind_blob64(Statement.get(),
                                     Position,
                                     Array.data(),
                                     Array.size(),
                                     SQLITE_STATIC);
    revng_assert(Result == SQLITE_OK);
  }

  // Bind string
  void bind(int Position, llvm::StringRef Str) {
    int Result = sqlite3_bind_text64(Statement.get(),
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
    int Result = sqlite3_bind_int64(Statement.get(), Position, Int);
    revng_assert(Result == SQLITE_OK);
  }

  // Execute the query, the query does not return any data
  void execute() {
    int Result = sqlite3_step(Statement.get());
    revng_assert(Result == SQLITE_OK or Result == SQLITE_DONE);
  }

  // Execute the query, the query is expected to return data
  template<typename... T>
    requires(sizeof...(T) > 0)
  cppcoro::generator<std::tuple<T...>> execute() {
    while (true) {
      int Result = sqlite3_step(Statement.get());
      revng_assert(Result == SQLITE_ROW or Result == SQLITE_DONE);

      if (Result == SQLITE_DONE)
        co_return;

      revng_assert(sqlite3_column_count(Statement.get()) == sizeof...(T));
      std::integer_sequence
        IS = std::make_integer_sequence<int, sizeof...(T)>();
      co_yield ([this]<int... I>(std::integer_sequence<int, I...>) {
        return std::make_tuple(this->unpackRow<T, I>()...);
      })(IS);
    }
  }

  // Reset the statement, can be used to bind other parameters and re-run the
  // statement
  void reset() {
    sqlite3_clear_bindings(Statement.get());
    sqlite3_reset(Statement.get());
  }

private:
  static void close(sqlite3_stmt *Ptr) {
    int Result;
    Result = sqlite3_clear_bindings(Ptr);
    revng_assert(Result == SQLITE_OK);
    Result = sqlite3_reset(Ptr);
    revng_assert(Result == SQLITE_OK);
    Result = sqlite3_finalize(Ptr);
    revng_assert(Result == SQLITE_OK);
  }

  void assertType(int I, int TargetType) {
    int Type = sqlite3_column_type(Statement.get(), I);
    revng_assert(Type == TargetType);
  }

  template<std::same_as<llvm::ArrayRef<const char>> T, int I>
  llvm::ArrayRef<const char> unpackRow() {
    assertType(I, SQLITE_BLOB);
    const void *Ptr = sqlite3_column_blob(Statement.get(), I);
    int Size = sqlite3_column_bytes(Statement.get(), I);
    return llvm::ArrayRef<const char>{ static_cast<const char *>(Ptr),
                                       static_cast<size_t>(Size) };
  }

  template<std::same_as<llvm::StringRef> T, int I>
  llvm::StringRef unpackRow() {
    assertType(I, SQLITE_TEXT);
    const void *Ptr = sqlite3_column_text(Statement.get(), I);
    int Size = sqlite3_column_bytes(Statement.get(), I);
    return llvm::StringRef{ static_cast<const char *>(Ptr),
                            static_cast<size_t>(Size) };
  }
};

class Sqlite3Db {
private:
  std::unique_ptr<sqlite3, void (*)(sqlite3 *)> Connection;

public:
  Sqlite3Db(llvm::StringRef Path) : Connection(nullptr, &close) {
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

    Connection = decltype(Connection)(Ptr, &close);
  }

  Sqlite3Statement makeStatement(llvm::StringRef SQL) {
    return Sqlite3Statement(Connection.get(), SQL);
  }

private:
  static inline void close(sqlite3 *Ptr) {
    int Result = sqlite3_close_v2(Ptr);
    revng_assert(Result == SQLITE_OK);
  }
};
