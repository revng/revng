#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Support/MemoryBuffer.h"

#include "revng/ADT/CUniquePtr.h"
#include "revng/Support/Generator.h"

#include "archive.h"

namespace revng {

enum class TarFormat {
  Plain,
  Gzip,
};

class TarReader {
private:
  CUniquePtr<archive_read_free, ARCHIVE_OK> Archive;

public:
  TarReader(llvm::ArrayRef<const char> Ref, TarFormat Format = TarFormat::Gzip);
  TarReader(const llvm::MemoryBuffer &Buffer,
            TarFormat Format = TarFormat::Gzip) :
    TarReader({ Buffer.getBufferStart(), Buffer.getBufferSize() }, Format){};

public:
  struct Entry {
    std::string Filename;
    llvm::SmallVector<char> Data;
  };

  cppcoro::generator<Entry> entries();
};

class TarWriter {
private:
  llvm::raw_ostream &OS;
  CUniquePtr<archive_write_free, ARCHIVE_OK> Archive;

public:
  TarWriter(llvm::raw_ostream &OS, TarFormat Format = TarFormat::Plain);
  ~TarWriter();

public:
  void addMember(llvm::StringRef Filename, llvm::ArrayRef<const char> Buffer);

  void addMember(llvm::StringRef Filename, llvm::ArrayRef<char> Buffer) {
    addMember(Filename,
              llvm::ArrayRef<const char>{ Buffer.data(), Buffer.size() });
  }

  void addMember(llvm::StringRef Filename, llvm::MemoryBuffer &Buffer) {
    addMember(Filename,
              llvm::ArrayRef<const char>{ Buffer.getBufferStart(),
                                          Buffer.getBufferSize() });
  }

private:
  static long
  archiveWrite(archive *Ptr, void *ClassPtr, const void *Data, size_t Size);
};

} // namespace revng
