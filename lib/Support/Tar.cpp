/// \file Tar.cpp
/// \brief Classes for reading and writing tar files with libarchive

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "archive_entry.h"

#include "llvm/Support/raw_ostream.h"

#include "revng/Support/Assert.h"
#include "revng/Support/Tar.h"

namespace revng {

TarReader::TarReader(llvm::ArrayRef<char> Ref, TarFormat Format) {
  Archive.reset(archive_read_new());
  revng_assert(Archive.get() != NULL);
  int RC = ARCHIVE_OK;

  if (Format == TarFormat::Gzip) {
    RC = archive_read_support_filter_gzip(Archive.get());
    revng_assert(RC == ARCHIVE_OK);
  }

  RC = archive_read_support_format_tar(Archive.get());
  revng_assert(RC == ARCHIVE_OK);

  RC = archive_read_open_memory(Archive.get(), Ref.data(), Ref.size());
  revng_assert(RC == ARCHIVE_OK);
}

cppcoro::generator<TarReader::Entry> TarReader::entries() {
  archive_entry *Entry = nullptr;
  while (true) {
    int Res = archive_read_next_header(Archive.get(), &Entry);
    if (Res == ARCHIVE_EOF)
      co_return;

    revng_assert(Res == ARCHIVE_OK);
    int64_t Size = archive_entry_size(Entry);
    revng_assert(Size >= 0);

    llvm::SmallVector<char, 0> Data;
    if (Size > 0) {
      Data.resize_for_overwrite(Size);
      size_t SizeRead = archive_read_data(Archive.get(), Data.data(), Size);
      revng_assert(SizeRead == static_cast<size_t>(Size));
    }

    co_yield TarReader::Entry{ archive_entry_pathname(Entry), std::move(Data) };
  }
}

TarWriter::TarWriter(llvm::raw_ostream &OS, TarFormat Format) : OS(OS) {
  Archive.reset(archive_write_new());
  revng_assert(Archive.get() != NULL);
  int RC = ARCHIVE_OK;

  if (Format == TarFormat::Gzip) {
    RC = archive_write_add_filter_gzip(Archive.get());
    revng_assert(RC == ARCHIVE_OK);
  }

  // pax_restricted is "lazy" pax, it will use the traditional ustar format if
  // possible or PAX if necessary
  RC = archive_write_set_format_pax_restricted(Archive.get());
  revng_assert(RC == ARCHIVE_OK);

  // Since we need to write to `OS` we use the advanced `archive_write_open2`
  // function. This allows specifying function handlers for when the archive is
  // opened, closed and importantly, written to, instead of writing directly to
  // a file. We set all the handlers to NULL except the
  // `archive_write_callback`, which is given `archiveWrite`.
  // This function will be called on each write with the following:
  // * An opaque pointer of type `void*`, which we assign here the value `this`
  // * A pointer + size which represents the data to be written
  RC = archive_write_open2(Archive.get(),
                           this,
                           NULL,
                           &archiveWrite,
                           NULL,
                           NULL);
  revng_assert(RC == ARCHIVE_OK);
}

TarWriter::~TarWriter() {
  int RC = archive_write_close(Archive.get());
  revng_assert(RC == ARCHIVE_OK);
  OS.flush();
}

void TarWriter::addMember(llvm::StringRef Filename,
                          llvm::ArrayRef<char> Buffer) {
  CUniquePtr<archive_entry_free, ARCHIVE_OK> Entry(archive_entry_new());

  archive_entry_set_pathname(Entry.get(), Filename.str().c_str());
  archive_entry_set_size(Entry.get(), Buffer.size());
  archive_entry_set_filetype(Entry.get(), AE_IFREG);
  archive_entry_set_perm(Entry.get(), 0644);

  int RC = archive_write_header(Archive.get(), Entry.get());
  revng_assert(RC == ARCHIVE_OK);

  const char *Data = Buffer.data();
  size_t Remaining = Buffer.size();
  while (Remaining > 0) {
    size_t ToWrite = Remaining > INT_MAX ? INT_MAX : Remaining;
    archive_write_data(Archive.get(), Data, ToWrite);
    Remaining -= ToWrite;
    Data += ToWrite;
  }
}

long TarWriter::archiveWrite(archive *Ptr,
                             void *ClassPtr,
                             const void *Data,
                             size_t Size) {
  TarWriter *Class = static_cast<TarWriter *>(ClassPtr);
  Class->OS.write(static_cast<const char *>(Data), Size);
  return Size;
}

} // namespace revng
