/// \file GzipTarFile.cpp
/// \brief Classes for reading and writing a slight variation of a .tar.gz

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Some snippets of code were adapted from llvm/llvm/lib/Support/TarWriter.cpp

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/Support/Debug.h"
#include "revng/Support/GzipStream.h"
#include "revng/Support/GzipTarFile.h"

#include "archive.h"
#include "archive_entry.h"

// Each file in an archive must be aligned to this block size.
static constexpr size_t BlockSize = 512;

// The struct definition of UstarHeader is copied from
// https://www.freebsd.org/cgi/man.cgi?query=tar&sektion=5
struct StructUstarHeader {
  char Name[100];
  char Mode[8];
  char Uid[8];
  char Gid[8];
  char Size[12];
  char Mtime[12];
  char Checksum[8];
  char TypeFlag;
  char Linkname[100];
  char Magic[6];
  char Version[2];
  char Uname[32];
  char Gname[32];
  char DevMajor[8];
  char DevMinor[8];
  char Prefix[155];
  char Pad[12];
};
static_assert(sizeof(StructUstarHeader) == BlockSize, "Invalid Ustar header");

template<size_t FieldN, size_t InputN>
  requires(InputN >= FieldN)
static void fillField(char (&In)[FieldN], const char (&Value)[InputN]) {
  std::copy_n(Value, FieldN, In);
}

union UstarHeader {
private:
  StructUstarHeader Struct;
  char Chars[BlockSize];
  uint8_t Bytes[BlockSize];

public:
  UstarHeader() : Struct() {
    fillField(Struct.Magic, "ustar"); // Ustar magic
    fillField(Struct.Version, "00"); // Ustar version
  }

  StructUstarHeader *operator->() { return &Struct; }

  // Computes a checksum for a tar header.
  void computeChecksum() {
    // Before computing a checksum, checksum field must be
    // filled with space characters.
    std::fill_n(Struct.Checksum, sizeof(Struct.Checksum), ' ');

    // Compute a checksum and set it to the checksum field.
    uint32_t Checksum = 0;
    for (size_t I = 0; I < sizeof(Bytes); I++)
      Checksum += Bytes[I];

    std::snprintf(Struct.Checksum, sizeof(Struct.Checksum), "%06o", Checksum);
  }

  llvm::StringRef stringRef() { return llvm::StringRef(Chars, sizeof(Chars)); }
};

static std::string formatPax(llvm::StringRef Key, llvm::StringRef Value) {
  // Format is '<size> <key>=<value>\n'
  std::string KV = " " + Key.str() + "=" + Value.str() + "\n";

  // The size headers includes the size indicator, so it needs to be computed
  // twice since it can overflow to the next digit
  size_t Total = std::to_string(KV.size()).size() + KV.size();
  Total = std::to_string(Total).size() + KV.size();
  return std::to_string(Total) + KV;
}

static size_t computePadding(size_t Size) {
  return BlockSize - (Size % BlockSize);
}

// Create a tar header and write it to a given output stream.
static llvm::SmallString<BlockSize * 3> writePaxHeader(llvm::StringRef Path,
                                                       size_t Size) {
  llvm::SmallString<BlockSize * 3> Result;

  // A PAX header consists of a 512-byte header followed
  // by key-value strings. First, create key-value strings.
  std::string PaxAttr = formatPax("path", Path)
                        + formatPax("size", std::to_string(Size));

  // Create a 512-byte header.
  UstarHeader Header;
  // In the context of a PAX header, the size it the size of the header section
  Header->TypeFlag = 'x'; // PAX magic
  std::snprintf(Header->Size, sizeof(Header->Size), "%011zo", PaxAttr.size());
  Header.computeChecksum();

  Result.append(Header.stringRef());
  Result.append(PaxAttr);
  Result.append(std::string(computePadding(PaxAttr.size()), '\0'));

  // The PAX header is followed by a ordinary Ustar header with no information
  // (In theory there should be to be backwards-compatible, but pax has been
  // around since 2001)
  UstarHeader TailHdr;
  fillField(TailHdr->Mode, "0000664");
  TailHdr.computeChecksum();
  Result.append(TailHdr.stringRef());

  return Result;
}

static void
writeFileHeader(llvm::raw_ostream &OS, llvm::StringRef Path, size_t Size) {
  llvm::SmallString<BlockSize * 3> FileHeader = writePaxHeader(Path, Size);
  gzipCompress(OS, { FileHeader.data(), FileHeader.size() });
}

static void compressedPadding(llvm::raw_ostream &OS, size_t Size) {
  llvm::SmallVector<char> Buffer(Size, '\0');
  return gzipCompress(OS, { Buffer.data(), Buffer.size() });
}

namespace revng {

// Append a given file to an archive.
OffsetDescriptor GzipTarWriter::append(llvm::StringRef Path,
                                       llvm::ArrayRef<char> Data) {
  revng_assert(OS != nullptr);
  revng_assert(not Filenames.contains(Path));

  OffsetDescriptor Result = { .Start = OS->tell() };
  writeFileHeader(*OS, Path, Data.size());

  Result.DataStart = OS->tell();
  gzipCompress(*OS, { Data.data(), Data.size() });

  Result.PaddingStart = OS->tell();
  if (size_t Padding = computePadding(Data.size()); Padding > 0)
    compressedPadding(*OS, Padding);

  Result.End = OS->tell();
  Filenames.insert(Path);
  return Result;
}

void GzipTarWriter::close() {
  revng_assert(OS != nullptr);
  // The tar archive needs to be ended with two blocks of zeros
  compressedPadding(*OS, BlockSize * 2);
  OS->flush();
  OS = nullptr;
}

GzipTarReader::GzipTarReader(llvm::ArrayRef<char> Ref) {
  Archive = archive_read_new();
  revng_assert(Archive != NULL);

  revng_assert(archive_read_support_filter_gzip(Archive) == ARCHIVE_OK);
  revng_assert(archive_read_support_format_tar(Archive) == ARCHIVE_OK);

  int EC = archive_read_open_memory(Archive, Ref.data(), Ref.size());
  revng_assert(EC == ARCHIVE_OK);
}

GzipTarReader::~GzipTarReader() {
  revng_assert(archive_read_free(Archive) == ARCHIVE_OK);
}

cppcoro::generator<ArchiveEntry> GzipTarReader::entries() {
  archive_entry *Entry;
  while (true) {
    int Res = archive_read_next_header(Archive, &Entry);
    if (Res == ARCHIVE_EOF)
      co_return;

    revng_assert(Res == ARCHIVE_OK);
    int64_t Size = archive_entry_size(Entry);
    revng_assert(Size >= 0);

    llvm::SmallVector<char, 0> Data;
    if (Size > 0) {
      Data.resize_for_overwrite(Size);
      size_t SizeRead = archive_read_data(Archive, Data.data(), Size);
      revng_assert(SizeRead == static_cast<size_t>(Size));
    }

    co_yield ArchiveEntry{ archive_entry_pathname(Entry), std::move(Data) };
  }
}

} // namespace revng
