#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Support/SHA256.h"

#include "revng/PipeboxCommon/Common.h"
#include "revng/Support/TarIO.h"

namespace revng::pypeline {

class BinariesContainer {
public:
  static constexpr llvm::StringRef Name = "BinariesContainer";
  static constexpr Kind Kind = Kinds::Binary;

private:
  struct File {
    std::string Filename;
    Buffer Contents;
  };
  std::vector<File> Files;

public:
  BinariesContainer() = default;

public:
  std::set<ObjectID> objects() const {
    if (Files.empty())
      return std::set<ObjectID>{};
    else
      return std::set{ ObjectID() };
  }

  void deserialize(const std::map<const ObjectID *, llvm::ArrayRef<const char>>
                     Data) {
    if (Data.size() == 0)
      return;

    revng_assert(Data.size() == 1);
    auto [Key, Value] = *Data.begin();
    revng_assert(Key->kind() == Kind);

    Files.clear();
    TarReader Reader(Value, TarFormat::Plain);
    for (TarReader::Entry Entry : Reader.entries()) {
      std::string Hash = hash(Entry.Data);
      revng_assert(Entry.Filename == Hash);
      Files.push_back({ std::move(Entry.Filename), std::move(Entry.Data) });
    }
  }

  std::map<ObjectID, Buffer>
  serialize(const std::vector<const ObjectID *> Objects) const {
    if (Objects.size() == 0)
      return {};

    revng_assert(Objects.size() == 1 and Objects[0]->kind() == Kind);

    Buffer Output;
    {
      llvm::raw_svector_ostream OS(Output.data());
      TarWriter Writer(OS, TarFormat::Plain);
      for (const File &Entry : Files)
        Writer.addMember(Entry.Filename, Entry.Contents.data());
    }

    return { { ObjectID(), Output } };
  }

  bool verify() const { return true; }

public:
  llvm::ArrayRef<char> getFile(size_t Index) const {
    revng_assert(Index < Files.size());
    return Files[Index].Contents.data();
  }

private:
  static std::string hash(llvm::ArrayRef<char> Data) {
    const uint8_t *DataPtr = reinterpret_cast<const uint8_t *>(Data.data());
    std::array<uint8_t, 32> Hash = llvm::SHA256::hash({ DataPtr, Data.size() });
    return llvm::toHex(Hash, true);
  }
};

} // namespace revng::pypeline
