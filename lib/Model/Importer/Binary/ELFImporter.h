#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Object/ELFObjectFile.h"

#include "revng/Model/Importer/Binary/BinaryImporterHelper.h"
#include "revng/Model/RawBinaryView.h"
#include "revng/Support/MetaAddress.h"

#include "DwarfReader.h"
#include "SegmentImportHelpers.h"

extern Logger<> ELFImporterLog;

namespace {

class FilePortion {
private:
  const RawBinaryView &File;
  bool HasAddress;
  bool HasSize;
  uint64_t Size;
  MetaAddress Address;

public:
  FilePortion(const RawBinaryView &File);

public:
  MetaAddress address() const { return Address; }
  uint64_t size() const { return Size; }

public:
  void setAddress(MetaAddress Address);
  void setSize(uint64_t Size);

  MetaAddress addressAtOffset(uint64_t Offset);

  template<typename T>
  MetaAddress addressAtIndex(uint64_t Index);

  bool isAvailable() const;
  bool isExact() const;

  llvm::StringRef extractString() const;

  template<typename T>
  llvm::ArrayRef<T> extractAs() const;
  llvm::ArrayRef<uint8_t> extractData() const;
};

class ELFImporterBase {
public:
  virtual ~ELFImporterBase() = default;
  virtual llvm::Error import(const ImporterOptions &Options) = 0;
};

template<typename T, bool HasAddend>
class ELFImporter : public BinaryImporterHelper, public ELFImporterBase {
private:
  RawBinaryView File;
  TupleTree<model::Binary> &Model;
  const llvm::object::ELFObjectFileBase &TheBinary;

  std::optional<MetaAddress> EHFrameHdrAddress;
  std::optional<MetaAddress> DynamicAddress;

private:
  llvm::SmallVector<DataSymbol, 32> DataSymbols;

protected:
  std::optional<uint64_t> SymbolsCount;
  std::unique_ptr<FilePortion> DynstrPortion;
  std::unique_ptr<FilePortion> DynsymPortion;
  std::unique_ptr<FilePortion> ReldynPortion;
  std::unique_ptr<FilePortion> RelpltPortion;
  std::unique_ptr<FilePortion> GotPortion;

public:
  ELFImporter(TupleTree<model::Binary> &Model,
              const llvm::object::ELFObjectFileBase &TheBinary) :
    File(*Model, toArrayRef(TheBinary.getData())),
    Model(Model),
    TheBinary(TheBinary) {}

private:
  using Elf_Rel = llvm::object::Elf_Rel_Impl<T, HasAddend>;
  using Elf_Rel_Array = llvm::ArrayRef<Elf_Rel>;
  using ConstElf_Shdr = const typename llvm::object::ELFFile<T>::Elf_Shdr;

public:
  llvm::Error import(const ImporterOptions &Options) override;

private:
  MetaAddress getGenericPointer(Pointer Ptr) const {
    if (not Ptr.isIndirect())
      return Ptr.value();

    auto MaybePointer = File.getFromAddressOn(Ptr.value());
    if (not MaybePointer)
      return MetaAddress::invalid();

    return fromGeneric(::readPointer<T>(MaybePointer->data()));
  }

  MetaAddress getCodePointer(Pointer Ptr) const {
    using namespace model::Architecture;
    auto Architecture = Model->Architecture();
    return this->getGenericPointer(Ptr).toPC(toLLVMArchitecture(Architecture));
  }

  /// Parse the .eh_frame_hdr section to obtain the address and the number of
  /// FDEs in .eh_frame
  ///
  /// \return a pair containing a (possibly invalid) pointer to the .eh_frame
  ///         section and the count of FDEs in the .eh_frame_hdr section (which
  ///         should match the number of FDEs in .eh_frame)
  std::pair<MetaAddress, uint64_t> ehFrameFromEhFrameHdr();

  /// Parse the .eh_frame section to collect all the landing pads
  ///
  /// \param EHFrameAddress the address of the .eh_frame section
  /// \param FDEsCount the count of FDEs in the .eh_frame section
  /// \param EHFrameSize the size of the .eh_frame section
  ///
  /// \note Either \p FDEsCount or \p EHFrameSize have to be specified
  void parseEHFrame(MetaAddress EHFrameAddress,
                    llvm::Optional<uint64_t> FDEsCount,
                    llvm::Optional<uint64_t> EHFrameSize);

  /// Parse an LSDA to collect its landing pads
  ///
  /// \param FDEStart the start address of the FDE to which this LSDA is
  ///        associated
  /// \param LSDAAddress the address of the target LSDA
  void parseLSDA(MetaAddress FDEStart, MetaAddress LSDAAddress);

  void
  parseSymbols(llvm::object::ELFFile<T> &TheELF, ConstElf_Shdr *SectionHeader);

  void parseProgramHeaders(llvm::object::ELFFile<T> &TheELF);

  void parseDynamicSymbol(llvm::object::Elf_Sym_Impl<T> &Symbol,
                          llvm::StringRef Dynstr);

  void findMissingTypes(llvm::object::ELFFile<T> &TheELF,
                        const ImporterOptions &Options);

protected:
  template<typename Q>
  using SmallVectorImpl = llvm::SmallVectorImpl<Q>;

  /// Register a label for each input relocation
  void registerRelocations(Elf_Rel_Array Relocations,
                           const FilePortion &Dynsym,
                           const FilePortion &Dynstr);

  void parseDynamicTag(uint64_t Tag,
                       MetaAddress Relocated,
                       SmallVectorImpl<uint64_t> &NeededLibraryNameOffsets,
                       uint64_t Val);

  /// Parse architecture dynamic tags.
  virtual void
  parseTargetDynamicTags(uint64_t Tag,
                         MetaAddress Relocated,
                         SmallVectorImpl<uint64_t> &NeededLibraryNameOffsets,
                         uint64_t Val) {}
};

} // end anonymous namespace
