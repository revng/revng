//
// This file is distributed under the MIT License. See LICENSE.md for details.
// \file DwarfExtractor.cpp
// \brief handles the extraction of symbols from dwarf files.
// from other tools.

#include "revng/DeclarationsDb/DeclarationsDb.h"
#include "revng/Model/Binary.h"
#include "revng/Model/Type.h"
#include "revng/Support/Debug.h"
#include "revng/Support/MetaAddress.h"
#include "revng/Support/revng.h"

#include "./DwarfExtractor.h"

// LLVM includes
#include <cstdint>
#include <type_traits>

#include "llvm/ADT/Triple.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/BinaryFormat/Wasm.h"
#include "llvm/DebugInfo/DIContext.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugLoc.h"
#include "llvm/DebugInfo/DWARF/DWARFDie.h"
#include "llvm/DebugInfo/DWARF/DWARFExpression.h"
#include "llvm/DebugInfo/DWARF/DWARFFormValue.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/Object/Archive.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/MachOUniversal.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace object;
using namespace dwarf;

static Logger<> Log("extractor-dwarf");

class RegisterOp {
public:
  std::string OperationEncodingString;
  std::string RegisterName;
  int64_t Operand[2];
  bool Good;

  RegisterOp(bool IsGood = true)
      : OperationEncodingString(""), RegisterName(""), Operand{0, 0},
        Good(IsGood) {}

  std::string toString() const {
    if (!RegisterName.empty())
      return RegisterName;
    return OperationEncodingString + " " + std::to_string(Operand[0]);
  }

  void dump(raw_ostream &OS) { OS << toString(); }
};

struct Location {
private:
  RegisterOp Operation;
  uint64_t Begin{std::numeric_limits<uint64_t>::min()};
  uint64_t End{std::numeric_limits<uint64_t>::max()};

public:
  Location(RegisterOp Op) : Operation(std::move(Op)) {}

  Location(RegisterOp Op, uint64_t B, uint64_t E)
      : Operation(std::move(Op)), Begin(B), End(E) {}

  void dump(raw_ostream &OS) { Operation.dump(OS); }
  std::string getOperationAsString() const { return Operation.toString(); }
};

class LocationList {
private:
  std::vector<Location> Locations;

public:
  void addLocation(Location Location) {
    Locations.push_back(std::move(Location));
  }
  void dump(raw_ostream &OS) {
    std::for_each(Locations.begin(), Locations.end(),
                  [&OS](auto Loc) { Loc.dump(OS); });
  }
  bool hasLocation() const { return !Locations.empty(); }
  const Location &getLocation(int Index) const { return Locations[Index]; }
};

class ExtractedParameter {
private:
  std::string Name;
  LocationList Loc;

public:
  ExtractedParameter(std::string Nm, LocationList Ls)
      : Name(std::move(Nm)), Loc(std::move(Ls)) {}

  ExtractedParameter(std::string Nm) : Name(std::move(Nm)) {}
  void dump(raw_ostream &OS) {
    OS << Name << ", ";
    Loc.dump(OS);
  }

  Parameter asSavableParameter() {
    std::string Location;
    if (Loc.hasLocation())
      Location = Loc.getLocation(0).getOperationAsString();

    return Parameter(Name, Location);
  }
};

class ExtractedFunction {
private:
  std::vector<ExtractedParameter> Parameters;
  std::string Name;

public:
  uint64_t LowPc;

public:
  ExtractedFunction(std::string Nm, uint64_t Low = 0)
      : Name(std::move(Nm)), LowPc(Low) {}
  void addParameter(ExtractedParameter Param) {
    Parameters.push_back(std::move(Param));
  }
  void dump(raw_ostream &OS) {
    OS << Name;
    std::for_each(Parameters.begin(), Parameters.end(), [&OS](auto Loc) {
      OS << ", ";
      Loc.dump(OS);
    });
    OS << "\n";
  }

  bool lowInBound(uint64_t Low, uint64_t High) {
    return LowPc >= Low && LowPc <= High;
  }

  FunctionDecl asSavableFunction(StringRef LibName) {
    FunctionDecl ToReturn(Name, LibName.str());
    for (ExtractedParameter &P : Parameters)
      ToReturn.getParameters().push_back(P.asSavableParameter());
    return ToReturn;
  }
};

static std::string manageRegisterOp(uint8_t Opcode, uint64_t Operands[2],
                                    const MCRegisterInfo *MRI, bool IsEH) {
  if (MRI == nullptr)
    return "";

  uint64_t DwarfRegNum;
  unsigned OpNum = 0;

  if (Opcode == DW_OP_bregx || Opcode == DW_OP_regx)
    DwarfRegNum = Operands[OpNum++];
  else if (Opcode >= DW_OP_breg0 && Opcode < DW_OP_bregx)
    DwarfRegNum = Opcode - DW_OP_breg0;
  else
    DwarfRegNum = Opcode - DW_OP_reg0;

  auto MaybeLLVMRegNum = MRI->getLLVMRegNum(DwarfRegNum, IsEH);
  if (not MaybeLLVMRegNum)
    return "";
  int LLVMRegNum = *MaybeLLVMRegNum;

  if (const char *RegName = MRI->getName(LLVMRegNum)) {
    if ((Opcode >= DW_OP_breg0 && Opcode <= DW_OP_breg31) ||
        Opcode == DW_OP_bregx) {
      std::string ToWrite;
      llvm::raw_string_ostream Ostream(ToWrite);
      Ostream << format("%s%+" PRId64, RegName, Operands[OpNum]) << "\n";
      Ostream.flush();
      return Ostream.str();
    }

    return std::string(RegName);
  }
  return "";
}

static RegisterOp manageOperation(DWARFExpression::Operation &Op,
                                  const MCRegisterInfo *RegInfo, bool IsEH) {
  RegisterOp ToReturn;
  if (Op.isError()) {
    ToReturn.Good = false;
    return ToReturn;
  }
  uint8_t Opcode = Op.getCode();

  ToReturn.OperationEncodingString = OperationEncodingString(Opcode);
  uint64_t Operands[2] = {Op.getRawOperand(0), Op.getRawOperand(1)};

  if ((Opcode >= DW_OP_breg0 && Opcode <= DW_OP_breg31) ||
      (Opcode >= DW_OP_reg0 && Opcode <= DW_OP_reg31) ||
      Opcode == DW_OP_bregx || Opcode == DW_OP_regx) {
    ToReturn.RegisterName = manageRegisterOp(Opcode, Operands, RegInfo, IsEH);
    if (!ToReturn.RegisterName.empty())
      return ToReturn;
  }

  for (unsigned Operand = 0; Operand < 2; ++Operand) {
    unsigned Size = Op.getDescription().Op[Operand];
    unsigned Signed = Size & DWARFExpression::Operation::SignBit;

    if (Size == DWARFExpression::Operation::SizeNA)
      break;

    if (Size == DWARFExpression::Operation::SizeBlock) {
      Log << "Unsupported Size Block\n";
      ToReturn.Good = false;
      return ToReturn;
    }

    if (Signed) {
      u_int64_t Val = Op.getRawOperand(Operand);
      ToReturn.Operand[Operand] = *reinterpret_cast<int64_t *>(&Val);
    } else {
      ToReturn.Operand[Operand] = Op.getRawOperand(Operand);
    }
  }
  return ToReturn;
}

static void error(StringRef Prefix, std::error_code EC) {
  if (!EC)
    return;
  std::string Str = Prefix.str();
  Str += ": " + EC.message();
  revng_abort(Str.c_str());
}

static RegisterOp manageExp(DWARFExpression &Exp, const MCRegisterInfo *RegInfo,
                            bool IsEH)

{
  if (std::distance(Exp.begin(), Exp.end()) != 1) {
    Log << "the Expression was larger than one operation, it "
           "will be skipped. "
           "Operations Count: "
        << std::distance(Exp.begin(), Exp.end()) << "\n";
    return RegisterOp(false);
  }

  auto Op = *Exp.begin();
  RegisterOp DecodedOp = manageOperation(Op, RegInfo, IsEH);
  return DecodedOp;
}

static LocationList manageLocation(const DWARFFormValue &FormValue,
                                   DWARFUnit *U, ExtractedFunction &Fun) {
  dbg << "Handling location\n";
  DWARFContext &Ctx = U->getContext();
  const DWARFObject &Obj = Ctx.getDWARFObj();
  const MCRegisterInfo *MRI = Ctx.getRegisterInfo();
  const auto AddressSize = Obj.getAddressSize();
  LocationList List;
  if (FormValue.isFormClass(DWARFFormValue::FC_Block) ||
      FormValue.isFormClass(DWARFFormValue::FC_Exprloc)) {
    ArrayRef<uint8_t> Expr = *FormValue.getAsBlock();
    DataExtractor Data(StringRef((const char *)Expr.data(), Expr.size()),
                       Ctx.isLittleEndian(), 0);
    // WIP: DwarfFormat (third argument)
    DWARFExpression Exp(Data, U->getAddressByteSize());
    // dbg << "111111111111111\n";
    // Exp.print(llvm::errs(), {}, nullptr, nullptr, false);
    // dbg << "\n";
    RegisterOp Op = manageExp(Exp, MRI, false);
    if (Op.Good)
      List.addLocation(Location(Op));
    return List;
  }

  if (FormValue.isFormClass(DWARFFormValue::FC_SectionOffset)) {
    const DWARFSection &LocSection = Obj.getLocSection();
    const DWARFSection &LocDWOSection = Obj.getLocDWOSection();
    uint64_t Offset = *FormValue.getAsSectionOffset();
    // dbg << "Section offset: " << Offset << " into " << LocSection.Data.size()
    // << "\n";
    uint64_t BaseAddr = 0;
    auto Asd = [&](const DWARFLocationEntry &E) {
#if 1
      if (E.Kind != DW_LLE_start_end && E.Kind != DW_LLE_start_length &&
          E.Kind != DW_LLE_offset_pair)
        return true;
#endif

      auto StartAddress = BaseAddr + E.Value0;
      if (StartAddress != Fun.LowPc)
        return true;

      // dbg << "Handling DWARFLocationEntry of size " << E.Loc.size() << "\n";
      DWARFDataExtractor Extractor(
          ArrayRef<uint8_t>(E.Loc.data(), E.Loc.size()), Ctx.isLittleEndian(),
          AddressSize);
      // WIP: dwarf::DWARF_VERSION
      DWARFExpression Exp(Extractor, AddressSize);
      // WIP
      uint64_t Begin = BaseAddr /* + E.Begin */;
      uint64_t End = BaseAddr /* + E.End */;
      RegisterOp Op = manageExp(Exp, MRI, false);
      // {
      // raw_os_ostream Lol(dbg);
      // Op.dump(Lol);
      // }
      // dbg << "\n";
      if (Op.Good /* && Fun.lowInBound(Begin, End) */)
        List.addLocation(Location(Op, Begin, End));
      return true;
    };
    if (!LocSection.Data.empty()) {
      DWARFDataExtractor Data(Obj, LocSection, Ctx.isLittleEndian(),
                              Obj.getAddressSize());
      DWARFDebugLoc DebugLoc(Data);
      // DebugLoc.dump(llvm::errs(), nullptr, Obj, {}, {});

      if (auto BA = U->getBaseAddress())
        BaseAddr = BA->Address;
      // dbg << "Visiting location list\n";
      auto Error = DebugLoc.visitLocationList(&Offset, Asd);
      // dbg << "Location list visited\n";
      // WIP
      revng_assert(not Error);
#if 0
      auto LL = DebugLoc.parseOneLocationList(Data, &Offset);
      if (LL) {
        uint64_t BaseAddr = 0;
        if (auto BA = U->getBaseAddress())
          BaseAddr = BA->Address;
        for (const DWARFDebugLoc::Entry &E : LL.getPointer()->Entries) {
          DWARFDataExtractor Extractor(StringRef(E.Loc.data(), E.Loc.size()),
                                       Ctx.isLittleEndian(),
                                       AddressSize);
          DWARFExpression Exp(Extractor, dwarf::DWARF_VERSION, AddressSize);
          uint64_t begin = BaseAddr + E.Begin;
          uint64_t end = BaseAddr + E.End;

          RegisterOp Op = manageExp(Exp, MRI, false);
          if (Op.Good && Fun.lowInBound(begin, end))
            List.addLocation(Location(Op, begin, end));
        }
      } else {
        Log << "error extracting location List.";
      }
#endif
      return List;
    }
    bool UseLocLists = !U->isDWOUnit();
    // WIP
    revng_assert(UseLocLists);
    StringRef LoclistsSectionData =
        UseLocLists ? Obj.getLoclistsSection().Data :
                    /*U->getLocSectionData()*/ StringRef{};
    if (!LoclistsSectionData.empty()) {

      DWARFDataExtractor Data(LoclistsSectionData, Ctx.isLittleEndian(),
                              Obj.getAddressSize());
      DWARFDebugLoclists DebugLoclists(Data, UseLocLists ? U->getVersion() : 4);

      uint64_t BaseAddr = 0;
      if (Optional<object::SectionedAddress> BA = U->getBaseAddress())
        BaseAddr = BA->Address;

      uint64_t Offset = 0;
      auto Error = DebugLoclists.visitLocationList(&Offset, Asd);
      // WIP
      revng_assert(not Error);

#if 0
      if (LL) {
        for (const auto &E : LL->Entries) {

          DataExtractor Data(LocDWOSection.Data, Ctx.isLittleEndian(), 0);
          uint64_t begin = E.Value0;
          uint64_t end = E.Value0 + E.Value1;
          // WIP: dwarf::DWARF_VERSION
          DWARFExpression Exp(Data, AddressSize);
          RegisterOp Op = manageExp(Exp, MRI, false);
          if (Op.Good && Fun.lowInBound(begin, end))
            List.addLocation(Location(Op, begin, end));
        }
      } else {
        Log << "error extracting location List.";
      }
#endif
    }
  }
  return List;
}

static ExtractedParameter manageParameter(DWARFDie &Die, DWARFDie Param,
                                          ExtractedFunction Fun) {
  std::string ParameterName;
  if (Param.getName(DINameKind::ShortName) != nullptr)
    ParameterName = Param.getName(DINameKind::ShortName);

  for (auto &Attr : Param.attributes()) {
    if (Attr.Attr == dwarf::DW_AT_location && Attr.isValid()) {
      return ExtractedParameter(
          ParameterName, manageLocation(Attr.Value, Die.getDwarfUnit(), Fun));
    }
  }

  return ExtractedParameter(ParameterName);
}

// WIP: s/manage/handle/
static void manageSubroutine(DWARFDie Die, raw_ostream &OS, ParameterSaver &Db,
                             StringRef LibNamem) {
  StringRef RoutineName(Die.getSubroutineName(DINameKind::ShortName));

  uint64_t High;
  uint64_t Low;
  uint64_t Index;
  Die.getLowAndHighPC(Low, High, Index);

  // WIP: drop
  ExtractedFunction Fun(RoutineName.str(), Low);
  for (auto Child : Die.children()) {
    if (Child.getTag() == dwarf::Tag::DW_TAG_formal_parameter) {
      manageParameter(Die, Child, Fun);
    }
  }
}

static model::PrimitiveTypeKind::Values
dwarfEncodingToModel(uint32_t Encoding) {
  switch (Encoding) {
  case dwarf::DW_ATE_unsigned_char:
  case dwarf::DW_ATE_unsigned:
    return model::PrimitiveTypeKind::Unsigned;
  case dwarf::DW_ATE_signed_char:
  case dwarf::DW_ATE_signed:
    return model::PrimitiveTypeKind::Signed;
  case dwarf::DW_ATE_float:
    return model::PrimitiveTypeKind::Float;
  default:
    return model::PrimitiveTypeKind::Invalid;
  }
}

class DwarfImporter {
private:
  TupleTree<model::Binary> &Model;
  std::vector<std::string> LoadedFiles;
  using DwarfID = std::pair<size_t, size_t>;
  std::map<DwarfID, model::QualifiedType> DwarfToModel;

public:
  DwarfImporter(TupleTree<model::Binary> &Model) : Model(Model) {}

public:
  model::QualifiedType *findType(DwarfID ID) {
    auto It = DwarfToModel.find(ID);
    return It == DwarfToModel.end() ? nullptr : &It->second;
  }

  void recordType(DwarfID ID, const model::QualifiedType &QT) {
    revng_assert(DwarfToModel.count(ID) == 0);
    DwarfToModel.insert({ID, QT});
  }

  TupleTree<model::Binary> &getModel() { return Model; }

public:
  void import(StringRef FileName);
};

class DwarfToModelConverter {
private:
  DwarfImporter &Importer;
  TupleTree<model::Binary> &Model;
  size_t Index;
  size_t AltIndex = 0; // WIP
  DWARFContext &DICtx;
  std::set<size_t> Placeholders;

public:
  DwarfToModelConverter(DwarfImporter &Importer, DWARFContext &DICtx,
                        size_t Index, size_t AltIndex)
      : Importer(Importer), Model(Importer.getModel()), Index(Index),
        AltIndex(AltIndex), DICtx(DICtx) {}

private:
  const model::QualifiedType &record(const DWARFDie &Die,
                                     const model::TypePath &Path,
                                     bool IsNotPlaceholder) {
    return record(Die, model::QualifiedType{Path}, IsNotPlaceholder);
  }

  const model::QualifiedType &record(const DWARFDie &Die,
                                     const model::QualifiedType &QT,
                                     bool IsNotPlaceholder) {
    size_t Offset = Die.getOffset();
    if (not IsNotPlaceholder)
      Placeholders.insert(Offset);

    Importer.recordType({Index, Die.getOffset()}, QT);

    return QT;
  }

  enum TypeSearchResult { Invalid, Absent, PlaceholderType, RegularType };

  std::pair<TypeSearchResult, model::QualifiedType *>
  findType(const DWARFDie &Die) {
    return findType(Die.getOffset());
  }

  std::pair<TypeSearchResult, model::QualifiedType *>
  findType(uint64_t Offset) {
    model::QualifiedType *Result = Importer.findType({Index, Offset});
    TypeSearchResult ResultType = Invalid;
    if (Result == nullptr) {
      ResultType = Absent;
    } else {
      if (Placeholders.count(Offset) != 0)
        ResultType = PlaceholderType;
      else
        ResultType = RegularType;
    }

    return {ResultType, Result};
  }

  model::QualifiedType *findAltType(uint64_t Offset) {
    return Importer.findType({AltIndex, Offset});
  }

private:
  static bool isType(dwarf::Tag Tag) {
    switch (Tag) {
    case llvm::dwarf::DW_TAG_base_type:
    case llvm::dwarf::DW_TAG_typedef:
    case llvm::dwarf::DW_TAG_restrict_type:
    case llvm::dwarf::DW_TAG_structure_type:
    case llvm::dwarf::DW_TAG_union_type:
    case llvm::dwarf::DW_TAG_enumeration_type:
    case llvm::dwarf::DW_TAG_array_type:
    case llvm::dwarf::DW_TAG_const_type:
    case llvm::dwarf::DW_TAG_pointer_type:
    case llvm::dwarf::DW_TAG_subroutine_type:
      return true;
    default:
      return false;
    }
  }

  static bool hasModelIdentity(dwarf::Tag Tag) {
    revng_assert(isType(Tag));
    switch (Tag) {
    case llvm::dwarf::DW_TAG_base_type:
    case llvm::dwarf::DW_TAG_typedef:
    case llvm::dwarf::DW_TAG_restrict_type:
    case llvm::dwarf::DW_TAG_structure_type:
    case llvm::dwarf::DW_TAG_union_type:
    case llvm::dwarf::DW_TAG_enumeration_type:
    case llvm::dwarf::DW_TAG_subroutine_type:
      return true;
    case llvm::dwarf::DW_TAG_array_type:
    case llvm::dwarf::DW_TAG_const_type:
    case llvm::dwarf::DW_TAG_pointer_type:
      return false;
    default:
      revng_abort();
    }
  }

  bool firstPass() {
    // First pass: materialize all the unqualified types and leave them empty
    for (const auto &CU : DICtx.compile_units()) {
      for (const auto &Entry : CU->dies()) {
        DWARFDie Die = {CU.get(), &Entry};
        auto Tag = Die.getTag();

        if (not isType(Tag) or not hasModelIdentity(Tag))
          continue;

        dbg << "[first pass] Emitting for " << std::hex << Die.getOffset()
            << "\n";

        auto MaybeDeclaration = Die.find(DW_AT_declaration);
        if (MaybeDeclaration &&
            *MaybeDeclaration->getAsUnsignedConstant() != 0) {
          revng_assert(Tag == llvm::dwarf::DW_TAG_structure_type or
                       Tag == llvm::dwarf::DW_TAG_union_type or
                       Tag == llvm::dwarf::DW_TAG_enumeration_type);
          record(Die,
                 Model->getPrimitiveType(model::PrimitiveTypeKind::Void, 0),
                 true);
          continue;
        }

        switch (Tag) {
        case llvm::dwarf::DW_TAG_base_type: {
          uint8_t Size = 0;
          model::PrimitiveTypeKind::Values Kind =
              model::PrimitiveTypeKind::Invalid;

          auto MaybeByteSize = Die.find(DW_AT_byte_size);
          if (MaybeByteSize)
            Size = *MaybeByteSize->getAsUnsignedConstant();

          auto MaybeEncoding = Die.find(DW_AT_encoding);
          if (MaybeEncoding)
            Kind =
                dwarfEncodingToModel(*MaybeEncoding->getAsUnsignedConstant());

          if (Kind == model::PrimitiveTypeKind::Invalid or Size == 0 or
              not isPowerOf2_32(Size)) {
            // WIP
            continue;
          }

          record(Die, Model->getPrimitiveType(Kind, Size), true);
        } break;

        case llvm::dwarf::DW_TAG_subroutine_type:
          record(
              Die,
              Model->recordNewType(model::makeType<model::CABIFunctionType>()),
              false);
          break;
        case llvm::dwarf::DW_TAG_typedef:
        case llvm::dwarf::DW_TAG_restrict_type:
          record(Die,
                 Model->recordNewType(model::makeType<model::TypedefType>()),
                 false);
          break;
        case llvm::dwarf::DW_TAG_structure_type:
          record(Die,
                 Model->recordNewType(model::makeType<model::StructType>()),
                 false);
          break;
        case llvm::dwarf::DW_TAG_union_type:
          record(Die, Model->recordNewType(model::makeType<model::UnionType>()),
                 false);
          break;
        case llvm::dwarf::DW_TAG_enumeration_type:
          record(Die, Model->recordNewType(model::makeType<model::EnumType>()),
                 false);
          break;

        default:
          revng_abort();
        }
      }
    }
    return true;
  }

  static StringRef getName(const DWARFDie &Die) {
    auto MaybeName = Die.find(DW_AT_name);
    if (MaybeName)
      return {*MaybeName->getAsCString()};
    else
      return {};
  }

  const model::QualifiedType &getTypeImpl(const DWARFDie &Die) {
    auto MaybeType = Die.find(DW_AT_type);
    if (MaybeType) {
      if (MaybeType->getForm() == llvm::dwarf::DW_FORM_GNU_ref_alt) {
        return *notNull(findAltType(MaybeType->getRawUValue()));
      } else {
        DWARFDie InnerDie = DICtx.getDIEForOffset(*MaybeType->getAsReference());
        return resolveType(InnerDie, false);
      }
    } else {
      static model::QualifiedType Empty;
      return Empty;
    }
  }

  model::QualifiedType getTypeOrVoid(const DWARFDie &Die) {
    model::QualifiedType Result = getTypeImpl(Die);
    if (Result.UnqualifiedType.isValid()) {
      return Result;
    } else {
      return {Model->getPrimitiveType(model::PrimitiveTypeKind::Void, 0)};
    }
  }

  model::QualifiedType getType(const DWARFDie &Die) {
    model::QualifiedType Result = getTypeImpl(Die);
    revng_assert(Result.UnqualifiedType.isValid());
    return Result;
  }

  // WIP: recursive coroutine
  const model::QualifiedType &resolveType(const DWARFDie &Die,
                                          bool ResolveIfHasIdentity) {
    auto Tag = Die.getTag();
    auto [MatchType, TypePath] = findType(Die);

    // WIP: drop dbg
    dbg << "[resolveType] Emitting for " << std::hex << Die.getOffset() << "\n";

    // WIP: drop all assertions
    switch (MatchType) {
    case Absent: {
      // At this stage, all the unqualified types (i.e., those with an identity
      // in the model) should have been materialized.
      // Therefore, here we only deal with DWARF types the model represents as
      // qualifiers.
      revng_assert(not hasModelIdentity(Tag));

      auto MaybeType = Die.find(DW_AT_type);
      model::QualifiedType Type = getTypeOrVoid(Die);

      model::Qualifier NewQualifier;
      switch (Tag) {

      case llvm::dwarf::DW_TAG_const_type:
        // WIP: revng_assert(MaybeType);
        NewQualifier.Kind = model::QualifierKind::Const;
        break;

      case llvm::dwarf::DW_TAG_array_type: {
        revng_assert(MaybeType);
        std::optional<size_t> UpperBound;
        for (const DWARFDie &ChildDie : Die.children()) {
          if (ChildDie.getTag() == llvm::dwarf::DW_TAG_subrange_type) {
            auto MaybeUpperBound = ChildDie.find(DW_AT_upper_bound);
            revng_assert(not UpperBound);
            if (MaybeUpperBound)
              UpperBound = *MaybeUpperBound->getAsUnsignedConstant();
          }
        }
        NewQualifier.Kind = model::QualifierKind::Array;
        // WIP
        if (UpperBound)
          NewQualifier.Size = *UpperBound + 1;
        else
          NewQualifier.Size = 1;
      } break;

      case llvm::dwarf::DW_TAG_pointer_type: {
        auto MaybeByteSize = Die.find(DW_AT_byte_size);
        NewQualifier.Kind = model::QualifierKind::Pointer;
        NewQualifier.Size = *MaybeByteSize->getAsUnsignedConstant();
      } break;

      default:
        revng_abort();
      }

      Type.Qualifiers.push_back(NewQualifier);
      return record(Die, Type, true);
    }
    case PlaceholderType: {
      revng_assert(TypePath != nullptr);
      // This die is already present in the map. Either it has already been
      // fully imported, or it's a type with an identity on the model.
      // In the latter case, proceed only if explicitly told to do so.
      if (ResolveIfHasIdentity) {
        revng_assert(TypePath->Qualifiers.empty());
        model::Type *T = TypePath->UnqualifiedType.get();

        StringRef Name = getName(Die);

        switch (Tag) {
        case llvm::dwarf::DW_TAG_subroutine_type: {
          auto *FunctionType = cast<model::CABIFunctionType>(T);
          FunctionType->CustomName = Name;
          // WIP
          FunctionType->ABI = model::abi::SystemV_x86_64;
          FunctionType->ReturnType = getTypeOrVoid(Die);

          uint64_t Index = 0;
          for (const DWARFDie &ChildDie : Die.children()) {
            if (ChildDie.getTag() == DW_TAG_formal_parameter) {
              model::Argument &NewArgument = FunctionType->Arguments[Index];
              NewArgument.Type = getType(ChildDie);
              Index += 1;
            }
          }

        } break;

        case llvm::dwarf::DW_TAG_typedef:
        case llvm::dwarf::DW_TAG_restrict_type: {
          model::QualifiedType TargetType = getTypeOrVoid(Die);
          auto *Typedef = cast<model::TypedefType>(T);
          Typedef->CustomName = Name;
          Typedef->UnderlyingType = TargetType;

        } break;

        case llvm::dwarf::DW_TAG_structure_type: {
          auto MaybeSize = Die.find(DW_AT_byte_size);
          revng_assert(MaybeSize);

          auto *Struct = cast<model::StructType>(T);
          Struct->CustomName = Name;
          Struct->Size = *MaybeSize->getAsUnsignedConstant();

          for (const DWARFDie &ChildDie : Die.children()) {
            if (ChildDie.getTag() == DW_TAG_member) {
              // Collect offset
              auto MaybeOffset = ChildDie.find(DW_AT_data_member_location);
              revng_assert(MaybeOffset);
              auto Offset = *MaybeOffset->getAsUnsignedConstant();

              //  Create new field
              auto &Field = Struct->Fields[Offset];
              Field.CustomName = getName(ChildDie);
              Field.Type = getType(ChildDie);
            }
          }

        } break;

        case llvm::dwarf::DW_TAG_union_type: {
          auto *Union = cast<model::UnionType>(T);
          Union->CustomName = Name;

          uint64_t Index = 0;
          for (const DWARFDie &ChildDie : Die.children()) {
            if (ChildDie.getTag() == DW_TAG_member) {
              // Create new field
              auto &Field = Union->Fields[Index];
              Field.CustomName = getName(ChildDie);
              Field.Type = getType(ChildDie);

              // Increment union index
              Index += 1;
            }
          }

        } break;

        case llvm::dwarf::DW_TAG_enumeration_type: {
          auto *Enum = cast<model::EnumType>(T);
          Enum->CustomName = Name;
          model::QualifiedType UnderlyingType = getType(Die);
          revng_assert(UnderlyingType.Qualifiers.empty());
          Enum->UnderlyingType =
              Model->getTypePath(UnderlyingType.UnqualifiedType.get());

          for (const DWARFDie &ChildDie : Die.children()) {
            if (ChildDie.getTag() == DW_TAG_enumerator) {
              // Collect value
              auto MaybeValue = ChildDie.find(DW_AT_const_value);
              revng_assert(MaybeValue);
              auto Value = *MaybeValue->getAsUnsignedConstant();

              // Create new entry
              StringRef EntryName = getName(ChildDie);

              // If it's the first time, set CustomName, otherwise, introduce
              // an alias
              auto It = Enum->Entries.find(Value);
              if (It == Enum->Entries.end()) {
                auto &Entry = Enum->Entries[Value];
                Entry.CustomName = EntryName;
              } else {
                It->Aliases.insert(EntryName);
              }
            }
          }

        } break;

        default:
          revng_abort();
        }
      }
    } break;

    case RegularType:
      revng_assert(TypePath != nullptr);

    default:
      revng_abort();
    }

    return *TypePath;
  }

  void secondPass() {
    for (const auto &CU : DICtx.compile_units()) {
      for (const auto &Entry : CU->dies()) {
        DWARFDie Die = {CU.get(), &Entry};
        if (not isType(Die.getTag()))
          continue;
        resolveType(Die, true);
      }
    }
  }

  void thirdPass() {
    for (const auto &CU : DICtx.compile_units()) {
      for (const auto &Entry : CU->dies()) {
        DWARFDie Die = {CU.get(), &Entry};

        if (Die.getTag() != DW_TAG_subprogram)
          continue;

        // Create function type
        auto Path =
            Model->recordNewType(model::makeType<model::CABIFunctionType>());
        auto *FunctionType = cast<model::CABIFunctionType>(Path.get());
        FunctionType->CustomName = getName(Die);
        FunctionType->ABI = model::abi::SystemV_x86_64;
        FunctionType->ReturnType = getTypeOrVoid(Die);

        uint64_t Index = 0;
        for (const DWARFDie &ChildDie : Die.children()) {
          if (ChildDie.getTag() == DW_TAG_formal_parameter) {
            model::Argument &NewArgument = FunctionType->Arguments[Index];
            NewArgument.CustomName = getName(ChildDie);
            NewArgument.Type = getType(ChildDie);
            Index += 1;
          }
        }

        // Create actual function
        model::DynamicFunction &Function =
            Model->DynamicFunctions[FunctionType->CustomName.str().str()];
        Function.CustomName = FunctionType->CustomName;
        Function.SymbolName = FunctionType->CustomName.str();
        Function.Prototype = Path;
      }
    }
  }

  void fix() {
    // TODO: collapse uint8_t typedefs into the primitive type

    std::set<std::string> UsedNames;

    for (auto &Type : Model->Types) {
      model::Type *T = Type.get();
      if (isa<model::PrimitiveType>(T)) {
        UsedNames.insert(T->name().str().str());
      }
    }

    for (auto &Type : Model->Types) {
      model::Type *T = Type.get();
      if (isa<model::PrimitiveType>(T))
        continue;

      std::string Name = T->name().str().str();
      while (UsedNames.count(Name) != 0) {
        Name += "_";
      }

      // Rename
      upcast(T, [&Name](auto &Upcasted) {
        using UpcastedType = std::remove_cvref_t<decltype(Upcasted)>;
        if constexpr (not std::is_same_v<model::PrimitiveType, UpcastedType>) {
          Upcasted.CustomName = Name;
        } else {
          revng_abort();
        }
      });

      // Record new name
      UsedNames.insert(Name);
    }
  }

public:
  void run() {
    firstPass();
    secondPass();
    thirdPass();
    fix();
    Model.serialize(llvm::errs());
    Model->verify(true);
  }
};

template <typename T>
ArrayRef<uint8_t> getSectionsContents(StringRef Name, T &ELF) {
  auto MaybeSections = ELF.sections();
  if (not MaybeSections)
    return {};

  for (const auto &Section : *MaybeSections) {
    auto MaybeName = ELF.getSectionName(Section);
    if (MaybeName) {
      if (MaybeName and *MaybeName == Name) {
        auto MaybeContents = ELF.getSectionContents(Section);
        if (MaybeContents)
          return *MaybeContents;
      }
    }
  }

  return {};
}

static StringRef getAltDebugLinkFileName(Binary *B) {

  auto Handler = [&](auto *ELFObject) -> StringRef {
    const auto &ELF = ELFObject->getELFFile();
    ArrayRef<uint8_t> Contents = getSectionsContents(".gnu_debugaltlink", ELF);

    if (Contents.size() == 0)
      return {};

    // TODO: improve accuracy
    // Extract path name and ignore everything after \0
    return StringRef(reinterpret_cast<const char *>(Contents.data()));
  };

  StringRef AltDebugLinkPath;
  if (auto *ELF = dyn_cast<ELF32BEObjectFile>(B)) {
    AltDebugLinkPath = Handler(ELF);
  } else if (auto *ELF = dyn_cast<ELF64BEObjectFile>(B)) {
    AltDebugLinkPath = Handler(ELF);
  } else if (auto *ELF = dyn_cast<ELF32LEObjectFile>(B)) {
    AltDebugLinkPath = Handler(ELF);
  } else if (auto *ELF = dyn_cast<ELF64LEObjectFile>(B)) {
    AltDebugLinkPath = Handler(ELF);
  } else {
    revng_abort();
  }

  return llvm::sys::path::filename(AltDebugLinkPath);
}

inline void DwarfImporter::import(StringRef FileName) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> BuffOrErr =
      MemoryBuffer::getFileOrSTDIN(FileName);
  error(FileName, BuffOrErr.getError());
  std::unique_ptr<MemoryBuffer> Buffer = std::move(BuffOrErr.get());
  Expected<std::unique_ptr<Binary>> BinOrErr = object::createBinary(*Buffer);
  error(FileName, errorToErrorCode(BinOrErr.takeError()));

  if (auto *ELF = dyn_cast<ELF32BEObjectFile>(BinOrErr->get())) {
    // Check if we already loaded the alt debug info file
    size_t AltIndex = -1;
    StringRef AltDebugLinkFileName = getAltDebugLinkFileName(ELF);
    if (AltDebugLinkFileName.size() > 0) {
      auto Begin = LoadedFiles.begin();
      auto End = LoadedFiles.end();
      auto It = std::find(Begin, End, AltDebugLinkFileName);
      if (It != End)
        AltIndex = It - Begin;
    }

    auto TheDWARFContext = DWARFContext::create(*ELF);
    DwarfToModelConverter Converter(*this, *TheDWARFContext,
                                    LoadedFiles.size() - 1, AltIndex);
    Converter.run();
  }

  LoadedFiles.push_back(sys::path::filename(FileName).str());
}

static bool dumpObjectFile(ObjectFile &Obj, DWARFContext &DICtx,
                           std::string FileName, raw_ostream &OS,
                           ParameterSaver &Db, StringRef LibName) {
  logAllUnhandledErrors(DICtx.loadRegisterInfo(Obj), errs(), FileName + ": ");

  TupleTree<model::Binary> Model;
  DwarfImporter Importer(Model);
  Importer.import(FileName);

  for (const auto &CU : DICtx.compile_units()) {
    for (const auto &Entry : CU->dies()) {
      DWARFDie Die = {CU.get(), &Entry};

      if (Die.isSubprogramDIE()) {
        manageSubroutine(Die, OS, Db, LibName);
      }
    }
  }

  return true;
}

static bool handleBuffer(StringRef Filename, MemoryBufferRef Buffer,
                         raw_ostream &OS, ParameterSaver &Db,
                         StringRef LibName) {
  Expected<std::unique_ptr<Binary>> BinOrErr = object::createBinary(Buffer);
  error(Filename, errorToErrorCode(BinOrErr.takeError()));

  bool Result = true;
  if (auto *Obj = dyn_cast<ObjectFile>(BinOrErr->get())) {
    std::unique_ptr<DWARFContext> DICtx = DWARFContext::create(*Obj);
    Result = dumpObjectFile(*Obj, *DICtx, Filename.str(), OS, Db, LibName);
  }

  return Result;
}

// WIP: return model
bool extractDwarf(StringRef Filename, raw_ostream &OS, ParameterSaver &Db,
                  StringRef LibName) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> BuffOrErr =
      MemoryBuffer::getFileOrSTDIN(Filename);
  error(Filename, BuffOrErr.getError());
  std::unique_ptr<MemoryBuffer> Buffer = std::move(BuffOrErr.get());
  return handleBuffer(Filename, *Buffer, OS, Db, LibName);
}

// WIP:
// TODO: when loading a binary
// 1. Load any available DWARF in the binary itself
// 2. Parse .note.gnu.build-id, .gnu_debugaltlink and .gnu_debuglink
// 3. Load from the following paths:
//    - /usr/lib/debug/.build-id/ab/cdef1234.debug
//    - /usr/bin/ls.debug
//    - /usr/bin/.debug/ls.debug
//    - /usr/lib/debug/usr/bin/ls.debug
//    In turn, parse .gnu_debugaltlink (and .gnu_debuglink?)
// 2. Parse DT_NEEDED
// 3. Look for each library in ld.so.conf directories
// 4. Go to 1
// https://sourceware.org/gdb/onlinedocs/gdb/Separate-Debug-Files.html
