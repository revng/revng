//
// This file is distributed under the MIT License. See LICENSE.md for details.
// \file DwarfExtractor.cpp
// \brief handles the extraction of symbols from dwarf files.
// from other tools.

#include "./DwarfExtractor.h"
#include "revng/DeclarationsDb/DeclarationsDb.h"
#include "revng/Model/Binary.h"
#include "revng/Support/Debug.h"
#include "revng/Support/MetaAddress.h"

// LLVM includes
#include "llvm/ADT/Triple.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/DebugInfo/DIContext.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugLoc.h"
#include "llvm/DebugInfo/DWARF/DWARFExpression.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/Object/Archive.h"
#include "llvm/Object/MachOUniversal.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>

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
    // dbg << "Section offset: " << Offset << " into " << LocSection.Data.size() << "\n";
    uint64_t BaseAddr = 0;
    auto Asd = [&](const DWARFLocationEntry &E) {
      #if 1
      if (E.Kind != DW_LLE_start_end
          && E.Kind != DW_LLE_start_length
          && E.Kind != DW_LLE_offset_pair)
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

static void manageSubroutine(DWARFDie Die, raw_ostream &OS, ParameterSaver &Db,
                             StringRef LibNamem, TupleTree<model::Binary> &Model) {
  StringRef RoutineName(Die.getSubroutineName(DINameKind::ShortName));
  
  uint64_t High;
  uint64_t Low;
  uint64_t Index;
  Die.getLowAndHighPC(Low, High, Index);

  // WIP: arch? thumb?
  auto EntryPC = MetaAddress::fromPC(llvm::Triple::x86_64, Low);
  model::Function &F = Model->Functions[EntryPC];

  // WIP: drop
  ExtractedFunction Fun(RoutineName.str(), Low);
  for (auto Child : Die.children()) {
    if (Child.getTag() == dwarf::Tag::DW_TAG_formal_parameter) {
      Fun.addParameter(manageParameter(Die, Child, Fun));
    }
  }

}

static bool dumpObjectFile(ObjectFile &Obj, DWARFContext &DICtx,
                           std::string Filename, raw_ostream &OS,
                           ParameterSaver &Db, StringRef LibName) {
  logAllUnhandledErrors(DICtx.loadRegisterInfo(Obj), errs(), Filename + ": ");

  TupleTree<model::Binary> Model;

  for (const auto &CU : DICtx.compile_units()) {
    for (const auto &Entry : CU->dies()) {
      DWARFDie Die = { CU.get(), &Entry };
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
