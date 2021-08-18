//
// This file is distributed under the MIT License. See LICENSE.md for details.
// \file main.cpp
// \brief Dumps call information present in PDB files.
//
// Local includes
#include "revng/Support/Assert.h"

#include "ParametersExtractor.h"

// LLVM includes
#include "llvm/DebugInfo/CodeView/CVRecord.h"
#include "llvm/DebugInfo/CodeView/CodeView.h"
#include "llvm/DebugInfo/CodeView/Formatters.h"
#include "llvm/DebugInfo/CodeView/LazyRandomTypeCollection.h"
#include "llvm/DebugInfo/CodeView/SymbolRecord.h"
#include "llvm/DebugInfo/CodeView/TypeRecord.h"
#include "llvm/DebugInfo/PDB/Native/PDBStringTable.h"
#include "llvm/Support/FormatVariadic.h"

using namespace llvm;
using namespace llvm::codeview;
using namespace llvm::pdb;

// mapping taken from:
// https://github.com/microsoft/microsoft-pdb/blob/master/include/cvconst.h
static std::string formatRegisterId(ulittle16_t Id, CPUType Cpu) {
  switch (Cpu) {
  case (CPUType::X64):
    switch (Id) {
    case 1:
      return "AL";
    case 2:
      return "CL";
    case 3:
      return "DL";
    case 4:
      return "BL";
    case 5:
      return "AH";
    case 6:
      return "CH";
    case 7:
      return "DH";
    case 8:
      return "BH";
    case 9:
      return "AX";
    case 10:
      return "CX";
    case 11:
      return "DX";
    case 12:
      return "BX";
    case 13:
      return "SP";
    case 14:
      return "BP";
    case 15:
      return "SI";
    case 16:
      return "DI";
    case 17:
      return "EAX";
    case 18:
      return "ECX";
    case 19:
      return "EDX";
    case 20:
      return "EBX";
    case 21:
      return "ESP";
    case 22:
      return "EBP";
    case 23:
      return "ESI";
    case 24:
      return "EDI";
    case 25:
      return "ES";
    case 26:
      return "CS";
    case 27:
      return "SS";
    case 28:
      return "DS";
    case 29:
      return "FS";
    case 30:
      return "GS";
    case 32:
      return "FLAGS";
    case 33:
      return "RIP";
    case 34:
      return "EFLAGS";
    case 80:
      return "CR0";
    case 81:
      return "CR1";
    case 82:
      return "CR2";
    case 83:
      return "CR3";
    case 84:
      return "CR4";
    case 88:
      return "CR8";
    case 90:
      return "DR0";
    case 91:
      return "DR1";
    case 92:
      return "DR2";
    case 93:
      return "DR3";
    case 94:
      return "DR4";
    case 95:
      return "DR5";
    case 96:
      return "DR6";
    case 97:
      return "DR7";
    case 98:
      return "DR8";
    case 99:
      return "DR9";
    case 100:
      return "DR10";
    case 101:
      return "DR11";
    case 102:
      return "DR12";
    case 103:
      return "DR13";
    case 104:
      return "DR14";
    case 105:
      return "DR15";
    case 110:
      return "GDTR";
    case 111:
      return "GDTL";
    case 112:
      return "IDTR";
    case 113:
      return "IDTL";
    case 114:
      return "LDTR";
    case 115:
      return "TR";
    case 128:
      return "ST0";
    case 129:
      return "ST1";
    case 130:
      return "ST2";
    case 131:
      return "ST3";
    case 132:
      return "ST4";
    case 133:
      return "ST5";
    case 134:
      return "ST6";
    case 135:
      return "ST7";
    case 136:
      return "CTRL";
    case 137:
      return "STAT";
    case 138:
      return "TAG";
    case 139:
      return "FPIP";
    case 140:
      return "FPCS";
    case 141:
      return "FPDO";
    case 142:
      return "FPDS";
    case 143:
      return "ISEM";
    case 144:
      return "FPEIP";
    case 145:
      return "FPEDO";
    case 146:
      return "MM0";
    case 147:
      return "MM1";
    case 148:
      return "MM2";
    case 149:
      return "MM3";
    case 150:
      return "MM4";
    case 151:
      return "MM5";
    case 152:
      return "MM6";
    case 153:
      return "MM7";
    case 154:
      return "XMM0";
    case 155:
      return "XMM1";
    case 156:
      return "XMM2";
    case 157:
      return "XMM3";
    case 158:
      return "XMM4";
    case 159:
      return "XMM5";
    case 160:
      return "XMM6";
    case 161:
      return "XMM7";
    case 162:
      return "XMM0_0";
    case 163:
      return "XMM0_1";
    case 164:
      return "XMM0_2";
    case 165:
      return "XMM0_3";
    case 166:
      return "XMM1_0";
    case 167:
      return "XMM1_1";
    case 168:
      return "XMM1_2";
    case 169:
      return "XMM1_3";
    case 170:
      return "XMM2_0";
    case 171:
      return "XMM2_1";
    case 172:
      return "XMM2_2";
    case 173:
      return "XMM2_3";
    case 174:
      return "XMM3_0";
    case 175:
      return "XMM3_1";
    case 176:
      return "XMM3_2";
    case 177:
      return "XMM3_3";
    case 178:
      return "XMM4_0";
    case 179:
      return "XMM4_1";
    case 180:
      return "XMM4_2";
    case 181:
      return "XMM4_3";
    case 182:
      return "XMM5_0";
    case 183:
      return "XMM5_1";
    case 184:
      return "XMM5_2";
    case 185:
      return "XMM5_3";
    case 186:
      return "XMM6_0";
    case 187:
      return "XMM6_1";
    case 188:
      return "XMM6_2";
    case 189:
      return "XMM6_3";
    case 190:
      return "XMM7_0";
    case 191:
      return "XMM7_1";
    case 192:
      return "XMM7_2";
    case 193:
      return "XMM7_3";
    case 194:
      return "XMM0L";
    case 195:
      return "XMM1L";
    case 196:
      return "XMM2L";
    case 197:
      return "XMM3L";
    case 198:
      return "XMM4L";
    case 199:
      return "XMM5L";
    case 200:
      return "XMM6L";
    case 201:
      return "XMM7L";
    case 202:
      return "XMM0H";
    case 203:
      return "XMM1H";
    case 204:
      return "XMM2H";
    case 205:
      return "XMM3H";
    case 206:
      return "XMM4H";
    case 207:
      return "XMM5H";
    case 208:
      return "XMM6H";
    case 209:
      return "XMM7H";
    case 211:
      return "MXCSR";
    case 220:
      return "EMM0L";
    case 221:
      return "EMM1L";
    case 222:
      return "EMM2L";
    case 223:
      return "EMM3L";
    case 224:
      return "EMM4L";
    case 225:
      return "EMM5L";
    case 226:
      return "EMM6L";
    case 227:
      return "EMM7L";
    case 228:
      return "EMM0H";
    case 229:
      return "EMM1H";
    case 230:
      return "EMM2H";
    case 231:
      return "EMM3H";
    case 232:
      return "EMM4H";
    case 233:
      return "EMM5H";
    case 234:
      return "EMM6H";
    case 235:
      return "EMM7H";
    case 236:
      return "MM00";
    case 237:
      return "MM01";
    case 238:
      return "MM10";
    case 239:
      return "MM11";
    case 240:
      return "MM20";
    case 241:
      return "MM21";
    case 242:
      return "MM30";
    case 243:
      return "MM31";
    case 244:
      return "MM40";
    case 245:
      return "MM41";
    case 246:
      return "MM50";
    case 247:
      return "MM51";
    case 248:
      return "MM60";
    case 249:
      return "MM61";
    case 250:
      return "MM70";
    case 251:
      return "MM71";
    case 260:
      return "XMM8_0";
    case 261:
      return "XMM8_1";
    case 262:
      return "XMM8_2";
    case 263:
      return "XMM8_3";
    case 264:
      return "XMM9_0";
    case 265:
      return "XMM9_1";
    case 266:
      return "XMM9_2";
    case 267:
      return "XMM9_3";
    case 268:
      return "XMM10_0";
    case 269:
      return "XMM10_1";
    case 270:
      return "XMM10_2";
    case 271:
      return "XMM10_3";
    case 272:
      return "XMM11_0";
    case 273:
      return "XMM11_1";
    case 274:
      return "XMM11_2";
    case 275:
      return "XMM11_3";
    case 276:
      return "XMM12_0";
    case 277:
      return "XMM12_1";
    case 278:
      return "XMM12_2";
    case 279:
      return "XMM12_3";
    case 280:
      return "XMM13_0";
    case 281:
      return "XMM13_1";
    case 282:
      return "XMM13_2";
    case 283:
      return "XMM13_3";
    case 284:
      return "XMM14_0";
    case 285:
      return "XMM14_1";
    case 286:
      return "XMM14_2";
    case 287:
      return "XMM14_3";
    case 288:
      return "XMM15_0";
    case 289:
      return "XMM15_1";
    case 290:
      return "XMM15_2";
    case 291:
      return "XMM15_3";
    case 292:
      return "XMM8L";
    case 293:
      return "XMM9L";
    case 294:
      return "XMM10L";
    case 295:
      return "XMM11L";
    case 296:
      return "XMM12L";
    case 297:
      return "XMM13L";
    case 298:
      return "XMM14L";
    case 299:
      return "XMM15L";
    case 300:
      return "XMM8H";
    case 301:
      return "XMM9H";
    case 302:
      return "XMM10H";
    case 303:
      return "XMM11H";
    case 304:
      return "XMM12H";
    case 305:
      return "XMM13H";
    case 306:
      return "XMM14H";
    case (307):
      return "XMM15H";
    case 308:
      return "EMM8L";
    case 309:
      return "EMM9L";
    case 310:
      return "EMM10L";
    case 311:
      return "EMM11L";
    case 312:
      return "EMM12L";
    case 313:
      return "EMM13L";
    case 314:
      return "EMM14L";
    case 315:
      return "EMM15L";
    case 316:
      return "EMM8H";
    case 317:
      return "EMM9H";
    case 318:
      return "EMM10H";
    case 319:
      return "EMM11H";
    case 320:
      return "EMM12H";
    case 321:
      return "EMM13H";
    case 322:
      return "EMM14H";
    case 323:
      return "EMM15H";
    case 324:
      return "SIL";
    case 325:
      return "DIL";
    case 326:
      return "BPL";
    case 327:
      return "SPL";
    case 328:
      return "RAX";
    case 329:
      return "RBX";
    case 330:
      return "RCX";
    case 331:
      return "RDX";
    case 332:
      return "RSI";
    case 333:
      return "RDI";
    case 334:
      return "RBP";
    case 335:
      return "RSP";
    case 336:
      return "R8";
    case 337:
      return "R9";
    case 338:
      return "R10";
    case 339:
      return "R11";
    case 340:
      return "R12";
    case 341:
      return "R13";
    case 342:
      return "R14";
    case 343:
      return "R15";
    case 344:
      return "R8B";
    case 345:
      return "R9B";
    case 346:
      return "R10B";
    case 347:
      return "R11B";
    case 348:
      return "R12B";
    case 349:
      return "R13B";
    case 350:
      return "R14B";
    case 351:
      return "R15B";
    case 352:
      return "R8W";
    case 353:
      return "R9W";
    case 354:
      return "R10W";
    case 355:
      return "R11W";
    case 356:
      return "R12W";
    case 357:
      return "R13W";
    case 358:
      return "R14W";
    case 359:
      return "R15W";
    case 360:
      return "R8D";
    case 361:
      return "R9D";
    case 362:
      return "R10D";
    case 363:
      return "R11D";
    case 364:
      return "R12D";
    case 365:
      return "R13D";
    case 366:
      return "R14D";
    case 367:
      return "R15D";

    default:
      revng_abort("unknown register");
      return "unknown register";
    }

  default:
    revng_abort("unknown register");
    return "unknown register";
  }

  revng_abort("unknown register");
  return "unknown register";
}

Error ParametersExtractor::visitSymbolBegin(codeview::CVSymbol &Record) {
  return visitSymbolBegin(Record, 0);
}

Error ParametersExtractor::visitSymbolBegin(codeview::CVSymbol &Record,
                                            uint32_t Offset) {
  return Error::success();
}

Error ParametersExtractor::visitSymbolEnd(CVSymbol &Record) {
  return Error::success();
}

std::string
ParametersExtractor::typeOrIdIndex(codeview::TypeIndex TI, bool IsType) const {
  auto &Container = IsType ? Types : Ids;
  StringRef Name = Container.getTypeName(TI);

  return Name.str();
}

Error ParametersExtractor::visitKnownRecord(CVSymbol &CVR, LocalSym &Local) {

  auto iter = FunctionMap.find(LastFunctionName);
  if (iter != FunctionMap.end()) {

    iter->second.getParameters().push_back(Parameter(Local.Name.str()));
    LastParamName = Local.Name.str();
  }

  return Error::success();
}

Error ParametersExtractor::visitKnownRecord(CVSymbol &CVR, ProcSym &Proc) {

  FunctionMap.emplace(Proc.Name.str(), FunctionDecl(Proc.Name.str(), LibName));

  LastFunctionName = Proc.Name.str();

  return Error::success();
}
Error ParametersExtractor::visitKnownRecord(CVSymbol &CVR, Compile3Sym &Symb) {
  CompilationCPU = Symb.Machine;
  return Error::success();
}

Error ParametersExtractor::visitKnownRecord(CVSymbol &CVR,
                                            DefRangeRegisterSym &Symb) {

  auto Function = FunctionMap.find(LastFunctionName);
  if (Function == FunctionMap.end())
    return Error::success();

  auto &v = Function->second.getParameters();
  auto iter = std::find_if(v.begin(), v.end(), [this](auto f) {
    return f.name() == LastParamName;
  });

  if (iter == v.end())
    return Error::success();
  std::string name = formatRegisterId(Symb.Hdr.Register, CompilationCPU);
  iter->setLocation(name);
  return Error::success();
}

Error ParametersExtractor::visitKnownRecord(CVSymbol &CVR,
                                            DefRangeRegisterRelSym &Symb) {

  auto Function = FunctionMap.find(LastFunctionName);
  if (Function == FunctionMap.end())
    return Error::success();

  auto &v = Function->second.getParameters();
  auto iter = std::find_if(v.begin(), v.end(), [this](auto f) {
    return f.name() == LastParamName;
  });

  if (iter == v.end())
    return Error::success();
  std::string name = formatRegisterId(Symb.Hdr.Register, CompilationCPU);
  name += "+" + std::to_string(Symb.Hdr.BasePointerOffset);
  iter->setLocation(name);
  return Error::success();
}
