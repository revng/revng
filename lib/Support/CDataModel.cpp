//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Support/CDataModel.h"

CDataModel CDataModel::getDefaultDataModel(uint8_t PointerSize) {
  CDataModel DM;
  DM.PointerSize = PointerSize;
  DM.setStandardTypeSize(CStandardType::Char, 1);
  DM.setStandardTypeSize(CStandardType::Short, 2);
  DM.setStandardTypeSize(CStandardType::Int, PointerSize <= 2 ? 2 : 4);
  DM.setStandardTypeSize(CStandardType::Long, PointerSize <= 4 ? 4 : 8);
  DM.setStandardTypeSize(CStandardType::LongLong, 8);
  DM.setStandardTypeSize(CStandardType::Float, 4);
  DM.setStandardTypeSize(CStandardType::Double, 8);
  DM.setStandardTypeSize(CStandardType::LongDouble, 8);
  return DM;
}

bool CDataModel::verify() const {
  if (getCharSize() < 1)
    return false;

  if (getShortSize() < 2 or getCharSize() > getShortSize())
    return false;

  if (getIntSize() < 2 or getShortSize() > getIntSize())
    return false;

  if (getLongSize() < 4 or getIntSize() > getLongSize())
    return false;

  if (getLongLongSize() < 8 or getLongSize() > getLongLongSize())
    return false;

  if (getFloatSize() > getDoubleSize())
    return false;

  if (getDoubleSize() > getLongDoubleSize())
    return false;

  return true;
}

void CDataModel::dump() const {
  dbg << "pointer ....... " << getPointerSize() << "\n";
  dbg << "char .......... " << getCharSize() << "\n";
  dbg << "short ......... " << getShortSize() << "\n";
  dbg << "int ........... " << getIntSize() << "\n";
  dbg << "long .......... " << getLongSize() << "\n";
  dbg << "long long ..... " << getLongLongSize() << "\n";
  dbg << "float ......... " << getFloatSize() << "\n";
  dbg << "double ........ " << getDoubleSize() << "\n";
  dbg << "long double ... " << getLongDoubleSize() << "\n";
}
