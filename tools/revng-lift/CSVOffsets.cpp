/// \file CSVOffsets.cpp

// Local libraries includes
#include "revng/Support/Debug.h"

// Local includes
#include "CSVOffsets.h"

void writeToLog(Logger<true> &L, const CSVOffsets &O, int /*Ignore*/) {
  L << "Kind: " << CSVOffsets::toString(O.OffsetKind);
  L << " Offsets = { ";
  for (const auto &Offset : O)
    L << Offset << ' ';
  L << "}";
}
