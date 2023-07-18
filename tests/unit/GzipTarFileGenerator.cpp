/// \file GzipTarFileGenerator.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//
// rcc-ignore: initrevng

#include "revng/Support/Debug.h"
#include "revng/Support/GzipTarFile.h"

int main(int argc, char *argv[]) {
  if (argc == 1 or argc % 2 == 0) {
    dbg << "Usage: " << argv[0] << " <filename 1> <contents 1>...\n";
    return 1;
  }

  revng::GzipTarWriter Writer(llvm::outs());

  for (int I = 1; I < argc; I += 2) {
    llvm::StringRef Data(argv[I + 1]);
    Writer.append(argv[I], { Data.data(), Data.size() });
  }

  Writer.close();
  return 0;
}
