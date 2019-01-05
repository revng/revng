#ifndef REVNG_MANGLING_H
#define REVNG_MANGLING_H

// This is very simple for now.
// In the future we might consider making it more robust using something like
// Punycode https://tools.ietf.org/html/rfc3492 , which also has the nice
// property of being deterministically reversible.
inline std::string makeCIdentifier(std::string S) {
  std::replace(S.begin(), S.end(), '.', '_');
  return S;
}

#endif // REVNG_MANGLING_H
