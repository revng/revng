--
-- This file is distributed under the MIT License. See LICENSE.md for details.
--

-- Ensure CreateFileA is only exported once (by the Fs partition).

SELECT COUNT(*)
FROM Symbol s
JOIN Library l ON l.LibraryID = s.LibraryID
JOIN Platform p ON p.PlatformID = l.PlatformID
WHERE s.Name = 'CreateFileA'
  AND p.OperatingSystem = 'Windows'
  AND l.Name NOT LIKE 'Fs%.pdb';

-- CHECK: {{^}}0{{$}}
