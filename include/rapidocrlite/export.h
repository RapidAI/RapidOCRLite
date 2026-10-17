#ifndef PPOCR_EXPORT_H_
#define PPOCR_EXPORT_H_

/* Shared-library export macro. Static consumers leave RPOCR_API empty.
   The shared build compiles with RPOCR_BUILDING_SHARED. Windows consumers
   of the DLL define RPOCR_SHARED. */
#if defined(_WIN32)
#  if defined(RPOCR_BUILDING_SHARED)
#    define RPOCR_API __declspec(dllexport)
#  elif defined(RPOCR_SHARED)
#    define RPOCR_API __declspec(dllimport)
#  else
#    define RPOCR_API
#  endif
#else
#  if defined(RPOCR_BUILDING_SHARED)
#    define RPOCR_API __attribute__((visibility("default")))
#  else
#    define RPOCR_API
#  endif
#endif

#endif
