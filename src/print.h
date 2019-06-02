#ifndef OU_PRINT_H
#define OU_PRINT_H
#include <iostream>
extern std::ostream& Oout;

#ifdef IS_R_BUILD
#define OPRINTF Rprintf
#endif

#endif
