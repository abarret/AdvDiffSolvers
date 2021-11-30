#include "ibamr/config.h"

#include "ADS/LSCartGridFunction.h"
#include "ADS/app_namespaces.h"

#include <SAMRAI_config.h>

namespace ADS
{
/////////////////////////////// PUBLIC ///////////////////////////////////////

LSCartGridFunction::LSCartGridFunction(const string& object_name) : CartGridFunction(object_name)
{
    // intentionally blank
    return;
} // LSCartGridFunction

} // namespace ADS
