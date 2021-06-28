#include "ibamr/config.h"

#include "CCAD/LSCartGridFunction.h"

#include "ibamr/app_namespaces.h"

#include <SAMRAI_config.h>

namespace CCAD
{
/////////////////////////////// PUBLIC ///////////////////////////////////////

LSCartGridFunction::LSCartGridFunction(const string& object_name) : CartGridFunction(object_name)
{
    // intentionally blank
    return;
} // LSCartGridFunction

} // namespace CCAD
