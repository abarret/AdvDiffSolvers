/////////////////////////////// INCLUDE GUARD ////////////////////////////////

#ifndef included_CCAD_app_namespaces
#define included_CCAD_app_namespaces

/////////////////////////////// INCLUDES /////////////////////////////////////

#include <ibamr/config.h>

/*!
 * Defines "using" declarations for all namespaces used in CCAD.  This
 * header file may be included in application codes, but it MUST NOT be included
 * in any other header (.h) or inline (.I) file in the library.
 */
namespace CCAD
{
}
using namespace CCAD;

namespace Eigen
{
}
using namespace Eigen;

namespace IBAMR
{
}
using namespace IBAMR;

namespace IBTK
{
}
using namespace IBTK;

namespace SAMRAI
{
namespace algs
{
}
namespace appu
{
}
namespace geom
{
}
namespace hier
{
}
namespace math
{
}
namespace mesh
{
}
namespace pdat
{
}
namespace solv
{
}
namespace tbox
{
}
namespace xfer
{
}
} // namespace SAMRAI
using namespace SAMRAI;
using namespace SAMRAI::algs;
using namespace SAMRAI::appu;
using namespace SAMRAI::geom;
using namespace SAMRAI::hier;
using namespace SAMRAI::math;
using namespace SAMRAI::mesh;
using namespace SAMRAI::pdat;
using namespace SAMRAI::solv;
using namespace SAMRAI::tbox;
using namespace SAMRAI::xfer;

namespace libMesh
{
}
using namespace libMesh;

namespace std
{
}
using namespace std;

//////////////////////////////////////////////////////////////////////////////

#endif //#ifndef included_IBAMR_app_namespaces
