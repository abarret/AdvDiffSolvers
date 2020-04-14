######################################################################
## Here specify the location of the IBAMR source and the location
## where IBAMR has been built.
IBAMR_SRC_DIR   = ${HOME}/sfw/ibamr/IBAMR
IBAMR_BUILD_DIR = ${HOME}/sfw/ibamr/ibamr-objs-opt
######################################################################
## Include variables specific to the particular IBAMR build.
include $(IBAMR_BUILD_DIR)/config/make.inc

######################################################################
## Build the IB tester application.
##
## main driver is in main.cpp
##
## PDIM = 2 implies two spatial dimensions
OBJS = src/SemiLagrangianAdvIntegrator.o
#main.o
CPPFLAGS += -Iinclude/
PDIM = 2

default: check-opt main2d

main2d: $(IBAMR_LIB_2D) $(IBTK_LIB_2D) $(OBJS) main.o
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(OBJS)  main.o \
	$(IBAMR_LIB_2D) $(IBTK_LIB_2D) $(LDFLAGS) $(LIBS) -DNDIM=$(PDIM) -o main2d

main3d: $(IBAMR_LIB_3D) $(IBTK_LIB_3D) $(OBJS) main.o
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(OBJS) main.o \
	$(IBAMR_LIB_3D) $(IBTK_LIB_3D) $(LDFLAGS) $(LIBS) -DNDIM=$(PDIM) -o main3d

check-opt:
	if test "$(OPT)" == "1" ; then				\
	  if test -f stamp-debug ; then $(MAKE) clean ; fi ;	\
	  touch stamp-opt ;					\
	else							\
	  if test -f stamp-opt ; then $(MAKE) clean ; fi ;	\
	  touch stamp-debug ;					\
	fi ;

clean:
	$(RM) main3d
	$(RM) main2d
	$(RM) stamp-{opt,debug}
	$(RM) *.o *.lo *.objs *.ii *.int.c fortran/*.o
	$(RM) src/*.o
	$(RM) -r .libs
