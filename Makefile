######################################################################
## Here specify the location of the IBAMR source and the location
## where IBAMR has been built.
IBAMR_SRC_DIR   = ${HOME}/sfw/ibamr/IBAMR
IBAMR_BUILD_DIR = ${HOME}/sfw/ibamr/ibamr-objs-opt
export IBAMR_SRC_DIR
export IBAMR_BUILD_DIR
######################################################################
## Include variables specific to the particular IBAMR build.
include $(IBAMR_BUILD_DIR)/config/make.inc

CPPFLAGS+= -ILSLib/include/

PDIM = 2
export PDIM

default: check-opt main2d

all: main2d

LSLib2D.a:
	$(MAKE) -C LSLib/src LSLib2D.a

LSLib3D.a:
	$(MAKE) -C LSLib/src LSLib3D.a

main2d: LSLib2D.a $(IBAMR_LIB_2D) $(IBTK_LIB_2D) main.o
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(OBJS)  main.o \
	LSLib/src/LSLib2D.a $(IBAMR_LIB_2D) $(IBTK_LIB_2D) $(LDFLAGS) $(LIBS) -DNDIM=$(PDIM) -o main2d

main3d: LSLib3D.a $(IBAMR_LIB_3D) $(IBTK_LIB_3D) $(OBJS) main.o
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(OBJS) main.o \
	LSLib/src/LSLib3D.a $(IBAMR_LIB_3D) $(IBTK_LIB_3D) $(LDFLAGS) $(LIBS) -DNDIM=$(PDIM) -o main3d

check-opt:
	if test "$(OPT)" == "1" ; then				\
	  if test -f stamp-debug ; then $(MAKE) clean ; fi ;	\
	  touch stamp-opt ;					\
	else							\
	  if test -f stamp-opt ; then $(MAKE) clean ; fi ;	\
	  touch stamp-debug ;					\
	fi ;

clean:
	$(MAKE) -C LSLib/src clean
	$(RM) main3d
	$(RM) main2d
	$(RM) stamp-{opt,debug}
	$(RM) *.o *.lo *.objs *.ii *.int.c fortran/*.o
	$(RM) src/*.o
	$(RM) -r .libs
