######################################################################
## Include variables specific to the particular IBAMR build.
include $(IBAMR_BUILD_DIR)/config/make.inc
SRC = $(wildcard *.cpp)

OBJS = $(patsubst %.cpp,%.o,$(SRC))

CPPFLAGS += -I../include/

LSLib2D.a: $(IBAMR_LIB_2D) $(IBTK_LIB_2D) $(OBJS)
	$(CXX) -c $(CXXFLAGS) $(LDFLAGS) -DNDIM=$(PDIM) $(OBJS)
	$(AR) rcs LSLib2D.a $(OBJS)

LSLib3D.a: $(IBAMR_LIB_3D) $(IBTK_LIB_3D) $(OBJS)
	$(CXX) -c $(CXXFLAGS) $(LDFLAGS) -DNDIM=$(PDIM) $(OBJS)
	$(AR) rcs LSLib3D.a $(OBJS)

clean:
	$(RM) stamp-{opt,debug}
	$(RM) *.o *.lo *.objs *.ii *.int.c fortran/*.o *.a
	$(RM) -r .libs
	$(RM) LSLib2D.a LSLib3D.a
