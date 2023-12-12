c234567
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c     Integrate paths that do not involve level set functions using an  c
c     explicit backward in time midpoint method.                        c
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine integrate_paths_midpoint(path, path_gcw, un_0, un_1,
     &         un_gcw, uh_0, uh_1, uh_gcw, dt, dx,
     &         ilower0, ilower1, iupper0, iupper1)
        implicit none

        integer ilower0, ilower1
        integer iupper0, iupper1

        integer path_gcw
        double precision path((ilower0-path_gcw):(iupper0+path_gcw),
     &                        (ilower1-path_gcw):(iupper1+path_gcw),
     &                        0:1)

        integer un_gcw
        double precision un_0((ilower0-un_gcw):(iupper0+un_gcw+1),
     &                        (ilower1-un_gcw):(iupper1+un_gcw))
        double precision un_1((ilower0-un_gcw):(iupper0+un_gcw),
     &                        (ilower1-un_gcw):(iupper1+un_gcw+1))

        integer uh_gcw
        double precision uh_0((ilower0-uh_gcw):(iupper0+uh_gcw+1),
     &                        (ilower1-uh_gcw):(iupper1+uh_gcw))
        double precision uh_1((ilower0-uh_gcw):(iupper0+uh_gcw),
     &                        (ilower1-uh_gcw):(iupper1+uh_gcw+1))

        double precision dt, dx(0:1)

        integer i0, i1
        double precision ux, uy
        double precision xcom, ycom
        double precision xcom_o, ycom_o

        do i0 = ilower0,iupper0
          do i1 = ilower1,iupper1
            xcom_o = DBLE(i0) + 0.5d0
            ycom_o = DBLE(i1) + 0.5d0
            call find_velocity(i0, i1, un_0, un_1, un_gcw, ilower0,
     &                ilower1, iupper0, iupper1,
     &                xcom_o, ycom_o, ux, uy)
            xcom = xcom_o - 0.5d0 * dt * ux / dx(0)
            ycom = ycom_o - 0.5d0 * dt * uy / dx(1)

            call find_velocity(i0, i1, uh_0, uh_1, uh_gcw, ilower0,
     &                ilower1, iupper0, iupper1, xcom, ycom, ux, uy)
            path(i0, i1, 0) = xcom_o - dt * ux / dx(0)
            path(i0, i1, 1) = ycom_o - dt * uy / dx(1)
          enddo
        enddo
      end

ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c     Integrate paths that do not involve level set functions using an  c
c     explicit backward in time Euler method.                           c
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine integrate_paths_forward(path, path_gcw, u_0, u_1,
     &   u_gcw, dt, dx, ilower0, ilower1, iupper0, iupper1)
        implicit none

        integer ilower0, ilower1
        integer iupper0, iupper1

        integer path_gcw
        double precision path((ilower0-path_gcw):(iupper0+path_gcw),
     &                        (ilower1-path_gcw):(iupper1+path_gcw),
     &                        0:1)

        integer u_gcw
        double precision u_0((ilower0-u_gcw):(iupper0+u_gcw+1),
     &                       (ilower1-u_gcw):(iupper1+u_gcw))
        double precision u_1((ilower0-u_gcw):(iupper0+u_gcw),
     &                       (ilower1-u_gcw):(iupper1+u_gcw+1))

        double precision dt, dx(0:1)
        integer i0, i1
        double precision ux, uy
        double precision xcom, ycom

        do i0 = ilower0,iupper0
          do i1 = ilower1,iupper1
            xcom = DBLE(i0) + 0.5d0
            ycom = DBLE(i1) + 0.5d0
            call find_velocity(i0, i1, u_0, u_1, u_gcw, ilower0,
     &                ilower1, iupper0, iupper1, xcom, ycom, ux, uy)
            path(i0, i1, 0) = xcom - dt * ux / dx(0)
            path(i0, i1, 1) = ycom - dt * uy / dx(1)
          enddo
        enddo
      end

ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c     Integrate paths that do not involve level set functions using an  c
c     explicit backward in time using a midpoint method. Also computes  c
c     departure points at the half time point using a trapezoidal rule. c
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine integrate_paths_midpoint_half(p, p_gcw, ph, ph_gcw,
     &         un_0, un_1, un_gcw, uh_0, uh_1, uh_gcw, dt, dx,
     &         ilower0, ilower1, iupper0, iupper1)
        implicit none

        integer ilower0, ilower1
        integer iupper0, iupper1

        integer p_gcw
        double precision p((ilower0-p_gcw):(iupper0+p_gcw),
     &                     (ilower1-p_gcw):(iupper1+p_gcw),
     &                     0:1)

        integer ph_gcw
        double precision ph((ilower0-ph_gcw):(iupper0+ph_gcw),
     &                      (ilower1-ph_gcw):(iupper1+ph_gcw),
     &                      0:1)

        integer un_gcw
        double precision un_0((ilower0-un_gcw):(iupper0+un_gcw+1),
     &                        (ilower1-un_gcw):(iupper1+un_gcw))
        double precision un_1((ilower0-un_gcw):(iupper0+un_gcw),
     &                        (ilower1-un_gcw):(iupper1+un_gcw+1))

        integer uh_gcw
        double precision uh_0((ilower0-uh_gcw):(iupper0+uh_gcw+1),
     &                        (ilower1-uh_gcw):(iupper1+uh_gcw))
        double precision uh_1((ilower0-uh_gcw):(iupper0+uh_gcw),
     &                        (ilower1-uh_gcw):(iupper1+uh_gcw+1))

        double precision dt, dx(0:1)

        integer i0, i1
        double precision ux, uy
        double precision xcom, ycom
        double precision xcom_o, ycom_o

        do i0 = ilower0,iupper0
          do i1 = ilower1,iupper1
            xcom_o = DBLE(i0) + 0.5d0
            ycom_o = DBLE(i1) + 0.5d0
            call find_velocity(i0, i1, un_0, un_1, un_gcw, ilower0,
     &                ilower1, iupper0, iupper1,
     &                xcom_o, ycom_o, ux, uy)
            xcom = xcom_o - 0.5d0 * dt * ux / dx(0)
            ycom = ycom_o - 0.5d0 * dt * uy / dx(1)

            ph(i0, i1, 0) = xcom_o - 0.25 * dt * ux / dx(0)
            ph(i0, i1, 1) = ycom_o - 0.25 * dt * uy / dx(1)

            call find_velocity(i0, i1, uh_0, uh_1, uh_gcw, ilower0,
     &                ilower1, iupper0, iupper1, xcom, ycom, ux, uy)
            p(i0, i1, 0) = xcom_o - dt * ux / dx(0)
            p(i0, i1, 1) = ycom_o - dt * uy / dx(1)
            ph(i0, i1, 0) = ph(i0, i1, 0) - 0.25 * dt * ux / dx(0)
            ph(i0, i1, 1) = ph(i0, i1, 1) - 0.25 * dt * uy / dx(1)
          enddo
        enddo
      end

ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c     Integrate paths that utilize a level set functions using an       c
c     explicit backward in time midpoint method.                        c
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine integrate_paths_ls_midpoint(path, path_gcw, un_0, un_1,
     &         un_gcw, uh_0, uh_1, uh_gcw, ls, ls_gcw, dt, dx,
     &         ilower0, ilower1, iupper0, iupper1)
        implicit none

        integer ilower0, ilower1
        integer iupper0, iupper1

        integer path_gcw
        double precision path((ilower0-path_gcw):(iupper0+path_gcw),
     &                        (ilower1-path_gcw):(iupper1+path_gcw),
     &                        0:1)

        integer un_gcw
        double precision un_0((ilower0-un_gcw):(iupper0+un_gcw+1),
     &                        (ilower1-un_gcw):(iupper1+un_gcw))
        double precision un_1((ilower0-un_gcw):(iupper0+un_gcw),
     &                        (ilower1-un_gcw):(iupper1+un_gcw+1))

        integer uh_gcw
        double precision uh_0((ilower0-uh_gcw):(iupper0+uh_gcw+1),
     &                        (ilower1-uh_gcw):(iupper1+uh_gcw))
        double precision uh_1((ilower0-uh_gcw):(iupper0+uh_gcw),
     &                        (ilower1-uh_gcw):(iupper1+uh_gcw+1))

        integer ls_gcw
        double precision ls((ilower0-ls_gcw):(iupper0+ls_gcw+1),
     &                      (ilower1-ls_gcw):(iupper1+ls_gcw+1))

        double precision dt, dx(0:1)
        integer i0, i1
        double precision ux, uy
        double precision xcom, ycom
        double precision xcom_o, ycom_o

        do i0 = ilower0,iupper0
          do i1 = ilower1,iupper1
            call find_cell_centroid(xcom_o, ycom_o, i0, i1,
     &               ls(i0,i1), ls(i0,i1+1), ls(i0+1,i1+1), ls(i0+1,i1))
            call find_velocity(i0, i1, un_0, un_1, un_gcw, ilower0,
     &                ilower1, iupper0, iupper1,
     &                xcom_o, ycom_o, ux, uy)
            xcom = xcom_o - 0.5d0 * dt * ux / dx(0)
            ycom = ycom_o - 0.5d0 * dt * uy / dx(1)

            call find_velocity(i0, i1, uh_0, uh_1, uh_gcw, ilower0,
     &                ilower1, iupper0, iupper1, xcom, ycom, ux, uy)
            path(i0, i1, 0) = xcom_o - dt * ux / dx(0)
            path(i0, i1, 1) = ycom_o - dt * uy / dx(1)
          enddo
        enddo
      end

ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c     Integrate paths that utilize a level set functions using an       c
c     explicit backward in time Euler method.                           c
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine integrate_paths_ls_forward(path, path_gcw, u_0, u_1,
     &   u_gcw, ls, ls_gcw, dt, dx, ilower0, ilower1, iupper0, iupper1)
        implicit none

        integer ilower0, ilower1
        integer iupper0, iupper1

        integer path_gcw
        double precision path((ilower0-path_gcw):(iupper0+path_gcw),
     &                        (ilower1-path_gcw):(iupper1+path_gcw),
     &                        0:1)
        integer u_gcw
        double precision u_0((ilower0-u_gcw):(iupper0+u_gcw+1),
     &                       (ilower1-u_gcw):(iupper1+u_gcw))
        double precision u_1((ilower0-u_gcw):(iupper0+u_gcw),
     &                       (ilower1-u_gcw):(iupper1+u_gcw+1))
        integer ls_gcw
        double precision ls((ilower0-ls_gcw):(iupper0+ls_gcw+1),
     &                      (ilower1-ls_gcw):(iupper1+ls_gcw+1))

        double precision dt, dx(0:1)
        integer i0, i1
        double precision ux, uy
        double precision xcom, ycom

        do i0 = ilower0,iupper0
          do i1 = ilower1,iupper1
            call find_cell_centroid(xcom, ycom, i0, i1,
     &               ls(i0,i1), ls(i0,i1+1), ls(i0+1,i1+1), ls(i0+1,i1))
            call find_velocity(i0, i1, u_0, u_1, u_gcw, ilower0,
     &                ilower1, iupper0, iupper1, xcom, ycom, ux, uy)
            path(i0, i1, 0) = xcom - dt * ux / dx(0)
            path(i0, i1, 1) = ycom - dt * uy / dx(1)
          enddo
        enddo
      end

ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c     Interpolate a velocity field to a point (x0, x1) using a          c
c     bilinear interpolant.                                             c
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine find_velocity(i0, i1, u0, u1, u_gcw,
     &      ilower0, ilower1, iupper0, iupper1,
     &      x0, x1, u0_ret, u1_ret)
        implicit none
        integer i0, i1
        integer ilower0, ilower1
        integer iupper0, iupper1

        integer u_gcw
        double precision u0((ilower0-u_gcw):(iupper0+u_gcw+1),
     &                      (ilower1-u_gcw):(iupper1+u_gcw))
        double precision u1((ilower0-u_gcw):(iupper0+u_gcw),
     &                      (ilower1-u_gcw):(iupper1+u_gcw+1))

        double precision x0, x1

        double precision u0_ret, u1_ret

        double precision xlow, ylow
        integer i_ll, i_ul, i_lu, i_uu
        integer j_ll, j_ul, j_lu, j_uu

        if(x1 .gt. (DBLE(i1) + 0.5d0)) then
          i_ll = i0
          j_ll = i1
          i_ul = i0+1
          j_ul = i1
          i_lu = i0
          j_lu = i1+1
          i_uu = i0+1
          j_uu = i1+1
          xlow = DBLE(i0)
          ylow = DBLE(i1) + 0.5d0
        else
          i_ll = i0
          j_ll = i1-1
          i_ul = i0+1
          j_ul = i1-1
          i_lu = i0
          j_lu = i1
          i_uu = i0+1
          j_uu = i1
          xlow = DBLE(i0)
          ylow = DBLE(i1) - 0.5d0
        endif

        u0_ret = u0(i_ll, j_ll) + (u0(i_ul, j_ul) - u0(i_ll, j_ll))
     &           * (x0 - xlow)
     &         + (u0(i_lu, j_lu) - u0(i_ll, j_ll))
     &           * (x1 - ylow)
     &         + (u0(i_uu, j_uu) - u0(i_lu, j_lu)
     &           - u0(i_ul, j_ul) + u0(i_ll, j_ll))
     &           *(x0 - xlow)*(x1 - ylow)

        if(x0 .gt. (DBLE(i0) + 0.5d0)) then
          i_ll = i0
          j_ll = i1
          i_ul = i0+1
          j_ul = i1
          i_lu = i0
          j_lu = i1+1
          i_uu = i0+1
          j_uu = i1+1
          xlow = DBLE(i0) + 0.5d0
          ylow = DBLE(i1)
        else
          i_ll = i0-1
          j_ll = i1
          i_ul = i0
          j_ul = i1
          i_lu = i0-1
          j_lu = i1+1
          i_uu = i0
          j_uu = i1+1
          xlow = DBLE(i0) - 0.5d0
          ylow = DBLE(i1)
        endif

        u1_ret = u1(i_ll, j_ll) + (u1(i_ul, j_ul) - u1(i_ll, j_ll))
     &            * (x0 - xlow)
     &         + (u1(i_lu, j_lu) - u1(i_ll, j_ll))
     &           * (x1 - ylow)
     &         + (u1(i_uu, j_uu) - u1(i_lu, j_lu)
     &           - u1(i_ul, j_ul) + u1(i_ll, j_ll))
     &           *(x0 - xlow)*(x1 - ylow)
      end
