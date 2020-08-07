c234567
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c     Integrate paths that do not involve level set functions using an c
c     explicit backward in time midpoint method.                       c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
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

cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c     Integrate paths that do not involve level set functions using an c
c     explicit backward in time Euler method.                          c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
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

cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c     Integrate paths that utilize a level set functions using an      c
c     explicit backward in time midpoint method.                       c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
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
       
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c     Integrate paths that utilize a level set functions using an      c
c     explicit backward in time Euler method.                          c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
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
       
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c     Interpolate a velocity field to a point (x0, x1) using a         c
c     bilinear interpolant.                                            c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
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
       
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c     Find a cell centroid given level set values on nodes.           c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine find_cell_centroid(xcom, ycom, i0, i1,
     &        ls_ll, ls_lu, ls_uu, ls_ul)
        implicit none
        double precision xcom, ycom
        double precision ls_ll, ls_lu, ls_ul, ls_uu
        integer i0, i1
        double precision x_bounds(0:7), y_bounds(0:7)
        integer num, i, i2
        double precision sgn_area, fac

        if (DABS(ls_ll) .lt. 1.0d-12) then
          ls_ll = DSIGN(1.0d-12, ls_ll)
        endif
        if (DABS(ls_lu) .lt. 1.0d-12) then
          ls_lu = DSIGN(1.0d-12, ls_lu)
        endif
        if (DABS(ls_ul) .lt. 1.0d-12) then
          ls_ul = DSIGN(1.0d-12, ls_ul)
        endif
        if (DABS(ls_uu) .lt. 1.0d-12) then
          ls_uu = DSIGN(1.0d-12, ls_uu)
        endif

        if ((ls_ll .lt. 0.d0 .and. ls_lu .lt. 0.d0
     &       .and. ls_ul .lt. 0.d0 .and. ls_uu .lt. 0.d0)
     &   .or. (ls_ll .gt. 0.d0 .and. ls_lu .gt. 0.d0
     &       .and. ls_ul .gt. 0.d0 .and. ls_uu .gt. 0.d0)) then
          xcom = DBLE(i0) + 0.5d0
          ycom = DBLE(i1) + 0.5d0
        else
          num = 0
          if (ls_ll .lt. 0.d0) then
            x_bounds(num) = DBLE(i0)
            y_bounds(num) = DBLE(i1)
            num = num + 1
          endif
          if (ls_ll * ls_lu .lt. 0.d0) then
            x_bounds(num) = DBLE(i0)
            y_bounds(num) = DBLE(i1) - ls_ll / (ls_lu - ls_ll)
            num = num + 1
          endif
          if (ls_lu .lt. 0.d0) then
            x_bounds(num) = DBLE(i0)
            y_bounds(num) = DBLE(i1) + 1.d0
            num = num + 1
          endif
          if (ls_lu * ls_uu .lt. 0.d0) then
            x_bounds(num) = DBLE(i0) - ls_lu / (ls_uu - ls_lu)
            y_bounds(num) = DBLE(i1) + 1.d0
            num = num+1
          endif
          if (ls_uu .lt. 0.d0) then
            x_bounds(num) = DBLE(i0) + 1.d0
            y_bounds(num) = DBLE(i1) + 1.d0
            num = num + 1
          endif
          if (ls_uu * ls_ul .lt. 0.d0) then
            x_bounds(num) = DBLE(i0) + 1.d0
            y_bounds(num) = DBLE(i1) - ls_ul / (ls_uu - ls_ul)
            num = num + 1
          endif
          if (ls_ul .lt. 0.d0) then
            x_bounds(num) = DBLE(i0) + 1.d0
            y_bounds(num) = DBLE(i1)
            num = num + 1
          endif
          if (ls_ul * ls_ll .lt. 0.d0) then
            x_bounds(num) = DBLE(i0) - ls_ll / (ls_ul - ls_ll)
            y_bounds(num) = DBLE(i1)
            num = num + 1
          endif

          xcom = 0.d0
          ycom = 0.d0
          sgn_area = 0.d0

          do i = 0,(num-1)
            i2 = mod(i + 1, num)
            fac = x_bounds(i)*y_bounds(i2) - x_bounds(i2)*y_bounds(i)
            xcom = xcom + (x_bounds(i) + x_bounds(i2)) * fac
            ycom = ycom + (y_bounds(i) + y_bounds(i2)) * fac
            sgn_area  = sgn_area + 0.5d0 * fac
          enddo
          xcom = xcom / (6.d0 * sgn_area)
          ycom = ycom / (6.d0 * sgn_area)

          if (sgn_area .lt. 1.0d-8) then
            xcom = 0.d0
            ycom = 0.d0
            do i = 0,(num-1)
              xcom = xcom + x_bounds(i)
              ycom = ycom + y_bounds(i)
            enddo
            xcom = xcom / num
            ycom = ycom / num
          endif
        endif
      end
