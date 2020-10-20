c234567
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c     Integrate paths that do not involve level set functions using an c
c     explicit backward in time midpoint method.                       c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine integrate_paths_midpoint(path, path_gcw, un_0, un_1,
     &         un_2, un_gcw, uh_0, uh_1, uh_2, uh_gcw, dt, dx,
     &         ilower0, ilower1, ilower2, iupper0, iupper1, iupper2)
        implicit none

        integer ilower0, ilower1, ilower2
        integer iupper0, iupper1, iupper2

        integer path_gcw
        double precision path((ilower0-path_gcw):(iupper0+path_gcw),
     &                        (ilower1-path_gcw):(iupper1+path_gcw),
     &                        (ilower2-path_gcw):(iupper2+path_gcw),
     &                        0:2)

        integer un_gcw
        double precision un_0((ilower0-un_gcw):(iupper0+un_gcw+1),
     &                        (ilower1-un_gcw):(iupper1+un_gcw),
     &                        (ilower2-un_gcw):(iupper2+un_gcw))
        double precision un_1((ilower0-un_gcw):(iupper0+un_gcw),
     &                        (ilower1-un_gcw):(iupper1+un_gcw+1),
     &                        (ilower2-un_gcw):(iupper2+un_gcw))
        double precision un_2((ilower0-un_gcw):(iupper0+un_gcw),
     &                        (ilower1-un_gcw):(iupper1+un_gcw),
     &                        (ilower2-un_gcw):(iupper2+un_gcw+1))

        integer uh_gcw
        double precision uh_0((ilower0-uh_gcw):(iupper0+uh_gcw+1),
     &                        (ilower1-uh_gcw):(iupper1+uh_gcw),
     &                        (ilower2-uh_gcw):(iupper2+uh_gcw))
        double precision uh_1((ilower0-uh_gcw):(iupper0+uh_gcw),
     &                        (ilower1-uh_gcw):(iupper1+uh_gcw+1),
     &                        (ilower2-uh_gcw):(iupper2+uh_gcw))
        double precision uh_2((ilower0-uh_gcw):(iupper0+uh_gcw),
     &                        (ilower1-uh_gcw):(iupper1+uh_gcw),
     &                        (ilower2-uh_gcw):(iupper2+uh_gcw+1))

        double precision dt, dx(0:2)

        integer i0, i1, i2
        double precision ux, uy, uz
        double precision xcom, ycom, zcom
        double precision xcom_o, ycom_o, zcom_o

        do i0 = ilower0,iupper0
          do i1 = ilower1,iupper1
            do i2 = ilower2,iupper2
              xcom_o = DBLE(i0) + 0.5d0
              ycom_o = DBLE(i1) + 0.5d0
              zcom_o = DBLE(i2) + 0.5d0
              call find_velocity(i0, i1, i2, un_0, un_1, un_2, un_gcw,
     &            ilower0, ilower1, ilower2, iupper0, iupper1, iupper2,
     &            xcom_o, ycom_o, zcom_o, ux, uy, uz)
              xcom = xcom_o - 0.5d0 * dt * ux / dx(0)
              ycom = ycom_o - 0.5d0 * dt * uy / dx(1)
              zcom = zcom_o - 0.5d0 * dt * uz / dx(2)

              call find_velocity(i0, i1, i2, uh_0, uh_1, uh_2, uh_gcw,
     &            ilower0, ilower1, ilower2, iupper0, iupper1, iupper2,
     &            xcom, ycom, zcom, ux, uy, uz)
              path(i0, i1, i2, 0) = xcom_o - dt * ux / dx(0)
              path(i0, i1, i2, 1) = ycom_o - dt * uy / dx(1)
              path(i0, i1, i2, 2) = zcom_o - dt * uz / dx(2)
            enddo
          enddo
        enddo
      end

cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c     Integrate paths that do not involve level set functions using an c
c     explicit backward in time Euler method.                          c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine integrate_paths_forward(path, path_gcw, u_0, u_1, u_2,
     &   u_gcw, dt, dx, ilower0, ilower1, ilower2,
     &   iupper0, iupper1, iupper2)
        implicit none

        integer ilower0, ilower1, ilower2
        integer iupper0, iupper1, iupper2

        integer path_gcw
        double precision path((ilower0-path_gcw):(iupper0+path_gcw),
     &                        (ilower1-path_gcw):(iupper1+path_gcw),
     &                        (ilower2-path_gcw):(iupper2+path_gcw),
     &                        0:2)

        integer u_gcw
        double precision u_0((ilower0-u_gcw):(iupper0+u_gcw+1),
     &                       (ilower1-u_gcw):(iupper1+u_gcw),
     &                       (ilower2-u_gcw):(iupper2+u_gcw))
        double precision u_1((ilower0-u_gcw):(iupper0+u_gcw),
     &                       (ilower1-u_gcw):(iupper1+u_gcw+1),
     &                       (ilower2-u_gcw):(iupper2+u_gcw))
        double precision u_2((ilower0-u_gcw):(iupper0+u_gcw),
     &                       (ilower1-u_gcw):(iupper1+u_gcw),
     &                       (ilower2-u_gcw):(iupper2+u_gcw+1))

        double precision dt, dx(0:2)
        integer i0, i1, i2
        double precision ux, uy, uz
        double precision xcom, ycom, zcom

        do i0 = ilower0,iupper0
          do i1 = ilower1,iupper1
            do i2 = ilower2,iupper2
              xcom = DBLE(i0) + 0.5d0
              ycom = DBLE(i1) + 0.5d0
              zcom = DBLE(i2) + 0.5d0
              call find_velocity(i0, i1, i2, u_0, u_1, u_2, u_gcw,
     &            ilower0, ilower1, ilower2, iupper0, iupper1, iupper2,
     &            xcom, ycom, zcom, ux, uy, uz)
              path(i0, i1, i2, 0) = xcom - dt * ux / dx(0)
              path(i0, i1, i2, 1) = ycom - dt * uy / dx(1)
              path(i0, i1, i2, 2) = zcom - dt * uz / dx(2)
            enddo
          enddo
        enddo
      end

cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c     Integrate paths that utilize a level set functions using an      c
c     explicit backward in time midpoint method.                       c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine integrate_paths_ls_midpoint(path, path_gcw, un_0, un_1,
     &         un_2, un_gcw, uh_0, uh_1, uh_2, uh_gcw, c_data, c_gcw,
     &         dt, dx, ilower0, ilower1, ilower2, iupper0, iupper1,
     &         iupper2)
        implicit none
       
        integer ilower0, ilower1, ilower2
        integer iupper0, iupper1, iupper2
         
        integer path_gcw
        double precision path((ilower0-path_gcw):(iupper0+path_gcw),
     &                        (ilower1-path_gcw):(iupper1+path_gcw),
     &                        (ilower2-path_gcw):(iupper2+path_gcw),
     &                        0:2)
     
        integer un_gcw
        double precision un_0((ilower0-un_gcw):(iupper0+un_gcw+1),
     &                        (ilower1-un_gcw):(iupper1+un_gcw),
     &                        (ilower2-un_gcw):(iupper2+un_gcw))
        double precision un_1((ilower0-un_gcw):(iupper0+un_gcw),
     &                        (ilower1-un_gcw):(iupper1+un_gcw+1),
     &                        (ilower2-un_gcw):(iupper2+un_gcw))
        double precision un_2((ilower0-un_gcw):(iupper0+un_gcw),
     &                        (ilower1-un_gcw):(iupper1+un_gcw),
     &                        (ilower2-un_gcw):(iupper2+un_gcw+1))
     
        integer uh_gcw
        double precision uh_0((ilower0-uh_gcw):(iupper0+uh_gcw+1),
     &                        (ilower1-uh_gcw):(iupper1+uh_gcw),
     &                        (ilower2-uh_gcw):(iupper2+uh_gcw))
        double precision uh_1((ilower0-uh_gcw):(iupper0+uh_gcw),
     &                        (ilower1-uh_gcw):(iupper1+uh_gcw+1),
     &                        (ilower2-uh_gcw):(iupper2+uh_gcw))
        double precision uh_2((ilower0-uh_gcw):(iupper0+uh_gcw),
     &                        (ilower1-uh_gcw):(iupper1+uh_gcw),
     &                        (ilower2-uh_gcw):(iupper2+uh_gcw+1))

        integer c_gcw
        double precision c_data((ilower0-c_gcw):(iupper0+c_gcw),
     &                          (ilower1-c_gcw):(iupper1+c_gcw),
     &                          (ilower2-c_gcw):(iupper2+c_gcw),
     &                          0:2)

        double precision dt, dx(0:2)
        integer i0, i1, i2
        double precision ux, uy, uz
        double precision xcom, ycom, zcom
        double precision xcom_o, ycom_o, zcom_o
         
        do i0 = ilower0,iupper0
          do i1 = ilower1,iupper1
            do i2 = ilower2,iupper2
              xcom_o = c_data(i0, i1, i2, 0)
              ycom_o = c_data(i0, i1, i2, 1)
              zcom_o = c_data(i0, i1, i2, 2)
              call find_velocity(i0, i1, i2, un_0, un_1, un_2, un_gcw,
     &            ilower0, ilower1, ilower2, iupper0, iupper1, iupper2,
     &            xcom_o, ycom_o, zcom_o, ux, uy, uz)
              xcom = xcom_o - 0.5d0 * dt * ux / dx(0)
              ycom = ycom_o - 0.5d0 * dt * uy / dx(1)
              zcom = zcom_o - 0.5d0 * dt * uz / dx(2)

              call find_velocity(i0, i1, i2, uh_0, uh_1, uh_2, uh_gcw,
     &            ilower0, ilower1, ilower2, iupper0, iupper1, iupper2,
     &            xcom, ycom, zcom, ux, uy, uz)
              path(i0, i1, i2, 0) = xcom_o - dt * ux / dx(0)
              path(i0, i1, i2, 1) = ycom_o - dt * uy / dx(1)
              path(i0, i1, i2, 2) = zcom_o - dt * uz / dx(2)
            enddo
          enddo
        enddo
      end
       
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c     Integrate paths that utilize a level set functions using an      c
c     explicit backward in time Euler method.                          c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine integrate_paths_ls_forward(path, path_gcw, u_0, u_1,
     &   u_2, u_gcw, c_data, c_gcw, dt, dx,
     &   ilower0, ilower1, ilower2, iupper0, iupper1, iupper2)
        implicit none

        integer ilower0, ilower1, ilower2
        integer iupper0, iupper1, iupper2
         
        integer path_gcw
        double precision path((ilower0-path_gcw):(iupper0+path_gcw),
     &                        (ilower1-path_gcw):(iupper1+path_gcw),
     &                        (ilower2-path_gcw):(iupper2+path_gcw),
     &                        0:2)
        integer u_gcw
        double precision u_0((ilower0-u_gcw):(iupper0+u_gcw+1),
     &                       (ilower1-u_gcw):(iupper1+u_gcw),
     &                       (ilower2-u_gcw):(iupper2+u_gcw))
        double precision u_1((ilower0-u_gcw):(iupper0+u_gcw),
     &                       (ilower1-u_gcw):(iupper1+u_gcw+1),
     &                       (ilower2-u_gcw):(iupper2+u_gcw))
        double precision u_2((ilower0-u_gcw):(iupper0+u_gcw),
     &                       (ilower1-u_gcw):(iupper1+u_gcw),
     &                       (ilower2-u_gcw):(iupper2+u_gcw+1))

        integer c_gcw
        double precision c_data((ilower0-c_gcw):(iupper0+c_gcw),
     &                          (ilower1-c_gcw):(iupper1+c_gcw),
     &                          (ilower2-c_gcw):(iupper2+c_gcw),
     &                          0:2)

        double precision dt, dx(0:2)
        integer i0, i1, i2
        double precision ux, uy, uz
        double precision xcom(0:2)

        do i0 = ilower0,iupper0
          do i1 = ilower1,iupper1
            do i2 = ilower2,iupper2
              xcom(0) = c_data(i0, i1, i2, 0)
              xcom(1) = c_data(i0, i1, i2, 1)
              xcom(2) = c_data(i0, i1, i2, 2)
              call find_velocity(i0, i1, i2, u_0, u_1, u_2, u_gcw,
     &            ilower0, ilower1, ilower2, iupper0, iupper1, iupper2,
     &            xcom(0), xcom(1), xcom(2), ux, uy, uz)
              path(i0, i1, i2, 0) = xcom(0) - dt * ux / dx(0)
              path(i0, i1, i2, 1) = xcom(1) - dt * uy / dx(1)
              path(i0, i1, i2, 2) = xcom(2) - dt * uz / dx(2)
            enddo
          enddo
        enddo
      end
       
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c     Interpolate a velocity field to a point (x0, x1) using a         c
c     bilinear interpolant.                                            c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine find_velocity(i0, i1, i2, u0, u1, u2, u_gcw,
     &      ilower0, ilower1, ilower2, iupper0, iupper1, iupper2,
     &      x0, x1, x2, u0_ret, u1_ret, u2_ret)
        implicit none
        integer i0, i1, i2
        integer ilower0, ilower1, ilower2
        integer iupper0, iupper1, iupper2

        integer u_gcw
        double precision u0((ilower0-u_gcw):(iupper0+u_gcw+1),
     &                      (ilower1-u_gcw):(iupper1+u_gcw),
     &                      (ilower2-u_gcw):(iupper2+u_gcw))
        double precision u1((ilower0-u_gcw):(iupper0+u_gcw),
     &                      (ilower1-u_gcw):(iupper1+u_gcw+1),
     &                      (ilower2-u_gcw):(iupper2+u_gcw))
        double precision u2((ilower0-u_gcw):(iupper0+u_gcw),
     &                      (ilower1-u_gcw):(iupper1+u_gcw),
     &                      (ilower2-u_gcw):(iupper2+u_gcw+1))

        double precision x0, x1, x2

        double precision u0_ret, u1_ret, u2_ret

        double precision xlow, ylow, zlow
        double precision coefs(1:8)
        integer i
        integer j
        integer k
     
c       X coordinate
        if(x1 .gt. (DBLE(i1) + 0.5d0)) then
          if (x2 .gt.(DBLE(i2) + 0.5d0)) then
            i = i0
            j = i1
            k = i2
            xlow = DBLE(i0)
            ylow = DBLE(i1) + 0.5d0
            zlow = DBLE(i2) + 0.5d0
          else
            i = i0
            j = i1
            k = i2-1
            xlow = DBLE(i0)
            ylow = DBLE(i1) + 0.5d0
            zlow = DBLE(i2) - 0.5d0
          endif
        else
          if (x2 .gt. (DBLE(i2) + 0.5d0)) then
            i = i0
            j = i1 - 1
            k = i2
            xlow = DBLE(i0)
            ylow = DBLE(i1) - 0.5d0
            zlow = DBLE(i2) + 0.5d0
          else
            i = i0
            j = i1 - 1
            k = i2 - 1
            xlow = DBLE(i0)
            ylow = DBLE(i1) - 0.5d0
            zlow = DBLE(i2) - 0.5d0
          endif
        endif
        coefs(1) = u0(i,j,k)
        coefs(2) = u0(i+1,j,k) - coefs(1)
        coefs(3) = u0(i,j+1,k) - coefs(1)
        coefs(4) = u0(i,j,k+1) - coefs(1)
        coefs(5) = u0(i+1,j+1,k) - coefs(1) - coefs(2)
        coefs(6) = u0(i,j+1,k+1) - coefs(1) - coefs(3)
        coefs(7) = u0(i+1,j,k+1) - coefs(1) - coefs(4)
        coefs(8) = u0(i+1,j+1,k+1) - SUM(coefs(1:7))
        u0_ret = coefs(1) + coefs(2) * (x0 - xlow)
     &       + coefs(3) * (x1 - ylow) + coefs(4) * (x2 - zlow)
     &       + coefs(5) * (x0 - xlow) * (x1 - ylow)
     &       + coefs(6) * (x1 - ylow) * (x2 - zlow)
     &       + coefs(7) * (x0 - xlow) * (x2 - zlow)
     &       + coefs(8) * (x0 - xlow) * (x1 - ylow) * (x2 - zlow)
c       Y coordinate
        if(x0 .gt. (DBLE(i1) + 0.5d0)) then
          if (x2 .gt.(DBLE(i2) + 0.5d0)) then
            i = i0
            j = i1
            k = i2
            xlow = DBLE(i0) + 0.5d0
            ylow = DBLE(i1)
            zlow = DBLE(i2) + 0.5d0
          else
            i = i0
            j = i1
            k = i2-1
            xlow = DBLE(i0) + 0.5d0
            ylow = DBLE(i1)
            zlow = DBLE(i2) - 0.5d0
          endif
        else
          if (x2 .gt. (DBLE(i2) + 0.5d0)) then
            i = i0 - 1
            j = i1
            k = i2
            xlow = DBLE(i0) - 0.5d0
            ylow = DBLE(i1)
            zlow = DBLE(i2) + 0.5d0
          else
            i = i0 - 1
            j = i1
            k = i2 - 1
            xlow = DBLE(i0) - 0.5d0
            ylow = DBLE(i1)
            zlow = DBLE(i2) - 0.5d0
          endif
        endif
        coefs(1) = u1(i,j,k)
        coefs(2) = u1(i+1,j,k) - coefs(1)
        coefs(3) = u1(i,j+1,k) - coefs(1)
        coefs(4) = u1(i,j,k+1) - coefs(1)
        coefs(5) = u1(i+1,j+1,k) - coefs(1) - coefs(2)
        coefs(6) = u1(i,j+1,k+1) - coefs(1) - coefs(3)
        coefs(7) = u1(i+1,j,k+1) - coefs(1) - coefs(4)
        coefs(8) = u1(i+1,j+1,k+1) - SUM(coefs(1:7))
        u1_ret = coefs(1) + coefs(2) * (x0 - xlow)
     &       + coefs(3) * (x1 - ylow) + coefs(4) * (x2 - zlow)
     &       + coefs(5) * (x0 - xlow) * (x1 - ylow)
     &       + coefs(6) * (x1 - ylow) * (x2 - zlow)
     &       + coefs(7) * (x0 - xlow) * (x2 - zlow)
     &       + coefs(8) * (x0 - xlow) * (x1 - ylow) * (x2 - zlow)
c       Z coordinate
        if(x0 .gt. (DBLE(i1) + 0.5d0)) then
          if (x1 .gt.(DBLE(i2) + 0.5d0)) then
            i = i0
            j = i1
            k = i2
            xlow = DBLE(i0) + 0.5d0
            ylow = DBLE(i1) + 0.5d0
            zlow = DBLE(i2)
          else
            i = i0
            j = i1-1
            k = i2
            xlow = DBLE(i0) + 0.5d0
            ylow = DBLE(i1) - 0.5d0
            zlow = DBLE(i2)
          endif
        else
          if (x1 .gt. (DBLE(i2) + 0.5d0)) then
            i = i0 - 1
            j = i1
            k = i2
            xlow = DBLE(i0) - 0.5d0
            ylow = DBLE(i1) + 0.5d0
            zlow = DBLE(i2)
          else
            i = i0 - 1
            j = i1 - 1
            k = i2
            xlow = DBLE(i0) - 0.5d0
            ylow = DBLE(i1) - 0.5d0
            zlow = DBLE(i2)
          endif
        endif
        coefs(1) = u2(i,j,k)
        coefs(2) = u2(i+1,j,k) - coefs(1)
        coefs(3) = u2(i,j+1,k) - coefs(1)
        coefs(4) = u2(i,j,k+1) - coefs(1)
        coefs(5) = u2(i+1,j+1,k) - coefs(1) - coefs(2)
        coefs(6) = u2(i,j+1,k+1) - coefs(1) - coefs(3)
        coefs(7) = u2(i+1,j,k+1) - coefs(1) - coefs(4)
        coefs(8) = u2(i+1,j+1,k+1) - SUM(coefs(1:7))
        u2_ret = coefs(1) + coefs(2) * (x0 - xlow)
     &       + coefs(3) * (x1 - ylow) + coefs(4) * (x2 - zlow)
     &       + coefs(5) * (x0 - xlow) * (x1 - ylow)
     &       + coefs(6) * (x1 - ylow) * (x2 - zlow)
     &       + coefs(7) * (x0 - xlow) * (x2 - zlow)
     &       + coefs(8) * (x0 - xlow) * (x1 - ylow) * (x2 - zlow)
      end
