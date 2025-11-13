c234567
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c     Perform a fast sweeping method to convert the level set stored    c
c     in u to a signed distance function. Fixed values are stored in    c
c     the integer valued v:                                             c
c        0 corresponds to values that should be changed                 c
c        1 corresponds to values that are fixed                         c
c          but can be used for derivatives                              c
c        2 corresponds to values that are fixed                         c
c          and should not be used for derivatives.                      c
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine fast_sweep_2d(u, u_gcw, ilow0, iup0, ilow1, iup1, dx,
     &                         v, v_gcw)
        implicit none

        integer ilow0, ilow1
        integer iup0, iup1

        integer u_gcw
        double precision u((ilow0-u_gcw):(iup0+u_gcw+1),
     &                     (ilow1-u_gcw):(iup1+u_gcw+1))

        integer v_gcw
        integer v((ilow0-v_gcw):(iup0+v_gcw+1),
     &            (ilow1-v_gcw):(iup1+v_gcw+1))

        double precision dx(0:1)

c       LOCAL VALUES
        integer i0, i1

        do i1 = ilow1,iup1+1
          do i0 = ilow0,iup0+1
            call do_fast_sweep(u, u_gcw, ilow0, iup0, ilow1, iup1, dx,
     &                         i0, i1, v, v_gcw)
          enddo
        enddo

        do i1 = ilow1,iup1+1
          do i0 = iup0+1,ilow0,-1
            call do_fast_sweep(u, u_gcw, ilow0, iup0, ilow1, iup1, dx,
     &                         i0, i1, v, v_gcw)
          enddo
        enddo

        do i1 = iup1+1,ilow1,-1
          do i0 = iup0+1,ilow0,-1
            call do_fast_sweep(u, u_gcw, ilow0, iup0, ilow1, iup1, dx,
     &                         i0, i1, v, v_gcw)
          enddo
        enddo

        do i1 = iup1+1,ilow1,-1
          do i0 = ilow0,iup0+1
            call do_fast_sweep(u, u_gcw, ilow0, iup0, ilow1, iup1, dx,
     &                         i0, i1, v, v_gcw)
          enddo
        enddo
      end

ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c     Perform the sweep on given indices                                c
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine do_fast_sweep(u, u_gcw, ilow0, iup0, ilow1, iup1, dx,
     &                          i0, i1, v, v_gcw)
        implicit none

        integer ilow0, ilow1
        integer iup0, iup1

        integer u_gcw
        double precision u((ilow0-u_gcw):(iup0+u_gcw+1),
     &                     (ilow1-u_gcw):(iup1+u_gcw+1))

        integer v_gcw
        integer v((ilow0-v_gcw):(iup0+v_gcw+1),
     &            (ilow1-v_gcw):(iup1+v_gcw+1))

        double precision dx(0:1)

        integer i0, i1

c       If this value is fixed, return without modifying
        if (v(i0,i1) .eq. 1 .or. v(i0,i1) .eq. 2) then
           return
        else if (u(i0,i1) .gt. 0) then
           call fs_pos(u, u_gcw, ilow0, iup0, ilow1, iup1,
     &                 v, v_gcw, dx, i0, i1)
        else
           call fs_neg(u, u_gcw, ilow0, iup0, ilow1, iup1,
     &                 v, v_gcw, dx, i0, i1)
        endif
      end

ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c     Perform the sweep on given indices for positive level set values  c
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine fs_pos(u, u_gcw, ilow0, iup0, ilow1, iup1,
     &                  v, v_gcw, dx, i0, i1)
        implicit none

        integer ilow0, ilow1
        integer iup0, iup1

        integer u_gcw
        double precision u((ilow0-u_gcw):(iup0+u_gcw+1),
     &                     (ilow1-u_gcw):(iup1+u_gcw+1))

        integer v_gcw
        integer v((ilow0-v_gcw):(iup0+v_gcw+1),
     &            (ilow1-v_gcw):(iup1+v_gcw+1))

        double precision dx(0:1)

        integer i0, i1

        double precision a, b, val, h

        if (v(i0-1,i1) .eq. 2) then
          a = u(i0+1,i1)
        else if (v(i0+1,i1) .eq. 2) then
          a = u(i0-1,i1)
        else
          a = MIN(u(i0-1,i1),u(i0+1,i1))
        endif
        if (v(i0,i1-1) .eq. 2) then
          b = u(i0,i1+1)
        else if (v(i0,i1+1) .eq. 2) then
          b = u(i0,i1-1)
        else
          b = MIN(u(i0,i1-1),u(i0,i1+1))
        endif

        h = MIN(dx(0),dx(1))

        if (DABS(a - b) .gt. h) then
          val = MIN(a,b) + h
        else
          val = (a + b + SQRT(2*h*h - (a - b)**2.d0)) * 0.5d0
        endif

          u(i0,i1) = MIN(u(i0,i1),val)
      end


ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c     Perform the sweep on given indices for negative level set values  c
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      subroutine fs_neg(u, u_gcw, ilow0, iup0, ilow1, iup1,
     &                  v, v_gcw, dx, i0, i1)
        implicit none

        integer ilow0, ilow1
        integer iup0, iup1

        integer u_gcw
        double precision u((ilow0-u_gcw):(iup0+u_gcw+1),
     &                     (ilow1-u_gcw):(iup1+u_gcw+1))

        integer v_gcw
        integer v((ilow0-v_gcw):(iup0+v_gcw+1),
     &            (ilow1-v_gcw):(iup1+v_gcw+1))

        double precision dx(0:1)

        integer i0, i1

        double precision a, b, val, h

        if (v(i0-1,i1) .eq. 2) then
          a = -u(i0+1,i1)
        else if (v(i0+1,i1) .eq. 2) then
          a = -u(i0-1,i1)
        else
          a = -MAX(u(i0-1,i1),u(i0+1,i1))
        endif
        if (v(i0,i1-1) .eq. 2) then
          b = -u(i0,i1+1)
        else if (v(i0,i1+1) .eq. 2) then
          b = -u(i0,i1-1)
        else
          b = -MAX(u(i0,i1-1),u(i0,i1+1))
        endif

        h = MIN(dx(0),dx(1))

        if (DABS(a - b) .gt. h) then
          val = MIN(a,b) + h
        else
          val = (a + b + SQRT(2*h*h - (a - b)**2.d0)) * 0.5d0
        endif

        u(i0,i1) = MAX(u(i0,i1),-val)
      end

ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c     Find a cell centroid given level set values on nodes.             c
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
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

          if (DABS(sgn_area) .lt. 1.0d-8) then
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
