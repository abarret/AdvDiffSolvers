c ---------------------------------------------------------------------
c
c Copyright (c) 2017 - 2018 by the IBAMR developers
c All rights reserved.
c
c This file is part of IBAMR.
c
c IBAMR is free software and is distributed under the 3-clause BSD
c license. The full text of the license can be found in the file
c COPYRIGHT at the top level directory of IBAMR.
c
c ---------------------------------------------------------------------

c
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
c     Carry out sign sweeping algorithm
c
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
      subroutine signsweep2dn(
     &     u,u_gcw,
     &     ilower0,iupper0,
     &     ilower1,iupper1,
     &     large_dist,
     &     n_updates)
c
      implicit none
c
c     Input.
c
      INTEGER ilower0,iupper0
      INTEGER ilower1,iupper1
      INTEGER u_gcw
      double precision large_dist
c
c     Input/Output.
c
      double precision u((ilower0-u_gcw):(iupper0+u_gcw+1),
     &                   (ilower1-u_gcw):(iupper1+u_gcw+1))
      INTEGER n_updates

c
c     Local variables.
c
      INTEGER i0,i1
      double precision sgn, sgn_nbr
      double precision one


c     Do the four sweeping directions.

      one = 1.d0
      do i1 = ilower1,iupper1+1
         do i0 = ilower0,iupper0+1
            if (dabs(u(i0,i1)) .ge. large_dist) then
               sgn = sign(one,u(i0,i1))
               sgn_nbr = sign(one,u(i0-1,i1))
               if (sgn .ne. sgn_nbr) then
                  u(i0,i1) = dabs(u(i0,i1))*sgn_nbr
                  n_updates = n_updates + 1
               endif
            endif
         enddo
      enddo

      do i1 = iupper1+1,ilower1,-1
         do i0 = ilower0,iupper0+1
            if (dabs(u(i0,i1)) .ge. large_dist) then
               sgn = sign(one,u(i0,i1))
               sgn_nbr = sign(one,u(i0,i1+1))
               if (sgn .ne. sgn_nbr) then
                  u(i0,i1) = dabs(u(i0,i1))*sgn_nbr
                  n_updates = n_updates + 1
               endif
            endif
         enddo
      enddo

      do i1 = ilower1,iupper1+1
         do i0 = iupper0+1,ilower0,-1
            if (dabs(u(i0,i1)) .ge. large_dist) then
               sgn = sign(one,u(i0,i1))
               sgn_nbr = sign(one,u(i0+1,i1-1))
               if (sgn .ne. sgn_nbr) then
                  u(i0,i1) = dabs(u(i0,i1))*sgn_nbr
                  n_updates = n_updates + 1
               endif
            endif
         enddo
      enddo

      do i1 = iupper1+1,ilower1,-1
         do i0 = iupper0+1,ilower0,-1
            if (dabs(u(i0,i1)) .ge. large_dist) then
               sgn = sign(one,u(i0,i1))
               sgn_nbr = sign(one,u(i0+1,i1+1))
               if (sgn .ne. sgn_nbr) then
                  u(i0,i1) = dabs(u(i0,i1))*sgn_nbr
                  n_updates = n_updates + 1
               endif
            endif
         enddo
      enddo

      return
      end

