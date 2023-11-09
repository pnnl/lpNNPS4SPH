!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!  This script is developed to search neighbors for SPH with all-list algorithm %
!                                                                         %
!  Author: Zirui Mao (Pacific Northwest National Laboratory)              %
!  Date last modified: Sept. 05, 2023                                     %
!                                                                         %
! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!  !!!!!!!!! Read me before using !!!!!!!!!!                              %
!  By using this freeware, you are agree to the following:                %
!  1. you are free to copy and redistribute the material in any format;   %
!  2. you are free to remix, transform, and build upon the material for   %
!     any purpose, even commercially;                                     %
!  3. you must provide the name of the creator and attribution parties,   %
!     a copyright notice, a license notice, a disclaimer notice, and a    % 
!     link to the material [link];                                        %
!  4. users are entirely at their own risk using this freeware.           %
!                                                                         %
!  Before use, please read the License carefully:                         %
!  <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">   %
!  <img alt="Creative Commons License" style="border-width:0"             %
!  src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a><br />    %
!  This work is licensed under a                                          %
!  <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">   %
!  Creative Commons Attribution 4.0 International License</a>.   

program search_neighbors

  implicit none

  integer, parameter :: N = 200000
  real(8) :: x(N), y(N), z(N)
  integer :: neighbors(N, 100), num_neighbors(N)
  real, parameter :: R_CUT = 0.02
  real(8) :: dx, dy, dz, r

  integer :: i, j, count, iter, k

  real(8) t_start,t_end,tt

  iter = 1

  ! Initialize particle positions randomly
  do i = 1, N
     call random_number(x(i))
     call random_number(y(i))
     call random_number(z(i))
  end do

!   open(unit = 10, file = 'coordinates_1000000.dat')
!   do i = 1, N
!     read(10,*) x(i),y(i),z(i)
!   enddo
!   close(10)

  ! Search for neighbors closer than R_CUT
  call cpu_time(t_start)
  do k = 1,iter
   do i = 1, N
      count = 0
      do j = 1, N
         if (i /= j) then
            
            dx = x(i) - x(j)
            dy = y(i) - y(j)
            dz = z(i) - z(j)
            r = sqrt(dx*dx + dy*dy + dz*dz)
            if (r <= R_CUT) then
               count = count + 1
               neighbors(i, count) = j
            end if
         end if
      end do
      num_neighbors(i) = count
   end do
   enddo
   call cpu_time(t_end)
   

  ! Print out neighbors of each particle
  do i = N, N
     write (*, '(A,I6,A)', advance='no') 'Particle ', i, ' neighbors: '
     do j = 1, num_neighbors(i)
        write (*, '(I6)', advance='no') neighbors(i, j)
     end do
  end do

  write(*,*) 'time =', t_end-t_start

end program search_neighbors
