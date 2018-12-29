Arnoldi_LAPACKs:
	2>result_make.txt
	g++ -m64 -g Source/LAPACK_routines.cpp -o Obj/Arnoldi_LAPACK_routines.o -c 2>>result_make.txt

Arnoldi_Products:
	2>result_make.txt
	g++ -m64 -g Source/Products.cpp -o Obj/Arnoldi_Products.o -c 2>>result_make.txt


Arnoldi_Shifts:
	2>result_make.txt
	g++ -m64 -g Source/Select_Shifts.cpp -o Obj/Arnoldi_Select_Shifts.o -c 2>>result_make.txt


Arnoldi_QRshifts:
	2>result_make.txt
	g++ -m64  -g Source/QR_Shifts.cpp -o Obj/Arnoldi_QR_Shifts.o -c 2>>result_make.txt


Arnoldi_timer:
	2>result_make.txt
	g++ -m64 -g Source/timer.cpp -o Obj/Arnoldi_timer.o -c 2>>result_make.txt

Arnoldi_Matrix_Vector_emulator:
	2>result_make.txt
	nvcc -m64 -g Source/Matrix_Vector_emulator.cu -o Obj/Arnoldi_Matrix_Vector_emulator.o -c 2>>result_make.txt

Arnoldi_Arnoldi_Driver:
	2>result_make.txt
	nvcc -m64 -g Source/Arnoldi_Driver.cu -o Obj/Arnoldi_Arnoldi_Driver.o -c 2>>result_make.txt

Arnoldi_file_operations:
	2>result_make.txt
	g++ -m64 -g Source/file_operations.cpp -o Obj/Arnoldi_file_operations.o -c 2>>result_make.txt


Arnoldi_memory_operations:
	2>result_make.txt
	nvcc -m64 -g Source/memory_operations.cu -o Obj/Arnoldi_memory_operations.o -c 2>>result_make.txt

Arnoldi_cuda_supp:
	2>result_make.txt
	nvcc -m64 -g Source/cuda_supp.cu -o Obj/Arnoldi_cuda_supp.o -c 2>>result_make.txt

Arnoldi_Implicit_restart_Arnoldi:
	2>result_make.txt
	nvcc  -m64 -g Source/Implicit_restart_Arnoldi.cu -o Obj/Arnoldi_Implicit_restart_Arnoldi.o  -c 2>>result_make.txt

BiCGstabL:
	2>result_make.txt
	nvcc -m64 -g Source/BiCGStabL.cu -o Obj/BiCGStabL.o -c 2>>result_make.txt

GMRES:
	2>result_make.txt
	nvcc -m64 -g Source/GMRES.cu -o Obj/GMRES.o -c 2>>result_make.txt

Arnoldi_deb:
	>result_make.txt
	nvcc -m64 -g Source/cuda_test.cu -o Arnoldi_test Obj/Arnoldi_LAPACK_routines.o Obj/Arnoldi_Products.o Obj/Arnoldi_Select_Shifts.o Obj/Arnoldi_QR_Shifts.o Obj/Arnoldi_timer.o Obj/Arnoldi_Matrix_Vector_emulator.o Obj/Arnoldi_Arnoldi_Driver.o Obj/Arnoldi_file_operations.o Obj/Arnoldi_memory_operations.o Obj/Arnoldi_cuda_supp.o Obj/Arnoldi_Implicit_restart_Arnoldi.o Obj/BiCGStabL.o -L/opt/OpenBLAS/lib -llapack -lblas -lopenblas -lgfortran -lm -lcublas  2>>result_make.txt


Arnoldi_all:
	make BiCGstabL Arnoldi_LAPACKs Arnoldi_Products Arnoldi_Shifts Arnoldi_QRshifts Arnoldi_timer Arnoldi_Matrix_Vector_emulator Arnoldi_Arnoldi_Driver Arnoldi_file_operations Arnoldi_memory_operations Arnoldi_cuda_supp Arnoldi_Implicit_restart_Arnoldi Arnoldi_deb



Arnoldi_linal_test_deb:
	>result_make.txt
	nvcc -m64 -g Source/linal_test.cpp -o linal_test Obj/Arnoldi_LAPACK_routines.o Obj/Arnoldi_Products.o Obj/Arnoldi_Select_Shifts.o Obj/Arnoldi_QR_Shifts.o Obj/Arnoldi_timer.o Obj/Arnoldi_Matrix_Vector_emulator.o Obj/Arnoldi_Arnoldi_Driver.o Obj/Arnoldi_file_operations.o Obj/Arnoldi_memory_operations.o Obj/Arnoldi_cuda_supp.o Obj/Arnoldi_Implicit_restart_Arnoldi.o Obj/BiCGStabL.o Obj/GMRES.o -L/opt/OpenBLAS/lib -llapack -lblas -lopenblas -lgfortran -lm -lcublas  2>>result_make.txt

