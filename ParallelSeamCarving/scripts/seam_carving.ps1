gcc -O2 -lm --openmp seam_carving.c -o seam_carving.exe
./seam_carving.exe ./test_images/720x480.png ./output_images/720x480.png 719