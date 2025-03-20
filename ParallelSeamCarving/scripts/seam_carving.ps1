$width = $args[0]
$source = "./test_images/720x480.png"
$target = "./output_images/720x480_sequential.png"

# Compile and run
gcc -O2 -lm --openmp seam_carving.c -o seam_carving.exe
./seam_carving.exe $source $target $width