$width = $args[0]
$source = "./test_images/1024x768.png"
$target = "./output_images/1024x768_parallel.png"

# Compile and run
gcc -O2 -lm --openmp parallel_seam_carving.c -o parallel_seam_carving.exe
./parallel_seam_carving.exe $source $target $width