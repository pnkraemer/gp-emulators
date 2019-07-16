cd $2


make -j 

cd ../gp-emulators/lshape/

echo ---------------------------------------------------
echo                     129
echo ---------------------------------------------------

time python3 makeprecon_tps2d.py $1 129
time ../../H2Lib/lshape/lshape_gmres $1 129 92 9


echo ---------------------------------------------------
echo                     1025
echo ---------------------------------------------------

time python3 makeprecon_tps2d.py $1 1025
time ../../H2Lib/lshape/lshape_gmres $1 1025 184 18

echo ---------------------------------------------------
echo                     3350
echo ---------------------------------------------------

time python3 makeprecon_tps2d.py $1 3350
time ../../H2Lib/lshape/lshape_gmres $1 3350 251 25

echo ---------------------------------------------------
echo                     10565
echo ---------------------------------------------------

time python3 makeprecon_tps2d.py $1 10565
time ../../H2Lib/lshape/lshape_gmres $1 10565 326 32
