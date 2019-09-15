
# run as:
# bash precon_uniform_lshape <path_to_files> <path_to_h2lib>

cd $2
make -j 

cd ../gp-emulators/lshape/

echo Creating file for output...

rm output_uniform_lshape.txt
touch output_uniform_lshape.txt

echo done.

echo ---------------------------------------------------
echo                     21
echo ---------------------------------------------------

time python3 makeprecon_tps2d.py $1 21 |& tee -a output_uniform_lshape.txt 

time ./../../H2Lib/lshape/lshape_gmres $1 21 24 5 |& tee -a output_uniform_lshape.txt  


echo ---------------------------------------------------
echo                     65
echo ---------------------------------------------------

time python3 makeprecon_tps2d.py $1 65

time ./../../H2Lib/lshape/lshape_gmres $1 65 68 6 >> output_uniform_lshape.txt  

echo ---------------------------------------------------
echo                     225
echo ---------------------------------------------------

time python3 makeprecon_tps2d.py $1 225 |& tee -a output_uniform_lshape.txt

time ./../../H2Lib/lshape/lshape_gmres $1 225 113 11 |& tee -a output_uniform_lshape.txt  

echo ---------------------------------------------------
echo                     833
echo ---------------------------------------------------

time python3 makeprecon_tps2d.py $1 833 |& tee -a output_uniform_lshape.txt

time ./../../H2Lib/lshape/lshape_gmres $1 833 173 17 |& tee -a output_uniform_lshape.txt  

echo ---------------------------------------------------
echo                     3201
echo ---------------------------------------------------

time python3 makeprecon_tps2d.py $1 3201 |& tee -a output_uniform_lshape.txt

time ./../../H2Lib/lshape/lshape_gmres $1 3201 248 25 |& tee -a output_uniform_lshape.txt  


echo ---------------------------------------------------
echo                     12545
echo ---------------------------------------------------

time python3 makeprecon_tps2d.py $1 12545 |& tee -a output_uniform_lshape.txt

time ./../../H2Lib/lshape/lshape_gmres $1 12545 338 34 |& tee -a output_uniform_lshape.txt  


echo ---------------------------------------------------
echo                     49665
echo ---------------------------------------------------

time python3 makeprecon_tps2d.py $1 49665 |& tee -a output_uniform_lshape.txt

time ./../../H2Lib/lshape/lshape_gmres $1 49665 444 44 |& tee -a output_uniform_lshape.txt  


echo ---------------------------------------------------
echo                     197633
echo ---------------------------------------------------

time python3 makeprecon_tps2d.py $1 197633 |& tee -a output_uniform_lshape.txt

time ./../../H2Lib/lshape/lshape_gmres $1 197633 563 56 |& tee -a output_uniform_lshape.txt  


