for example in `ls -d -- */`; do 
  cp makenek $example
  cd $example
  ./makenek clean
  ./makenek 
  time ./nekbone
  cd ..
done

