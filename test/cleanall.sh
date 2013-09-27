for example in `ls -d -- */`; do 
  cp makenek $example
  cd $example
  ./makenek clean
  rm -f makenek compiler.out makefile
  cd ..
done

