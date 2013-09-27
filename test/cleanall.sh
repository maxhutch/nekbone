for example in `ls -d -- */`; do 
  cp makenek $example
  cd $example
  ./makenek clean
  rm makenek
  cd ..
done

