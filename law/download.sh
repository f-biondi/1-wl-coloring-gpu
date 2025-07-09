mkdir $1
cd $1
for ext in .properties .graph .md5sums; do
    wget -c http://data.law.di.unimi.it/webdata/$1/$1$ext
done
