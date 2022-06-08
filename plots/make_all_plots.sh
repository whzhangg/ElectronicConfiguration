for folder in $(ls -d figure*)
do
    cd $folder
    python figure*.py
    cd -
done