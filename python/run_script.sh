# Fix for some kind of bug in matplotlib
touch /usr/local/lib/python3.6/site-packages/mpl_toolkits/__init__.py

# Run Python files when the container launches
python setup.py build_ext --inplace
wait

python run_script.py $EXPERIMENT $FULLRUN
wait

