import sys
import scripts.example1_random_effects as example1
import scripts.example2a_higgs as example2a
import scripts.example2b_higgs as example2b
import scripts.example3_earthquake as example3

if len(sys.argv) > 1:
    if int(sys.argv[1]) == 1:
        for i in range(5):
            example1.main(seed_offset=i)
    elif int(sys.argv[1]) == 2:
        data = example2a.load_data(file_path='/home/jhd956/archive/data/higgs', subset=110000)
        for i in range(10):
            example2a.main(data, use_all_data=False, seed_offset=i)
        #example2b.main(data, use_all_data=False, seed_offset=0)
    elif int(sys.argv[1]) == 3:
        for i in range(25):
            example3.main(seed_offset=i)
    else:
        raise NameError("Unknown example to run...")
else:
    raise NameError("Need to supply an argument to function call (1, 2, 3) corresponding to the numerical illustration to run.")
