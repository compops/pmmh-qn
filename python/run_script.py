import sys
import scripts_draft1.example1_random_effects as example1
import scripts_draft1.example2_earthquake as example2
import scripts_draft1.example3_higgs as example3

if len(sys.argv) > 1:
    if int(sys.argv[1]) == 1:
        for i in range(5):
            example1.main(seed_offset=i)
    elif int(sys.argv[1]) == 2:
        for i in range(25):
            example2.main(seed_offset=i)
    elif int(sys.argv[1]) == 3:
        data = example3.load_data(file_path='/home/compops/archive/data/higgs', subset=110000)
        example3.main(data, use_all_data=False, seed_offset=0)
    else:
        raise NameError("Unknown example to run...")
else:
    raise NameError("Need to supply an argument to function call (1, 2, 3) corresponding to the numerical illustration to run.")
