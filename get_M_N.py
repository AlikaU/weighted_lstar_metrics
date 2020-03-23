import argparse, math, ast, os
from Learner import learn
from our_grammars import uhl1, uhl2, uhl3
from Helper_Functions import prepare_directory, overwrite_file, clean_val
from LanguageModel import LanguageModel
from time import process_time
from RNNTokenPredictor import RNNTokenPredictor, train_rnn, load_rnn


def make_spice_style_train(lm,n_samples,max_len,filename):
	prepare_directory(filename,includes_filename=True)
	with open(filename,"w") as f:
		print(n_samples,len(lm.internal_alphabet),file=f)
		for _ in range(n_samples):
			s = lm.sample(cutoff=max_len)
			print(len(s),*s,file=f)

def read_spice_style_train_data(filename):
	print("loading from file:",filename)
	if not os.path.exists(filename):
		return None, None
	with open(filename,"r") as f:
		res = f.readlines()
	len_res, alpha_size = tuple(map(int,res[0].split())) 
	alphabet = list(range(alpha_size)) 
	res = res[1:] # first line has metadata, read above ^
	res = list(map(lambda x:x.split()[1:],res)) # first number in each line is just its length 
	res = list(map(lambda x:list(map(int,x)), res)) # input file had strings for characters, return to the numbers
	return res, alphabet

def lapse_str(lapse,digs=2):
	return str(clean_val(lapse,digs))+"s ( "+str(clean_val(lapse/(60*60),2))+" hours)"

def clock_str(clock_start):
	return lapse_str(process_time()-clock_start,0)

def do_lstar(rnn, rnn_folder, args):
	print("~~~running weighted lstar extraction~~~")
	lstar_folder = rnn_folder+"/lstar"
	prepare_directory(lstar_folder)
	lstar_start = process_time()
	lstar_prints_filename = lstar_folder + "/extraction_prints.txt"
	print("progress prints will be in:",lstar_prints_filename)
	with open(lstar_prints_filename,"w") as f:
		lstar_pdfa,table,minimiser = learn(rnn,
			max_states = args.max_states,
			max_P = args.max_P,
			max_S=args.max_S,
			pdfas_path = lstar_folder,
			prints_path = f,
			atol = args.t_tol, 
			interval_width=args.interval_width,
			n_cex_attempts=args.num_cex_attempts,
			max_counterexample_length=args.max_counterexample_length,
			expanding_time_limit=args.lstar_time_limit,\
			s_separating_threshold=args.lstar_s_threshold,\
			interesting_p_transition_threshold=args.lstar_p_threshold,\
			progress_P_print_rate=args.progress_P_print_rate) 
	lstar_pdfa.creation_info = {"extraction time":process_time()-lstar_start,"size":len(lstar_pdfa.transitions)}
	lstar_pdfa.creation_info.update(vars(args)) # get all the extraction hyperparams as well, though this will also catch other hyperparams like the ngrams and stuff..
	overwrite_file(lstar_pdfa,lstar_folder+"/pdfa") # will end up in .gz
	with open(lstar_folder+"/extraction_info.txt","w") as f:
		print(lstar_pdfa.creation_info,file=f)
	lstar_pdfa.draw_nicely(keep=True,filename=lstar_folder+"/pdfa") # will end up in .img
	print("finished lstar extraction, that took:",clock_str(lstar_start))
	return lstar_pdfa

def get_rnn(args):
    uhl = {1:uhl1(),2:uhl2(),3:uhl3()}
    folder = "results"
    prepare_directory(folder)

    if args.spice_example:
        train_filename = "example_spice_data/0.spice.train"
        all_samples, alphabet = read_spice_style_train_data(train_filename)
        informal_name = "spice_0"
        rnn_folder = folder + "/"+ informal_name+"_"+str(process_time())
        prepare_directory(rnn_folder)
    else:
        target = uhl[args.uhl_num]
        lm = LanguageModel(target)
        informal_name = target.informal_name
        rnn_folder = folder + "/"+ informal_name + "_" + str(process_time())
        prepare_directory(rnn_folder)
        print("making samples for",informal_name,end=" ... ")
        train_filename = rnn_folder+"/target_samples.txt"
        make_spice_style_train(lm,args.total_generated_train_samples,args.max_generated_train_sample_len,train_filename)
        print("done")
        target.draw_nicely(keep=True,filename=rnn_folder+"/target_pdfa")

    all_samples, alphabet = read_spice_style_train_data(train_filename)

    train_frac = 0.9
    val_frac = 0.05
    train_stop = int(train_frac*len(all_samples))
    val_stop = train_stop + int(val_frac*len(all_samples))
    train_set = all_samples[:train_stop]
    validation_set = all_samples[train_stop:val_stop]
    test_set = all_samples[val_stop:]
    print("have train, test, val for",informal_name)


    training_prints_filename = rnn_folder + "/training_prints.txt" # this is the same one train_rnn will write into later, so just agree with it

    rnn = RNNTokenPredictor(alphabet,args.input_dim,args.hidden_dim,args.num_layers,\
        args.RNNClass,dropout=args.dropout)

    with open(training_prints_filename,"w") as f:
        print("currently training with train and validation sets of size:",\
            len(train_set),"(total len:",sum([len(s) for s in train_set]),"),",\
            len(validation_set),"(total len:",sum([len(s) for s in validation_set]),"), respectively",file=f)

    print("made rnn, and beginning to train. will print train prints and final losses in:",training_prints_filename,flush=True)
    train_start_time = process_time()
    rnn = train_rnn(rnn,train_set,validation_set,rnn_folder,
            iterations_per_learning_rate=args.iterations_per_learning_rate,
            learning_rates=args.learning_rates,
            batch_size=args.batch_size,
            check_improvement_every=math.ceil(len(train_set)/args.batch_size)) 
            # might not return same rnn object as original prolly cause im doing it not how pytorch wants

    print("done training (took ",clock_str(train_start_time),"), checking last losses on train, test, val and keeping in train prints file",flush=True)	
    loss_start = process_time()

    with open(training_prints_filename,"a") as f:
        print("\n\ntotal training time according to python's time.clock:",clock_str(train_start_time),file=f)
        print("\n\nultimately reached:\nlosses:",file=f)
        rnn_final_losses = {}
        rnn_final_losses["train"] = rnn.detached_average_loss_on_group(train_set)
        print("train loss:        ",rnn_final_losses["train"],file=f,flush=True)
        rnn_final_losses["validation"] = rnn.detached_average_loss_on_group(validation_set)
        print("validation loss:   ",rnn_final_losses["validation"],file=f,flush=True)
        rnn_final_losses["test"] = rnn.detached_average_loss_on_group(test_set)
        print("test loss:         ",rnn_final_losses["test"],file=f,flush=True)

    print("done getting losses, that took:",clock_str(loss_start),flush=True)
    return rnn, rnn_folder


def parse_args():
    parser = argparse.ArgumentParser()
    # language (either spice or uhl, it wont do both)
    parser.add_argument('--spice-example',action='store_true')
    parser.add_argument('--uhl-num',type=int,default=-1)

    # train params
    parser.add_argument('--RNNClass',type=str,default="LSTM",choices=["LSTM","GRU"])
    parser.add_argument('--hidden-dim',type=int,default=50)
    parser.add_argument('--input-dim',type=int,default=10)
    parser.add_argument('--num-layers', type=int,default=2)
    parser.add_argument('--dropout',type=float,default=0.5)
    parser.add_argument('--learning-rates',type=ast.literal_eval,default=[0.01, 0.008, 0.006, 0.004, 0.002, 0.001, 0.0005, 0.0001, 5e-05])
    parser.add_argument('--iterations-per-learning-rate',type=int,default=10)
    parser.add_argument('--batch-size',type=int,default=100)
    parser.add_argument('--total-generated-train-samples',type=int,default=5000,help="only relevant when its going to make samples for you, ie when using a uhl") 
    parser.add_argument('--max-generated-train-sample-len',type=int,default=200,help="only relevant when its going to make samples for you, ie when using a uhl") 

    # # lstar extraction params
    parser.add_argument('--t-tol',type=float,default=0.1) # referred to as atol elsewhere
    parser.add_argument('--interval_width',type=float,default=0.2) # generally keep it about 2*t-tol, but pointless if that ends up being close to 1
    parser.add_argument('--dump-every',type=int,default=None)
    parser.add_argument('--num-cex-attempts',type=int,default=500)
    parser.add_argument('--max-counterexample-length',type=int,default=50)
    parser.add_argument('--max-P',type=int,default=1000)
    parser.add_argument('--max-states',type=int,default=math.inf) # effectively matching max-P by default
    parser.add_argument('--max-S',type=int,default=50)
    parser.add_argument('--lstar-time-limit',type=int,default=math.inf)
    parser.add_argument('--progress-P-print-rate',type=int,default=math.inf)
    parser.add_argument('--lstar-p-threshold',type=float,default=-1)
    parser.add_argument('--lstar-s-threshold',type=float,default=-1)

    args = parser.parse_args()
    # if not args.spice_example and None is args.uhl_num:
    #     print("pick a spice or uhl")
    #     exit()
    return args

def save_rnn_folder_name(rnn_folder):
    fname='rnn_folder_name.txt' 
    f = open(fname, 'w')
    f.write(rnn_folder)
    f.close()

def get_rnn_folder_name():
    fname='rnn_folder_name.txt' 
    f = open(fname, 'r')
    rnn_folder = f.read()
    f.close()
    return rnn_folder

def get_M_N_hack(rnn_folder, train_new_rnn = True):
    args = parse_args()
    N = load_rnn(rnn_folder)

    print("beginning lstar extraction! all will be saved and printed in subdirectories in",rnn_folder)
    M = do_lstar(N, rnn_folder, args)
    return M, N

def get_M_N(train_new_rnn = True):
    args = parse_args()
    if (train_new_rnn):
        N, rnn_folder = get_rnn(args)
        save_rnn_folder_name(rnn_folder)
    else:
        rnn_folder = get_rnn_folder_name()
        N = load_rnn(rnn_folder)

    print("beginning lstar extraction! all will be saved and printed in subdirectories in",rnn_folder)
    M = do_lstar(N, rnn_folder, args)
    return M, N

    

  