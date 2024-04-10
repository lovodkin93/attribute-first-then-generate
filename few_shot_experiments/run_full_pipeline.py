import os
import argparse
from utils import *
from run_script import main as main_func
from run_iterative_sentence_generation import main as iterative_sent_gen_main
import logging
from copy import deepcopy
from pathlib import Path
# Set the logging level to INFO
logging.basicConfig(level=logging.INFO)

def run_subtask(full_configs, subtask_name, curr_outdir, original_args_dict, indir_alignments=None):
    """
    full_configs: full pipeline configs
    subtask_name: curr subtask name (either one of "content_selection", "clustering", "iterative_sentence_generation", or "fusion_in_context")
    curr_outdir: curr subtask's outdir
    original_args_dict: full otiginal args
    indir_alignments: path to previous subtask's alignments (for content_selection this should be None or a pre-defined alignment path)
    """
    curr_configs = [elem for elem in full_configs if elem['subtask']==subtask_name][0]
    curr_configs.update({"outdir":curr_outdir,
                         "indir_alignments":indir_alignments})
    func_args = deepcopy(original_args_dict) # initialize args that didn't appear in the subask's configs file to default values
    func_args.update(curr_configs)
    if subtask_name!="iterative_sentence_generation":
        main_func(argparse.Namespace(**func_args))
    else:
        iterative_sent_gen_main(argparse.Namespace(**func_args))


def main(args):
    original_args_dict = deepcopy(args.__dict__) 
    with open(args.config_file, 'r') as f1:
        full_configs= json.loads(f1.read())
    
    # make sure all configs for all subtasks are supplied
    if not any(elem['subtask']=="content_selection" for elem in full_configs):
        raise Exception("must provide content_selection configs")
    if not set([elem['subtask'] for elem in full_configs]) in [{"content_selection", "clustering", "iterative_sentence_generation"}, {"content_selection", "fusion_in_context"}]:
        raise Exception('configs must be of the following subtasks: (1) "content_selection", "clustering", "iterative_sentence_generation"; or (2) "content_selection", "fusion_in_context"')
    
    # make sure all config files share the same split and setting
    all_splits, all_settings = [], []
    for elem in full_configs:
        with open(elem['config_file'], 'r') as f1:
            curr_configs = json.loads(f1.read())
            all_splits.append(curr_configs['split'])
            all_settings.append(curr_configs['setting'])
    if len(set(all_splits))!=1 or len(set(all_settings))!=1:
        raise Exception("all subtasks must have the same split (test/dev) and the same setting (MDS/LFQA)")

    # define and create outdir
    pipeline_subdir = "full_CoT_pipeline" if "fusion_in_context" in set([elem['subtask'] for elem in full_configs]) else "full_pipeline"
    outdir = args.outdir if args.outdir else  f"results/{all_splits[0]}/{all_settings[0]}/{pipeline_subdir}"
    logging.info(f"saving results to {outdir}")
    path = Path(outdir)
    path.mkdir(parents=True, exist_ok=True) # create outdir if doesn't exist

    intermediate_outdir = os.path.join(outdir, "itermediate_results") # subdir with results of intermediate subtasks
    path = Path(intermediate_outdir)
    path.mkdir(parents=True, exist_ok=True) # create outdir if doesn't exist

    # content selection
    content_selection_outdir = os.path.join(intermediate_outdir, "content_selection")
    logging.info("running content seletion:")
    run_subtask(full_configs=full_configs, 
                subtask_name="content_selection", 
                curr_outdir=content_selection_outdir, 
                original_args_dict=original_args_dict, 
                indir_alignments=args.indir_alignments)

    # fully-decomposted pipeline (not CoT)
    if "clustering" in [elem['subtask'] for elem in full_configs]:
        # clustering
        clustering_outdir = os.path.join(intermediate_outdir, "clustering")
        logging.info("running clustering:")
        run_subtask(full_configs=full_configs, 
                    subtask_name="clustering", 
                    curr_outdir=clustering_outdir, 
                    original_args_dict=original_args_dict,
                    indir_alignments=os.path.join(content_selection_outdir, "pipeline_format_results.json")) # the alignments are the outputs of the previous subtask (content_selection)
        # iterative_sentence_generation
        logging.info("running final iterative sentence generation:")
        run_subtask(full_configs=full_configs, 
                    subtask_name="iterative_sentence_generation", 
                    curr_outdir=outdir, 
                    original_args_dict=original_args_dict,
                    indir_alignments=os.path.join(clustering_outdir, "pipeline_format_results.json")) # the alignments are the outputs of the previous subtask (clustering)
    
    # CoT approach pipeline
    else:
        logging.info("running CoT-style fusion:")
        run_subtask(full_configs=full_configs, 
                    subtask_name="fusion_in_context", 
                    curr_outdir=outdir, 
                    original_args_dict=original_args_dict,
                    indir_alignments=os.path.join(content_selection_outdir, "pipeline_format_results.json")) # the alignments are the outputs of the previous subtask (content_selection)







if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="")
    argparser.add_argument('--config-file', type=str, required=True, help='path to json config file.')
    argparser.add_argument('-o', '--outdir', type=str, default=None, help='path to output csv.')
    argparser.add_argument('--indir-alignments', type=str, default=None, help='path to json file with alignments (if nothing is passed - goes to default under data/{setting}/{split}.json).')
    argparser.add_argument('--indir-prompt', type=str, default=None, help='path to json file with the prompt structure and ICL examples (if nothing is passed - goes to default under prompts/{setting}.json).')
    argparser.add_argument('--model-name', type=str, default="gemini-pro", help='model name')
    argparser.add_argument('--n-demos', type=int, default=2, help='number of ICL examples (default 2)')
    argparser.add_argument('--num-retries', type=int, default=1, help='number of retries of running the model.')
    argparser.add_argument('--temperature', type=float, default=0.2, help='temperature of generation')
    argparser.add_argument('--debugging', action='store_true', default=False, help='if debugging mode.')
    argparser.add_argument('--merge-cross-sents-highlights', action='store_true', default=False, help='whether to merge consecutive highlights that span across several sentences.')    
    argparser.add_argument('--CoT', action='store_true', default=False, help='whether to use a CoT approach (relevant for FiC and clustering).')    
    argparser.add_argument('--cut-surplus', action='store_true', default=False, help='whether to cut surplus text from prompts (in subtask with given highlights - everything after last highlight, and in tasks without - last prct_surplus sentences).')
    argparser.add_argument('--prct-surplus', type=float, default=None, help='for subtasks without given highlights (e.g. content_selection, e2e_only_setting, or ALCE) - what percentage of top document sents to drop in cases when the prompts are too long.')
    argparser.add_argument('--always-with-question', action='store_true', default=False, help='relevant for LFQA - whether to always add the question (also to clustering and FiC)')
    argparser.add_argument('--num-demo-changes', type=int, default=4, help='number of changing demos when the currently-chosen set of demos returns an ERROR.')
    argparser.add_argument('--rerun', action='store_true', default=False, help='if need to rerun on instances that had errors')
    argparser.add_argument('--rerun-path', type=str, default=None, help='path to rerun on (where the results are)')
    argparser.add_argument('--rerun-n-demos', type=int, default=None, help='new n_demos for rerun in cases when the current n_demos doesnt work.')
    argparser.add_argument('--rerun-temperature', type=float, default=None, help='new temperature for rerun in cases when the current temperature doesnt work.')
    argparser.add_argument('--no-prefix', action='store_true', default=False, help='ablation study where the prefix is not add.')
    args = argparser.parse_args()
    main(args)