import argparse
from utils import *
from subtask_specific_utils import *
import logging
from pathlib import Path
# Set the logging level to INFO
logging.basicConfig(level=logging.INFO)

def main(args):

    # get the outdir
    CoT_suffix = "_CoT" if args.CoT else ""
    cut_surplus_suffix="_shortened_prompts" if args.cut_surplus else ""
    merged_cross_sent_highlights_suffix = "_merged_cross_sents_sep" if args.merge_cross_sents_highlights else ""
    outdir = args.outdir if args.outdir else  f"results/{args.setting}/{args.subtask}{CoT_suffix}{cut_surplus_suffix}{merged_cross_sent_highlights_suffix}"
    outdir = os.path.join(outdir, f"{args.model_name}_num_demos_{args.n_demos}")
    logging.info(f"saving results to {outdir}")

    # create outdir if doesn't exist
    path = Path(outdir)
    path.mkdir(parents=True, exist_ok=True)

    # save as args to json file
    with open(os.path.join(outdir, "args.json"), 'w') as f1:
        f1.write(json.dumps(args.__dict__, indent=2))
    
    prompt_dict, alignments_dict = get_data(args)

    # get subtask related functions
    parse_response_func, convert_to_pipeline_style_func = get_subtask_funcs(args.subtask)


    if args.cut_surplus and args.subtask in SUBTASK_WITHOUT_GIVEN_HIGHLIGHTS and not args.prct_surplus:
        logging.error(f"when passing --cut-surplus with the subtask {args.subtask} - you need to also pass --prct-surplus")
        exit(1)


    # get subtask related prompt structures (instructions and answer-related)
    specific_prompt_details = get_subtask_prompt_structures(prompt_dict=prompt_dict, setting=args.setting, subtask=args.subtask, CoT=args.CoT, always_with_question=args.always_with_question)
    
    used_demos, prompts, additional_data = construct_prompts(prompt_dict=prompt_dict, 
                                                             alignments_dict=alignments_dict, 
                                                             n_demos=args.n_demos, 
                                                             debugging=args.debugging, 
                                                             merge_cross_sents_highlights=args.merge_cross_sents_highlights, 
                                                             specific_prompt_details=specific_prompt_details,
                                                             tkn_counter=get_token_counter(args.model_name),
                                                             no_highlights=args.subtask in SUBTASK_WITHOUT_GIVEN_HIGHLIGHTS,
                                                             cut_surplus=args.cut_surplus,
                                                             prct_surplus=args.prct_surplus)


    responses = prompt_model(prompts=prompts, 
                             model_name=args.model_name, 
                             parse_response_fn=parse_response_func, 
                             num_retries=args.num_retries, 
                             temperature=args.temperature)

    
    ############# SAVE #############
    # combine results with all instances' details
    final_results = {key:dict() for key in responses.keys()}
    for instance_name, resp in responses.items():
        final_results[instance_name].update(additional_data[instance_name])
        final_results[instance_name]['gold_summary'] = [elem['response'] for elem in alignments_dict if elem['unique_id']==instance_name][0]
        if args.subtask=="FiC" and args.CoT and not "alignments" in resp.keys(): # when there is an ERROR in the FiC-CoT
            final_results[instance_name]["alignments"] = []
        final_results[instance_name].update(resp)

    pipeline_format_results = None
    if convert_to_pipeline_style_func:
        try:
            pipeline_format_results = convert_to_pipeline_style_func(final_results, alignments_dict)
        except:
            logging.info("The coversion to the pipeline format wasn't successful - please check.")

    # save
    save_results(outdir, used_demos, final_results, pipeline_format_results)        

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="")
    argparser.add_argument('--split', type=str, default="test", help='data split (test or dev)')
    argparser.add_argument('--setting', type=str, required=True, help='setting (MDS or LFQA)')
    argparser.add_argument('--subtask', type=str, required=True, help='subtask to run (content_selection, clustering, FiC, e2e_only_setting, ALCE)')
    argparser.add_argument('--indir-alignments', type=str, default=None, help='path to json file with alignments (if nothing is passed - goes to default under data/{setting}/{split}.json).')
    argparser.add_argument('--indir-prompt', type=str, default=None, help='path to json file with the prompt structure and ICL examples (if nothing is passed - goes to default under prompts/{setting}.json).')
    argparser.add_argument('-o', '--outdir', type=str, default=None, help='path to output csv.')
    argparser.add_argument('--model-name', type=str, default="meta-llama/Llama-2-7b-hf", help='model name')
    argparser.add_argument('--n-demos', type=int, default=2, help='number of ICL examples (default 2)')
    argparser.add_argument('--num-retries', type=int, default=1, help='number of retries of running the model.')
    argparser.add_argument('--temperature', type=float, default=0.2, help='temperature of generation')
    argparser.add_argument('--debugging', action='store_true', default=False, help='if debugging mode.')
    argparser.add_argument('--merge-cross-sents-highlights', action='store_true', default=False, help='whether to merge consecutive highlights that span across several sentences.')    
    argparser.add_argument('--CoT', action='store_true', default=False, help='whether to use a CoT approach (relevant for FiC and clustering).')    
    argparser.add_argument('--cut-surplus', action='store_true', default=False, help='whether to cut surplus text from prompts (in subtask with given highlights - everything after last highlight, and in tasks without - last prct_surplus sentences).')
    argparser.add_argument('--prct-surplus', type=float, default=None, help='for subtasks without given highlights (e.g. content_selection, e2e_only_setting, or ALCE) - what percentage of top document sents to drop in cases when the prompts are too long.')
    argparser.add_argument('--always-with-question', action='store_true', default=False, help='relevant for LFQA - whether to always add the question (also to clustering and FiC)')
    args = argparser.parse_args()
    main(args)