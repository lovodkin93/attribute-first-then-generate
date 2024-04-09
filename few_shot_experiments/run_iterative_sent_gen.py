import json
import os
from tqdm import tqdm
import numpy as np
from utils import *
import logging
from pathlib import Path
from copy import deepcopy
# Set the logging level to INFO
logging.basicConfig(level=logging.INFO)
from subtask_specific_utils import *

def keep_specific_occurrences(s, substring, occurrences):
    """Keep only specific occurrences of a substring in a string."""
    parts = []
    count = 0
    last_index = 0
    while True:
        index = s.find(substring, last_index)
        if index == -1:
            # Append the remaining part of the string
            parts.append(s[last_index:])
            break
        count += 1
        if count in occurrences:
            # Include the substring in the result
            parts.append(s[last_index:index + len(substring)])
        else:
            # Exclude the substring from the result
            parts.append(s[last_index:index])
        # Update the last index
        last_index = index + len(substring)
    return ''.join(parts)

def adapt_demo(train_item, curr_cluster_ind, no_prefix):
    if no_prefix: # ablation - no prefix
            division_ind = 0
    else:
        division_ind = curr_cluster_ind if curr_cluster_ind<len(train_item['planning'])-1 else len(train_item['planning'])-1
    prefix = " ".join([elem['output'] for elem in train_item['planning'][:division_ind]])
    next_sentence = train_item['planning'][division_ind]['output']
    relevant_highlights = train_item['planning'][division_ind]['highlights_cluster']
    adapted_docs = []
    for i,doc in enumerate(train_item['docs']):
        adapted_doc = deepcopy(doc['text'])
        doc_relevant_highlights = [elem for elem in relevant_highlights if elem['doc']==i+1]
        if doc_relevant_highlights:
            doc_relevant_highlights = sorted(doc_relevant_highlights[0]['relative_highlights'])

        # leave only the relevant highlights
        adapted_doc = keep_specific_occurrences(doc['text'], "{HS}", doc_relevant_highlights)
        adapted_doc = keep_specific_occurrences(adapted_doc, "{HE}", doc_relevant_highlights)

        # make sure the text didn't change
        assert adapted_doc.replace("{HS}", "").replace("{HE}", "") == doc['text'].replace("{HS}", "").replace("{HE}", "")
        # make sure the correct number of highlights remained
        assert adapted_doc.count("{HS}")==len(doc_relevant_highlights)
        assert adapted_doc.count("{HE}")==len(doc_relevant_highlights)
        
        # make sure the correct highlights remained
        original_highlights = extract_highlights(doc['text'], "{HS}", "{HE}")
        assert extract_highlights(adapted_doc, "{HS}", "{HE}") == [elem for h_i,elem in enumerate(original_highlights) if h_i+1 in doc_relevant_highlights]

        adapted_docs.append({"text":adapted_doc})
    
    return_dict = {"answer" : next_sentence,
                   "docs" : adapted_docs,
                   "prefix" : prefix}

    if "question" in train_item:
        return_dict.update({"question":train_item["question"]})

    return return_dict

def construct_curr_non_demo_part(curr_docs, alignments, highlight_start_tkn_placeholder, highlight_end_tkn_placeholder, merge_cross_sents_highlights, prompt_dict, prefix, instruction_prompt, prompt_structure, always_with_question : bool, instance_question: str, cut_surplus: bool = False):
    highlighted_texts = get_highlighted_doc(docs=curr_docs, 
                                        highlights=alignments,
                                        highlight_start_tkn = highlight_start_tkn_placeholder,
                                        highlight_end_tkn=highlight_end_tkn_placeholder,
                                        merge_cross_sents_highlights=merge_cross_sents_highlights)
    
    # leave only documents with highlights - to help deal with too long prompts and also better focus the model
    highlighted_texts = {key:value for key,value in highlighted_texts.items() if highlight_start_tkn_placeholder in value}

    if cut_surplus: # if need to shorten prompt - remove everything after last highlight
        highlighted_texts = {doc_name:rmv_txt_after_last_highlight(doc_text, highlight_end_tkn_placeholder) for doc_name,doc_text in highlighted_texts.items()}

    docs_order = [{"doc_name":doc_name, "doc_text":doc_text} for doc_name,doc_text in highlighted_texts.items()]
    doc_names = [dct["doc_name"] for dct in docs_order]
    eval_item = {"docs":[{'text':dct["doc_text"]} for dct in docs_order]}
    eval_item["prefix"]=prefix 
    if always_with_question and instance_question:
        eval_item['question'] = instance_question

    answer_related_prompts = {"answer_prompt":prompt_dict["answer_next_cluster_fusion_prompt"],
                              "answer_highlights_listing_prompt":prompt_dict["answer_highlights_listing_prompt"]}

    curr_prompt, _ = make_demo(
        item=eval_item, prompt=prompt_structure, doc_prompt=prompt_dict["doc_prompt"], 
        instruction=instruction_prompt, answer_related_prompts=answer_related_prompts,
        highlight_start_tkn=prompt_dict["highlight_start_tkn"], highlight_end_tkn=prompt_dict["highlight_end_tkn"],
        test=True
    )
    return curr_prompt, docs_order

def construct_curr_prompt(prompt_dict, alignments, curr_docs, used_demos, curr_cluster_ind, prefix, merge_cross_sents_highlights, tkn_counter: Dict, cut_surplus: bool = False, always_with_question: bool = False, instance_question: str = None, no_prefix: bool = False):
    # Generate the demonstration part
    highlight_start_tkn_placeholder = "{HS}"
    highlight_end_tkn_placeholder="{HE}"
    head_prompt, head_prompt_shorter = "", ""
    for train_item in used_demos:
        adapted_train_item = adapt_demo(train_item, curr_cluster_ind, no_prefix)

        if always_with_question and "question" in adapted_train_item:
            instruction_prompt = prompt_dict["instruction-next-cluster-fusion-with-question"]
            prompt_structure = prompt_dict["demo_prompt_next_cluster_fusion_with_question"]
        else:
            instruction_prompt = prompt_dict["instruction-next-cluster-fusion"]
            prompt_structure = prompt_dict["demo_prompt_next_cluster_fusion"]

        answer_related_prompts = {"answer_prompt":prompt_dict["answer_next_cluster_fusion_prompt"],
                                  "answer_highlights_listing_prompt":prompt_dict["answer_highlights_listing_prompt"]}

        curr_prompt_demo, _ = make_demo(
            item=adapted_train_item, prompt=prompt_structure, doc_prompt=prompt_dict["doc_prompt"], 
            instruction=instruction_prompt, answer_related_prompts=answer_related_prompts,
            highlight_start_tkn=prompt_dict["highlight_start_tkn"], highlight_end_tkn=prompt_dict["highlight_end_tkn"]
        )
        head_prompt += curr_prompt_demo
        head_prompt += prompt_dict["demo_sep"]

        # get shorter version
        adapted_train_item_shorter = {key:[{doc_key:rmv_txt_after_last_highlight(doc_value, highlight_end_tkn_placeholder) if doc_key=="text" else doc_value 
                                    for doc_key,doc_value in elem.items()} for elem in value] if key=="docs" else value for key,value in adapted_train_item.items()}
        # remove demos that had highlights in the current iteration
        adapted_train_item_shorter = {key:[elem for elem in value if elem['text']] if key=="docs" else value for key,value in adapted_train_item_shorter.items()}


        curr_prompt_demo_shorter, _ = make_demo(
            item=adapted_train_item_shorter, prompt=prompt_structure, doc_prompt=prompt_dict["doc_prompt"], 
            instruction=instruction_prompt, answer_related_prompts=answer_related_prompts,
            highlight_start_tkn=prompt_dict["highlight_start_tkn"], highlight_end_tkn=prompt_dict["highlight_end_tkn"]
        )
        head_prompt_shorter += curr_prompt_demo_shorter
        head_prompt_shorter += prompt_dict["demo_sep"]

    curr_prompt, docs_order = construct_curr_non_demo_part(curr_docs, alignments, highlight_start_tkn_placeholder, highlight_end_tkn_placeholder, merge_cross_sents_highlights, prompt_dict, prefix, instruction_prompt, prompt_structure, always_with_question, instance_question)     
    
    # the current prompt is too long - need to "cut" the surplus texts (or pre-decided to cut it)
    if cut_surplus or tkn_counter["tkn_counter"].token_count(head_prompt + curr_prompt)>=tkn_counter["tkn_max_limit"]: 
        curr_prompt, docs_order_shorter = construct_curr_non_demo_part(curr_docs, alignments, highlight_start_tkn_placeholder, highlight_end_tkn_placeholder, merge_cross_sents_highlights, prompt_dict, prefix, instruction_prompt, prompt_structure, always_with_question, instance_question, cut_surplus=True)
        final_prompt = head_prompt_shorter + curr_prompt
    else:
        final_prompt = head_prompt + curr_prompt
        docs_order_shorter = []
    highlighted_docs = [{"doc_name":elem["doc_name"],
                        "doc_text":elem["doc_text"].replace(highlight_start_tkn_placeholder, prompt_dict["highlight_start_tkn"]).replace(highlight_end_tkn_placeholder, prompt_dict["highlight_end_tkn"])} for elem in docs_order]
    non_highlighted_docs = [{"doc_name":elem["doc_name"],
                             "doc_text":elem["doc_text"].replace(highlight_start_tkn_placeholder, "").replace(highlight_end_tkn_placeholder, "")} for elem in docs_order]
    highlighted_docs_shorter = [{"doc_name":elem["doc_name"],
                                 "doc_text":elem["doc_text"].replace(highlight_start_tkn_placeholder, prompt_dict["highlight_start_tkn"]).replace(highlight_end_tkn_placeholder, prompt_dict["highlight_end_tkn"])} for elem in docs_order_shorter]
    non_highlighted_docs_shorter = [{"doc_name":elem["doc_name"],
                                     "doc_text":elem["doc_text"].replace(highlight_start_tkn_placeholder, "").replace(highlight_end_tkn_placeholder, "")} for elem in docs_order_shorter]

    additional_data = {"highlighted_docs" : highlighted_docs,
                       "non_highlighted_docs" : non_highlighted_docs,
                       "highlighted_docs_shorter" : highlighted_docs_shorter,
                       "non_highlighted_docs_shorter" : non_highlighted_docs_shorter,
                       "curr_alignments" : alignments,
                       "curr_prefix" : prefix}
    return final_prompt, additional_data

def parse_itertive_sent_gen_response(response, prompt):
    parsed_response = response.strip()
    if parsed_response.lower().strip().startswith("answer:"):
        parsed_response = parsed_response.strip()[len("answer:"):].strip()
    if parsed_response.lower().strip().startswith("the next sentence is:"):
        parsed_response = parsed_response.strip()[len("the next sentence is:"):].strip()

    assert not "next sentence" in parsed_response, f"\"some variation of \"next sentence\" appeared in the response: {parsed_response}"
    
    return {"final_output":parsed_response,
            "full_model_response":response}

def iterative_sent_gen_prompting(alignments_dict, prompt_dict, used_demos, model_name, num_retries, debugging, n_demos, num_demo_changes, temperature, merge_cross_sents_highlights, tkn_counter, cut_surplus: bool = False, always_with_question: bool = False, no_prefix: bool = False):
    final_data_instances = {}
    alignments_dict = [elem for elem in alignments_dict if not elem['unique_id'] in ['test59', 'test62', 'test63', 'test67', 'test91']]
    if debugging:
        alignments_dict = alignments_dict[:3]
    for instance in tqdm(alignments_dict):
        topic_name = instance['unique_id']
        curr_scuSentCharIdxs = sorted(list(set(elem['scuSentCharIdx'] for elem in instance['set_of_highlights_in_context'])))
        final_data_instances[topic_name] = {"non_highlighted_docs_full": {elem["documentFile"]:elem["rawDocumentText"] for elem in instance["documents"]}}
        generated_summary_sents = []
        generation_history = []
        for i,scuSentCharIdx in enumerate(curr_scuSentCharIdxs):
            curr_alignments = [elem for elem in instance['set_of_highlights_in_context'] if elem['scuSentCharIdx']==scuSentCharIdx]
            curr_used_demos = used_demos
            no_errors = False
            for demo_i in range(num_demo_changes+1):
                
                curr_step_prompt, curr_additional_data = construct_curr_prompt(prompt_dict=prompt_dict,
                                                                               alignments=curr_alignments, 
                                                                               curr_docs={elem["documentFile"]:elem["rawDocumentText"] for elem in instance["documents"]},
                                                                               used_demos=curr_used_demos, 
                                                                               curr_cluster_ind=i,
                                                                               prefix="" if no_prefix else " ".join(generated_summary_sents),
                                                                               merge_cross_sents_highlights=merge_cross_sents_highlights,
                                                                               tkn_counter=tkn_counter,
                                                                               cut_surplus=cut_surplus,
                                                                               always_with_question=always_with_question,
                                                                               instance_question=instance['query'] if 'query' in instance.keys() else None,
                                                                               no_prefix = no_prefix)
                
                responses = prompt_model(prompts={topic_name:curr_step_prompt}, 
                                         model_name=model_name, 
                                         parse_response_fn=parse_itertive_sent_gen_response, 
                                         num_retries=num_retries,
                                         verbose=False,
                                         temperature=temperature)

                
                if not responses[topic_name]['final_output'].startswith("ERROR"):
                    generated_summary_sents.append(responses[topic_name]['final_output'])
                    curr_generation_history = curr_additional_data
                    curr_generation_history.update(responses[topic_name])
                    updated_demos = curr_used_demos if demo_i>0 else [] # save only cases when the demos were updated
                    curr_generation_history.update({"updated_demos":updated_demos})
                    generation_history.append(curr_generation_history)
                    no_errors = True
                    break

                # use different demos
                new_used_demos_ids = np.random.choice(len(prompt_dict["demos"]), n_demos, replace=False)
                curr_used_demos = [prompt_dict["demos"][demo_id] for demo_id in new_used_demos_ids]

            if not no_errors: # no need to continue problematic generations
                logging.info(f"in instance {topic_name} couldn't generate output")
                generated_summary_sents.append(responses[topic_name]['final_output'])
                curr_generation_history = curr_additional_data
                curr_generation_history.update(responses[topic_name])
                updated_demos = [] # save only cases when the demos were updated
                curr_generation_history.update({"updated_demos":updated_demos})
                generation_history.append(curr_generation_history)
                break
            
        final_data_instances[topic_name].update({"generated_summary_sents":generated_summary_sents,
                                                    "generation_history":generation_history})
    return final_data_instances

def get_set_of_highlights_in_context_iterative_sent_gen(curr_instance, curr_original_inst):
    highlights_in_context_list = []
    final_output = ""
    curr_scuSentCharIdx = 0
    for gen_inst in curr_instance['generation_history']:
            curr_alignments = [{key:value for key,value in elem.items() if key!="prefix"} for elem in gen_inst['curr_alignments']]
            # change alignments scuSentence
            curr_alignments = [{key:value if key!="scuSentence" else gen_inst['final_output'] for key,value in elem.items()} for elem in curr_alignments]
            # change alignments scuSentCharIdx
            curr_alignments = [{key:value if key!="scuSentCharIdx" else float(curr_scuSentCharIdx) for key,value in elem.items()} for elem in curr_alignments]
            # save curr highlights in context
            highlights_in_context_list+=curr_alignments
            # update the final output and curr_scuSentCharIdx
            final_output = final_output + gen_inst['final_output'] + " "
            curr_scuSentCharIdx = curr_scuSentCharIdx + len(gen_inst['final_output']) + 1

    assert all(final_output[int(elem['scuSentCharIdx']):].startswith(elem['scuSentence']) for elem in highlights_in_context_list), "scuSentence doesn't match scuSentCharIdx"
    assert sum([len(elem['curr_alignments']) for elem in curr_instance['generation_history']]) == len(highlights_in_context_list), "num of final highlights in context doesn't match original number of highlights in context"
    return highlights_in_context_list, final_output.strip()

def convert_iterative_sent_gen_to_pipeline_format(results, alignments_dict, *args):
    nlp = spacy.load("en_core_web_sm")
    pipeline_style_data = []
    for key,value in results.items():
        curr_original_inst = deepcopy([elem for elem in alignments_dict if elem["unique_id"]==key][0])
        curr_documents = curr_original_inst["documents"]
        highlights_in_context, final_output = get_set_of_highlights_in_context_iterative_sent_gen(value, curr_original_inst)
        curr_original_inst.update({"set_of_highlights_in_context":highlights_in_context, 
                                   "response" : final_output, 
                                   "response_sents" : value['generated_summary_sents'], 
                                   "gold_summary" : value["gold_summary"]})
        pipeline_style_data.append(curr_original_inst)
    return pipeline_style_data

def main(args):
    if not args.config_file and (not args.setting or not args.split):
        raise Exception("If no config file is passed, then must explicitly determine setting and split.")

    # if config_file is passed - load its arguments
    if args.config_file:
        args = update_args(args)

    # get the outdir
    no_prefix_suffix = "_no_prefix" if args.no_prefix else ""
    merged_cross_sent_highlights_suffix = "_merged_cross_sents_sep" if args.merge_cross_sents_highlights else ""
    if args.rerun:
        outdir = args.rerun_path
    else:
        outdir = args.outdir if args.outdir else  f"results/{args.setting}/iterative_sent_gen{no_prefix_suffix}{merged_cross_sent_highlights_suffix}" 
    logging.info(f"saving results to {outdir}")

    # create outdir if doesn't exist
    path = Path(outdir)
    path.mkdir(parents=True, exist_ok=True)
    
    n_demos = args.n_demos 
    model_name = args.model_name 
    debugging = args.debugging 
    num_retries = args.num_retries
    num_demo_changes = args.num_demo_changes
    temperature = args.temperature
    merge_cross_sents_highlights = args.merge_cross_sents_highlights
    always_with_question = args.always_with_question
    no_prefix = args.no_prefix


    if args.rerun:
        if not args.rerun_path:
            raise Exception("if passing --rerun, also pass relevant --rerun-path.")
        
        with open(os.path.join(args.rerun_path, "args.json"), 'r') as f1:
            original_args = json.loads(f1.read())
        
        # get original args
        model_name = original_args['model_name']
        num_retries = original_args['num_retries']
        debugging = original_args['debugging']
        n_demos = args.rerun_n_demos if args.rerun_n_demos else original_args['n_demos']
        num_demo_changes = original_args['num_demo_changes']
        temperature = args.rerun_temperature if args.rerun_temperature else original_args['temperature']
        merge_cross_sents_highlights = original_args['merge_cross_sents_highlights'] if 'merge_cross_sents_highlights' in original_args.keys() else False
        always_with_question = original_args['always_with_question'] if 'always_with_question' in original_args.keys() else False
        no_prefix = original_args['no_prefix'] if 'no_prefix' in original_args.keys() else False
        
        prompt_dict, alignments_dict = get_data(original_args)
        org_alignments_dict = deepcopy(alignments_dict) # needed for the conversion to pipeline style

        with open(os.path.join(args.rerun_path, "used_demonstrations.json"), 'r') as f1:
            used_demos = json.loads(f1.read())

        with open(os.path.join(args.rerun_path, "results.json"), 'r') as f1:
            original_results = json.loads(f1.read())    
        
        # filter only instances with errors
        instances_with_errors = [key for key,value in original_results.items() if any("ERROR" in elem for elem in value['generated_summary_sents'])]
        alignments_dict = [elem for elem in alignments_dict if elem['unique_id'] in instances_with_errors]

    else:
        # save as args to json file
        with open(os.path.join(outdir, "args.json"), 'w') as f1:
            f1.write(json.dumps(args.__dict__, indent=2))

        prompt_dict, alignments_dict = get_data(args)
        org_alignments_dict = deepcopy(alignments_dict) # needed for the conversion to pipeline style (and when rerunning - the origin alters)
    
        used_demos_ids = np.random.choice(len(prompt_dict["demos"]), n_demos, replace=False) if not debugging else [0,2]
        used_demos = [prompt_dict["demos"][demo_id] for demo_id in used_demos_ids]

    responses = iterative_sent_gen_prompting(alignments_dict=alignments_dict, 
                                             prompt_dict=prompt_dict, 
                                             used_demos=used_demos,
                                             model_name=model_name,
                                             num_retries=num_retries,
                                             debugging=debugging,
                                             n_demos=n_demos,
                                             num_demo_changes=num_demo_changes,
                                             temperature=temperature,
                                             merge_cross_sents_highlights=merge_cross_sents_highlights,
                                             tkn_counter=get_token_counter(model_name),
                                             cut_surplus=args.cut_surplus,
                                             always_with_question=always_with_question,
                                             no_prefix=no_prefix)
    
    ############# SAVE #############
    # combine results with the gold summaries
    if args.rerun:
            final_results = original_results
    else:
        final_results = {key:dict() for key in responses.keys()}
    for instance_name, resp in responses.items():
        final_results[instance_name]['gold_summary'] = [elem['response'] for elem in alignments_dict if elem['unique_id']==instance_name][0]
        final_results[instance_name].update(resp)
    

    pipeline_format_results = None
    try:
        pipeline_format_results = convert_iterative_sent_gen_to_pipeline_format(final_results, org_alignments_dict)
    except:
        logging.info("The coversion to the pipeline format wasn't successful - please check.")

    # save
    save_results(outdir, used_demos, final_results, pipeline_format_results)
        

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="")
    argparser.add_argument('--config-file', type=str, default=None, help='path to json config file. Should come instead of all the other parameters')
    argparser.add_argument('--indir-alignments', type=str, default=None, help='path to json file with alignments (if nothing is passed - goes to default under data/{setting}/{split}.json).')
    argparser.add_argument('--indir-prompt', type=str, default=None, help='path to json file with the prompt structure and ICL examples (if nothing is passed - goes to default under prompts/{setting}.json).')
    argparser.add_argument('--setting', type=str, default=None, help='setting (MDS or LFQA)')
    argparser.add_argument('--split', type=str, default=None, help='data split (test or dev)')
    argparser.add_argument('-o', '--outdir', type=str, default=None, help='path to output csv.')
    argparser.add_argument('--model-name', type=str, default="gemini-pro", help='model name')
    argparser.add_argument('--n-demos', type=int, default=2, help='number of ICL examples (default 2)')
    argparser.add_argument('--num-retries', type=int, default=1, help='number of retries of running the model.')
    argparser.add_argument('--num-demo-changes', type=int, default=4, help='number of changing demos when the currently-chosen set of demos returns an ERROR.')
    argparser.add_argument('--temperature', type=float, default=0.2, help='temperature of generation')
    argparser.add_argument('--rerun', action='store_true', default=False, help='if need to rerun on instances that had errors')
    argparser.add_argument('--rerun-path', type=str, default=None, help='path to rerun on (where the results are)')
    argparser.add_argument('--rerun-n-demos', type=int, default=None, help='new n_demos for rerun in cases when the current n_demos doesnt work.')
    argparser.add_argument('--rerun-temperature', type=float, default=None, help='new temperature for rerun in cases when the current temperature doesnt work.')
    argparser.add_argument('--debugging', action='store_true', default=False, help='if debugging mode.')
    argparser.add_argument('--merge-cross-sents-highlights', action='store_true', default=False, help='whether to merge consecutive highlights that span across several sentences.')    
    argparser.add_argument('--cut-surplus', action='store_true', default=False, help='whether to cut surplus text from prompts (everything after last highlight).')
    argparser.add_argument('--always-with-question', action='store_true', default=False, help='relevant for LFQA - whether to add the question')
    argparser.add_argument('--no-prefix', action='store_true', default=False, help='ablation study where the prefix is not add.')
    args = argparser.parse_args()
    main(args)