import json
import argparse
import os
import pandas as pd
from tqdm import tqdm
import numpy as np
from typing import List, Dict
import re
import psutil
import GPUtil
import logging
import string
import time
import openai
import pathlib
import textwrap
import google.generativeai as genai
# from IPython.display import display
# from IPython.display import Markdown

openai.api_key = os.getenv("OPENAI_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
genai_model = None

# Set the logging level to INFO
logging.basicConfig(level=logging.INFO)

SPAN_SEP="<HIGHLIGHT_SEP>"
SENT_SEP="<SENT_SEP>"
SUBTASK_WITHOUT_GIVEN_HIGHLIGHTS = ["content_selection", "e2e_only_setting", "ALCE", "iterative_blue_print"]

class TokenCounter:
    def __init__(self, model_name: str):
        """
        model_name: name of model that is being prompted
        """
        self.model_name = model_name

        if model_name == "gemini-pro":
            self.model =  genai.GenerativeModel(model_name)
        else:
            raise NotImplementedError(f"Token counter for {model_name} not supported yet. Please add support in the constructor and token_count functions.")
    
    def token_count(self, prompt):
        if self.model_name == "gemini-pro":  
            return self.model.count_tokens(prompt).total_tokens
        else:
            raise NotImplementedError(f"token_count for {self.model_name} not supported yet. Please add support.")

def update_args(args):
    """update args with arguments from config file"""
    with open(args.config_file, 'r') as f1:
        updated_args = json.loads(f1.read())
    additional_args = {key:value for key,value in args.__dict__.items() if not key in updated_args.keys()}
    assert not set(additional_args.keys()).intersection(set(updated_args.keys())), "overlapping keys"
    updated_args.update(additional_args)
    return argparse.Namespace(**updated_args)


def get_token_counter(model_name):
    if model_name=="gemini-pro":
        return {"tkn_counter" : TokenCounter(model_name),
                "tkn_max_limit" : 30720}
    else:
        raise Exception(f"not supported yet for {model_name}. Please add a token counter and max limit for {model_name}")

def highlight_sep_strip(txt):
    "remove trailing and opening SPAN_SEP"
    txt = txt.strip()
    if txt.startswith(SPAN_SEP):
        txt = txt[len(SPAN_SEP):]
    if txt.endswith(SPAN_SEP):
        txt = txt[:-len(SPAN_SEP)]
    return txt

def get_max_memory():
    """Get the maximum memory available for the currently visible GPU cards GPU, as well as, CPU for loading models."""
    max_memory_dict = dict()

    # GPU
    # Get list of GPUs that are visible to the process
    visible_devices = os.getenv('CUDA_VISIBLE_DEVICES')
    if visible_devices:
        visible_devices = [int(dev) for dev in visible_devices.split(',')]
    else:
        # If CUDA_VISIBLE_DEVICES is not set, consider all GPUs as visible
        visible_devices = list(range(len(GPUtil.getGPUs())))
    
    # Get list of all GPUs (both visible and not visible)
    gpus = GPUtil.getGPUs()

    visible_gpu_cnt = 0
    for i, gpu in enumerate(gpus):
        if i in visible_devices:
            free_in_GB = int(gpu.memoryFree / 1024)
            
            if visible_gpu_cnt == 0: # the first card needs more memory
                if free_in_GB<60:
                    raise Exception("Make sure you first visible GPU card has at least 60GiB available in memory.")
                max_memory = f'{free_in_GB-30}GiB'
            else:
                if free_in_GB<40:
                    raise Exception("Make sure all you visible GPU cards have at least 40GiB available in memory.")
                max_memory = f'{free_in_GB-10}GiB'
                
            max_memory_dict[visible_gpu_cnt] = max_memory
            visible_gpu_cnt+=1
    
    # CPU
    available_memory = psutil.virtual_memory().available

    # Convert bytes to gigabytes for easier reading
    available_memory_gb = available_memory / (1024 ** 3)

    if available_memory_gb<100:
        raise Exception("Make sure there are at least 100GiB available in the CPU memory.")
    max_memory_dict['cpu'] = f"{min(int(available_memory_gb/2), 100)}GiB"

    gpu_max_memory_used_str = "\n".join([f"card {str(visible_devices[gpu_i])}: {max_memory_dict[gpu_i]}" for gpu_i in range(len(max_memory_dict)-1)])
    max_memory_used_str = f"GPU:\n{gpu_max_memory_used_str}\nCPU:\n{max_memory_dict['cpu']}"
    logging.info(f'max memory used:\n{max_memory_used_str}')
    return max_memory_dict

def find_substring_indices(s, sub):
    indices = []
    i = s.find(sub)
    while i >= 0:
        indices.append(i)
        i = s.find(sub, i + 1)
    return indices

def longest_common_subsequence(list1, list2):
    m = len(list1)
    n = len(list2)

    # Initializing the matrix
    dp = [[None]*(n+1) for i in range(m+1)]
    
    # Building dp matrix in bottom-up way
    for i in range(m+1):
        for j in range(n+1):
            if i == 0 or j == 0 :
                dp[i][j] = 0
            elif list1[i-1] == list2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    # Following the backtrack from the bottom-right to find the LCS
    i = m
    j = n
    lcs = []
    indices1 = []
    indices2 = []

    while i > 0 and j > 0:
        if list1[i-1] == list2[j-1]:
            lcs.insert(0, list1[i-1])
            indices1.insert(0, i-1)
            indices2.insert(0, j-1)
            i -= 1
            j -= 1
        elif dp[i-1][j] > dp[i][j-1]:
            i -= 1
        else:
            j -= 1

    return lcs, indices1, indices2

def rmv_txt_after_last_highlight(text, highlight_end_tkn):
    shortened_text = highlight_end_tkn.join(text.split(highlight_end_tkn)[:-1]) + highlight_end_tkn
    # if there were no highlights in the current text - then return an empty string
    if shortened_text.strip()==highlight_end_tkn:
        return ""
    return shortened_text

def rmv_spaces_and_punct(txt):
    return txt.lower().translate(str.maketrans('', '', string.whitespace + string.punctuation))

def remove_spaces_and_punctuation(text):
    cleaned = re.sub(r'[' + string.punctuation + string.whitespace + '\s]', '', text)
    cleaned = ''.join([char for char in cleaned if char.isalnum()])
    
    return cleaned

def find_substring(s, sb):
    
    # if text appears as it is - return its indices
    if sb.lower().strip() in s.lower():
        start_index = s.lower().index(sb.lower().strip())
        return start_index, start_index + len(sb.strip())

    # Remove spaces and punctuation from both strings
    modified_s = remove_spaces_and_punctuation(s).lower()
    modified_sb = remove_spaces_and_punctuation(sb).lower()

    # Find the modified substring in the modified string
    index = modified_s.find(modified_sb)

    # If substring is not found, return -1 for both start and end
    if index == -1:
        return -1, -1

    # Find the actual start index in the original string
    actual_start_index = 0
    count = 0
    for char in s:
        is_char_removed = remove_spaces_and_punctuation(char) == ''
        if count == index and not is_char_removed: # the second term - to not stop count when getting to a space/punctuation
            break
        if not is_char_removed:
            count += 1
        actual_start_index += 1

    # Find the actual end index in the original string
    actual_end_index = actual_start_index
    modified_sb_length = len(modified_sb)
    while modified_sb_length > 0:
        if s[actual_end_index].isalnum():
            modified_sb_length -= 1
        actual_end_index += 1

    assert remove_spaces_and_punctuation(sb.lower())==remove_spaces_and_punctuation(s[actual_start_index:actual_end_index].lower()), "found substring doesn't match indices" # make sure span align with sb
    return actual_start_index, actual_end_index

def get_highlighted_doc(docs, highlights, highlight_start_tkn, highlight_end_tkn, merge_cross_sents_highlights: bool = False, doc_sents: List = None):
    highlighted_docs = {}
    for doc_name, doc_text in docs.items():
        curr_highlights = [elem for elem in highlights if elem['documentFile']==doc_name]
        
        if not merge_cross_sents_highlights: # merge highlights for each separate sentence
            curr_merged_spans = []
            for docSentCharIdx in set([elem['docSentCharIdx'] for elem in curr_highlights]):
                curr_docSent_highlights = [elem for elem in curr_highlights if elem['docSentCharIdx']==docSentCharIdx]
                curr_docSent_highlights_spans = [elem['docSpanOffsets'] for elem in curr_docSent_highlights]

                # make sure the docSpanOffsets align with the docSpanText
                assert all(highlight_sep_strip(curr_docSent_highlights[i]['docSpanText'].replace(SENT_SEP, '')).replace(" ", "").replace("\n", "").lower()==highlight_sep_strip(SPAN_SEP.join([doc_text[span[0]:span[1]] for span in instance])).replace(" ", "").replace("\n", "").lower() for i,instance in enumerate(curr_docSent_highlights_spans)), f"not all docSpanOffsets align with the docSpanText for docSentCharIdx {docSentCharIdx}"
                
                # merge spans
                curr_docSent_merged_spans = merge_spans(curr_docSent_highlights_spans)
                curr_merged_spans.extend(curr_docSent_merged_spans)
        
        else:
            curr_highlights_spans = [elem['docSpanOffsets'] for elem in curr_highlights]
            
            # make sure the docSpanOffsets align with the docSpanText
            assert all(highlight_sep_strip(curr_highlights[i]['docSpanText'].replace(SENT_SEP, '')).replace(" ", "").replace("\n", "").lower()==highlight_sep_strip(SPAN_SEP.join([doc_text[span[0]:span[1]] for span in instance])).replace(" ", "").replace("\n", "").lower() for i,instance in enumerate(curr_highlights_spans)), "not all docSpanOffsets align with the docSpanText"

            # merge spans
            curr_merged_spans = merge_spans(curr_highlights_spans)

        # remove duplicates
        curr_merged_spans = [list(tpl) for tpl in set([tuple(elem) for elem in curr_merged_spans])]

        # clean spans of sentences that only consist of spaces and punctuations
        curr_merged_spans = [span for span in curr_merged_spans if rmv_spaces_and_punct(doc_text[span[0]:span[1]])]

        # order spans according to start idx
        curr_merged_spans = sorted(curr_merged_spans, key=lambda x: x[0])

        # make sure all end idxs are ordered as well (no overlapping)
        assert sorted(curr_merged_spans, key=lambda x: x[1]) == curr_merged_spans, "merged spans' end idx order doesn't align with their start idx order"
        # make sure all start idxs are smaller than end idxs
        assert all(elem[0]<elem[1] for elem in curr_merged_spans), "some merged spans exhibit start_idx>=end_idx"
        
        # on rare occasions, when there are identical spans in the documents that were highlighted, their spans overlap - so remove those instances 
        if not all(j==0 or elem[0]>=curr_merged_spans[j-1][1] for j,elem in enumerate(curr_merged_spans)):
            curr_merged_spans = [curr_merged_spans[j] for j,elem in enumerate(curr_merged_spans) if j==0 or elem[0]>=curr_merged_spans[j-1][1]]

        # add highlights
        highlighted_docs[doc_name] = add_highlights(doc_text, curr_merged_spans, highlight_start_tkn, highlight_end_tkn)

        # make sure the correct spans were highlighted
        assert extract_highlights(highlighted_docs[doc_name], highlight_start_tkn, highlight_end_tkn) == [doc_text[span[0]:span[1]] for span in curr_merged_spans], "the correct spans weren't highlighted"

    return highlighted_docs

def save_results(outdir, used_demos, final_results, pipeline_format_results=None):
    # save the used demonstrations
    with open(os.path.join(outdir, "used_demonstrations.json"), 'w') as f1:
        f1.write(json.dumps(used_demos, indent=2))

    # save results as jsons
    with open(os.path.join(outdir, "results.json"), 'w') as f1:
        f1.write(json.dumps(final_results, indent=2))
    
    # save the results in the pipeline format so they can be used in the next steps of the pipeline
    # only if there weren't any problems with the conversion to pipeline format
    if pipeline_format_results:
        with open(os.path.join(outdir, "pipeline_format_results.json"), 'w') as f1:
            for instance_result in pipeline_format_results:
                f1.write(json.dumps(instance_result))
                f1.write("\n")

    # save results as csv
    df_style_data = [{'instance': key, **value} for key, value in final_results.items()]
    df_style_data = [{key:json.dumps(value) if type(value) in [list,dict] else value for key,value in instance.items()} for instance in df_style_data]
    final_results_dataframe = pd.DataFrame(df_style_data)
    final_results_dataframe.to_csv(os.path.join(outdir, "results.csv"), index=False)

def gemini_call(prompt, model_name, output_max_length: int = 2048, temperature: int = 0):
    global genai_model
    if not genai_model:
        genai_model = genai.GenerativeModel(model_name)
    generation_config = {
        "temperature": temperature,
        "max_output_tokens": output_max_length
    }

    safety_settings=[
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_NONE",
        }
    ]

    response = genai_model.generate_content(
        prompt,
        generation_config = generation_config,
        safety_settings=safety_settings
        )
    try:
        return response.text.strip()
    except Exception as e:
        raise Exception(str(e) + str(response.prompt_feedback)) 

def openai_call(prompt, model_name, output_max_length: int = 2048, temperature: int = 0):
    response = openai.chat.completions.create(
            model=model_name,
            messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ],
            max_tokens=output_max_length,
            temperature=temperature
        )
    return response.choices[0].message.content.strip()

def model_call_wrapper(prompt: str, model_name: str, parse_response_fn, output_max_length : int = 4096, num_retries: int = 5, temperature: int = 0):
    call_func = openai_call if "gpt" in model_name else gemini_call
    parsed_response, response = None, None
    for _ in range(num_retries):
        try:
            response = call_func(prompt=prompt,
                                 model_name=model_name,
                                 output_max_length=output_max_length,
                                 temperature=temperature)
            parsed_response = parse_response_fn(
                response=response,
                prompt=prompt
            )
            parsed_response['prompt']=prompt
            break
        except Exception as exception:
            print(f"{exception}. Retrying...")
            error_message = str(exception)
            time.sleep(1)
    if not parsed_response:
        if response: # there was a problem with the parsing
            return {"final_output":f"ERROR - {error_message}",
                    "full_model_response":response,
                    "prompt":prompt}
        else:
            return {"final_output":f"ERROR - {error_message}",
                    "full_model_response":"",
                    "prompt":prompt}
    else:
        return parsed_response   

def prompt_model(prompts: Dict, model_name: str, parse_response_fn, output_max_length : int = 4096, num_retries: int = 5, verbose: bool = True, temperature: int = 0):
    prompts_tpls = [(inst_name, prompt) for inst_name,prompt in prompts.items()]
    results = dict()
    prompts_tpls = tqdm(prompts_tpls) if verbose else prompts_tpls
    for inst_name, prompt in prompts_tpls:
        results[inst_name] = model_call_wrapper(prompt=prompt, 
                                                model_name=model_name,
                                                parse_response_fn=parse_response_fn, 
                                                output_max_length=output_max_length,
                                                num_retries=num_retries,
                                                temperature=temperature)
    return results

def find_sublist(lst1, lst2):
    """
    check if lst2 is a sublist of lst1 (consecutive and same order) 
    if yes - return index in lst1 where lst2 starts, else return -1
    """
    len_lst2 = len(lst2)
    for i in range(len(lst1) - len_lst2 + 1):
        if lst1[i:i+len_lst2] == lst2:
            return i
    return -1  

def get_consecutive_subspans(idx_lst):
    if not idx_lst:
        return []
    idx_subspans = []
    low_lim, up_lim = -1, -1
    for i in range(len(idx_lst)-1):
        if low_lim == -1:
            low_lim = idx_lst[i]
            up_lim = -1
        if idx_lst[i+1] > idx_lst[i]+1:
            up_lim = idx_lst[i]
            idx_subspans.append([low_lim, up_lim])
            low_lim = -1
    if low_lim == -1:
        idx_subspans.append([idx_lst[-1], idx_lst[-1]])
    else:
        idx_subspans.append([low_lim, idx_lst[-1]])
    return idx_subspans

def merge_spans(spans: List[List[List[int]]]) -> List[List[int]]:
    flattened_spans = [span for elem in spans for span in elem]
    idxs = sorted([idx for span in flattened_spans for idx in range(span[0], span[1]+1)])
    return get_consecutive_subspans(idxs)

def add_highlights(doc, highlights, highlight_start_tkn, highlight_end_tkn):
    if not highlights:
        return doc
    highlights = sorted(highlights, key=lambda x: x[0])
    highlighted_doc = doc[:highlights[0][0]] # start with the text until first highlight

    for i,span in enumerate(highlights):
        end_idx_non_highlighted = highlights[i+1][0] if i<len(highlights)-1 else len(doc) # if not final highlight - next non-highlighted span's end idx is the start of the next highlight, otherwise - it is the end of the doc
        addition_txt = highlight_start_tkn + doc[span[0]:span[1]] + highlight_end_tkn + doc[span[1]:end_idx_non_highlighted]
        highlighted_doc += addition_txt
    
    # make sure the removal of the highlights yields the original text
    assert highlighted_doc.replace(highlight_start_tkn, "").replace(highlight_end_tkn, "") == doc

    return highlighted_doc

def extract_highlights(highlighted_doc, highlight_start_tkn, highlight_end_tkn):
    pattern = fr'{highlight_start_tkn}(.*?){highlight_end_tkn}'
    return re.findall(pattern, highlighted_doc, re.DOTALL)

def one_doc_fusion_prompt(doc_cluster, single_doc_prompt):
    # For single doc fusion prompt:
    # - {ID}: doc id (starting from 1)
    # - {HLIST_DOC_FUSION}: the highlights list (only highlights numbers, starting from 1 - comma delimited)
    highlights_text = ','.join([str(h) for h in doc_cluster['highlights']])
    return single_doc_prompt.replace("{ID}", str(doc_cluster['doc'])).replace("{HLIST_DOC_FUSION}", highlights_text)

def make_highlights_fusion_prompt(highlights_fuse_dict, sent_id, curr_prompt):
    # For fusion prompt:
    # - {HFUSE_SENT}: highlights list to be fused
    # - {SENT_ID}: output sentence id (starting from 1)
    # - {SENT}: the sentence text
    sent_text = highlights_fuse_dict['output']
    fusion_dict = sorted(highlights_fuse_dict['highlights_cluster'], key=lambda x: x['doc']) # sort according to the documents
    fusion_dict = [{'doc': elem['doc'],
                    "highlights":sorted(elem['highlights'])} for elem in fusion_dict] # sort the highlights within each doc
    # fusion_text = " + ".join([one_doc_fusion_prompt(elem, single_doc_prompt) for elem in fusion_dict])
    highlights_indices = sorted([highlight_index for doc_cluster in fusion_dict for highlight_index in doc_cluster['highlights']])
    fusion_text = ",".join([str(highlights_index) for highlights_index in highlights_indices])
    return curr_prompt.replace("{HFUSE_SENT}", fusion_text).replace("{SENT_ID}", str(sent_id+1)).replace("{SENT}", sent_text)

def make_clustering_prompt(highlights_fuse_dict, answer_clustering_format):
    # For clustering prompt:
    # - {HCLUSTER}/{COT_HCLUSTER}: highlights list to be clustered (comma-delimited)
    # - {COT_HCLUSTER_TOPIC}: the common topic of the highlights (only for CoT)
    curr_highlights_list = sorted([h_i for elem in highlights_fuse_dict['highlights_cluster'] for h_i in elem['highlights']])
    highlights_list_str = ",".join([str(h_i) for h_i in curr_highlights_list])
    return answer_clustering_format.replace("{HCLUSTER}", highlights_list_str).replace("{COT_HCLUSTER}", highlights_list_str).replace("{COT_HCLUSTER_TOPIC}", highlights_fuse_dict['cluster_CoT_topic'])

def make_content_selection_prompt(highlights_list, doc_id, answer_content_selection_format):
    # For content selection prompt:
    # - {ID}: doc id (starting from 1)
    # - {CONTENT_LIST}: <SPAN_DELIM>-delimited selected content
    text = " <SPAN_DELIM> ".join(highlights_list)
    return answer_content_selection_format.replace("{ID}", str(doc_id+1)).replace("{CONTENT_LIST}", text)

def make_highlights_listing_prompt(highlights_list, doc_id, highlights_cnt, curr_prompt):
    # For highlights listing prompt:
    # - {ID}: doc id (starting from 1)
    # - {HLIST_DOC}: the highlights list (starting from 1 - '\n' delimited)
    text = "\n".join(f"{i+highlights_cnt+1}. {highlight}" for i,highlight in enumerate(highlights_list))
    return curr_prompt.replace("{HLIST_DOC}", text).replace("{ID}", str(doc_id+1))

def make_doc_prompt(doc, doc_id, doc_prompt):
    # For doc prompt:
    # - {ID}: doc id (starting from 1)
    # - {P}: text

    text = doc['text']
    return doc_prompt.replace("{P}", text).replace("{ID}", str(doc_id+1))

def make_ALCE_prompt(sent_plan, prompt):
    # For ALCE-style sentence prompt
    # - {ALCE_SENT}: the curr output sentence
    # - {ALCE_CITATIONS}: the curr output sentence's citations
    cited_docs_ids = sorted([elem['doc'] for elem in sent_plan['highlights_cluster']])
    cited_docs_str = "".join([f"[{str(i)}]" for i in cited_docs_ids])
    return prompt.replace("{ALCE_SENT}", sent_plan['output']).replace("{ALCE_CITATIONS}", cited_docs_str)

def make_demo(item, prompt, doc_prompt=None, instruction=None, answer_related_prompts=None, highlight_start_tkn=None, highlight_end_tkn=None, test=False, content_selection=False):
    # For demo prompt
    # - {INST}: the instruction
    # - {Q}: the questions (relevant for LFQA)
    # - {D}: the documents
    # - {A}: the answers
    prompt = prompt.replace("{INST}", instruction)
    if "{D}" in prompt:
        doc_list = item["docs"]
        text = "".join([make_doc_prompt(doc, doc_id, doc_prompt) for doc_id, doc in enumerate(doc_list)])
        # text = text.replace("\"", "\\\"") # escape quotes to not confuse the json structure
        prompt = prompt.replace("{D}", text)

    # for the content selection - remove the highlights from the input docs
    if content_selection:
        prompt = prompt.replace("{HS}", "").replace("{HE}", "")
    
    if "{A}" in prompt:
        prompt = prompt.replace("{A}", answer_related_prompts["answer_prompt"])
    
    if "{Q}" in prompt:
        prompt = prompt.replace("{Q}", item["question"])

    # ensure that either both {HS} and {HE} in prompt, or neither
    assert ("{HS}" in prompt and "{HE}" in prompt) or (not "{HS}" in prompt and not "{HE}" in prompt)

    if "{HS}" in prompt:
        prompt = prompt.replace("{HS}", highlight_start_tkn).replace("{HE}", highlight_end_tkn)

    highlight_lists = [extract_highlights(doc['text'], "{HS}", "{HE}") for doc in doc_list]

    if "{HLIST}" in prompt:
        highlights_list_text = ""
        highlights_cnt = 0
        for doc_id, doc in enumerate(doc_list):
            highlights_list_text += make_highlights_listing_prompt(highlights_list=highlight_lists[doc_id], 
                                                       doc_id=doc_id, 
                                                       highlights_cnt=highlights_cnt,
                                                       curr_prompt=answer_related_prompts["answer_highlights_listing_prompt"])
            highlights_cnt+=len(highlight_lists[doc_id]) # the listing should continue from the previous doc's enumeration
        prompt = prompt.replace("{HLIST}", highlights_list_text)

    if "{PRFX}" in prompt: #sentence-wise fusion
        prompt = prompt.replace("{PRFX}", item['prefix'])

    if not test:
        if "{HDOCS}" in prompt: # the content selection prompt
            doc_list = item["docs"]
            curr_answer_format = answer_related_prompts['answer_content_selection_format'] 
            text = "\n".join([make_content_selection_prompt(highlights_list=highlight_lists[doc_id], 
                                                           doc_id=doc_id, 
                                                           answer_content_selection_format=curr_answer_format) for doc_id, doc in enumerate(doc_list)])
            prompt = prompt.replace("{HDOCS}", text)
            prompt = prompt.replace("{HS}", highlight_start_tkn).replace("{HE}", highlight_end_tkn).strip()

        if "{PLANNING}" in prompt: # the full FiC-CoT prompt
            prompt = prompt.replace("{PLANNING}", answer_related_prompts["answer_FiC_planning_prompt"])
            fusion_text = "".join([make_highlights_fusion_prompt(highlights_fuse_dict=fusion_dict,
                                                                 sent_id=sent_id, 
                                                                 curr_prompt=answer_related_prompts["answer_highlights_fusion_prompt"]) for sent_id, fusion_dict in enumerate(item['planning'])])
            prompt = prompt.replace("{HFUSE}", fusion_text).replace("{SUMM}", item['answer'])
        
        if "{CoT_RESP}" in prompt: # for FiC (non-CoT) 
            prompt = prompt.replace("{CoT_RESP}", item['answer'])
        
        if "{RESP}" in prompt: # for e2e_only_setting
            prompt = prompt.replace("{RESP}", item['answer'])
        
        if "{ALCE_RESP}" in prompt: # for ALCE-style prompt (generate output with citations)
            alce_response = " ".join([make_ALCE_prompt(sent_plan=elem, 
                                                       prompt=answer_related_prompts['answer_ALCE_format']) for elem in item['planning']])
            prompt = prompt.replace("{ALCE_RESP}", alce_response)

        if "{CoT_CLUSTERING}" in prompt: # the CoT-style clustering prompt
            prompt = prompt.replace("{CoT_CLUSTERING}", answer_related_prompts["answer_clustering_CoT_prompt_intermediate"])
            cot_clustering_text = "".join([make_clustering_prompt(highlights_fuse_dict=fusion_dict, 
                                                                  answer_clustering_format=answer_related_prompts["answer_clustering_CoT_format"]) for sent_id, fusion_dict in enumerate(item['planning'])])
            prompt = prompt.replace("{CHAINS_CLUSTERING}", cot_clustering_text)

        if "{CLUSTERS}" in prompt: # the clustering prompt
            clustering_text = ",".join([make_clustering_prompt(highlights_fuse_dict=fusion_dict, 
                                                               answer_clustering_format=answer_related_prompts["answer_clustering_format"]) for sent_id, fusion_dict in enumerate(item['planning'])])
            prompt = prompt.replace("{CLUSTERS}", f"[{clustering_text}]")
        if "{NEXT_SENT}" in prompt: # the sentence-wise fusion
            prompt = prompt.replace("{NEXT_SENT}", item['answer'])

    else:
        if not answer_related_prompts["answer_prompt"].startswith("Answer:"):
            prompt = prompt.replace(answer_related_prompts["answer_prompt"], "")
        prompt = prompt.replace("{PLANNING}", "").replace("{CLUSTERS}", "").replace("{CoT_CLUSTERING}", "").replace("{HDOCS}", "").replace("{NEXT_SENT}", "").replace("{CoT_RESP}", "").replace("{RESP}", "").replace("{ALCE_RESP}", "").strip()
    return prompt, highlight_lists