from utils import *
from copy import deepcopy
import spacy
import torch
from transformers import AutoTokenizer, AutoModel
from scipy.spatial.distance import cosine

COSINE_SIMILARITY_THR = 0.6
# Load the Sentence-BERT model and tokenizer
sent_transformer_model_name = "sentence-transformers/paraphrase-distilroberta-base-v1"
sent_transformer_tokenizer = AutoTokenizer.from_pretrained(sent_transformer_model_name)
sent_transformer_model = AutoModel.from_pretrained(sent_transformer_model_name)
nlp = spacy.load("en_core_web_sm")

def get_sentence_embedding(sentence, model, tokenizer):
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

def get_indir_paths(args):
    if type(args)!=dict: # i.e. argparse
        args = args.__dict__
    # get the prompt structures and ICL examples and the instances to run
    indir_alignments = args['indir_alignments'] if args['indir_alignments'] else f"data/{args['setting']}/{args['split']}.json" 
    indir_prompt = args['indir_prompt'] if args['indir_prompt'] else f"prompts/{args['setting']}.json" 
    return indir_prompt, indir_alignments

def get_data(args):
    # get the prompt structures and ICL examples and the instances to run
    indir_prompt, indir_alignments = get_indir_paths((args))

    # get setting's prompt_structures and curr alignment instances
    with open(indir_alignments, 'r') as f1:
        alignments_dict = [json.loads(line) for line in f1.readlines()]

    with open(indir_prompt, 'r') as f1:
        prompt_dict = json.loads(f1.read())
    
    return prompt_dict, alignments_dict

def get_subtask_funcs(subtask):
    """returns the relevant functions for each task - parse_response, convert_to_pipeline_style"""
    if subtask == "FiC":
        return parse_FiC_response, convert_FiC_CoT_results_to_pipeline_format
    elif subtask == "content_selection":
        return parse_content_selection_response, convert_content_selection_results_to_pipeline_format
    elif subtask == "clustering":
        return parse_clustering_response, convert_clustering_results_to_pipeline_format
    elif subtask == "e2e_only_setting":
        return parse_e2e_only_setting_response, convert_e2e_only_setting_to_pipeline_format
    elif subtask == "ALCE":
        return parse_ALCE_response, convert_ALCE_to_pipeline_format
    else:
        raise Exception(f"{subtask} is not yet supported")

def get_subtask_prompt_structures(prompt_dict : Dict, setting: str, subtask : str, CoT : bool, always_with_question : bool) -> Dict:
    """returns the subtask relevant prompt structures (instruction and answer-related and demo_prompt)"""
    
    demo_prompt = prompt_dict["demo_prompt_content_selection"] if setting=="LFQA" and (subtask in SUBTASK_WITHOUT_GIVEN_HIGHLIGHTS or always_with_question) else prompt_dict["demo_prompt"]

    with_question_suffix = "-with-question" if always_with_question and setting=="LFQA" else ""
    
    if subtask == "FiC":
        CoT_suffix = "-CoT" if CoT else ""
        answer_related_prompts = {"answer_prompt":prompt_dict[f"answer_FiC{CoT_suffix}_prompt"],
                                  "answer_FiC_planning_prompt":prompt_dict["answer_FiC_planning_prompt"],
                                  "answer_highlights_listing_prompt":prompt_dict["answer_highlights_listing_prompt"],
                                  "answer_highlights_fusion_prompt":prompt_dict["answer_highlights_fusion_prompt"]}
        instruction_prompt = prompt_dict[f"instruction-FiC{CoT_suffix}{with_question_suffix}"]
    elif subtask == "content_selection":
        answer_related_prompts = {"answer_prompt":prompt_dict["answer_content_selection_prompt"],
                                  "answer_content_selection_format":prompt_dict["answer_content_selection_format"]}
        instruction_prompt = prompt_dict["instruction-content-selection"]
    elif subtask == "clustering":
        CoT_suffix = "-CoT" if CoT else ""
        answer_related_prompts = {"answer_prompt":prompt_dict[f"answer_clustering{CoT_suffix}_prompt"],
                                  "answer_highlights_listing_prompt":prompt_dict["answer_highlights_listing_prompt"],
                                  "answer_clustering_CoT_prompt_intermediate":prompt_dict["answer_clustering-CoT_prompt_intermediate"],
                                  "answer_clustering_format":prompt_dict["answer_clustering_format"],
                                  "answer_clustering_CoT_format":prompt_dict["answer_clustering-CoT_format"]} 
        instruction_prompt = prompt_dict[f"instruction-clustering{with_question_suffix}"]
    elif subtask == "e2e_only_setting":
        answer_related_prompts = {"answer_prompt":prompt_dict["answer_e2e_only_setting_prompt"]}
        instruction_prompt = prompt_dict["instruction-e2e-only-setting"]
    elif subtask == "ALCE":
        answer_related_prompts = {"answer_prompt":prompt_dict["answer_ALCE_prompt"],
                                  "answer_ALCE_format":prompt_dict["answer_ALCE_format"]}
        instruction_prompt = prompt_dict["instruction-ALCE"]        
    else:
        raise Exception(f"{subtask} is not yet supported") 
    
    return {"answer_related_prompts" : answer_related_prompts, 
            "instruction_prompt" : instruction_prompt,
            "demo_prompt" : demo_prompt}                  

def construct_non_demo_part(instance, merge_cross_sents_highlights, specific_prompt_details, prompt_dict, no_highlights, cut_surplus: bool = False, prct_surplus: float = 0.25):
        """prct_surplus: the percentage of last sentences to consider as surplus and remove in subtasks without given highlights (e.g., content_selection or end-to-end or ALCE)"""
        highlight_start_tkn = "{HS}"
        highlight_end_tkn="{HE}"
        topic_name = instance['unique_id']
        highlighted_texts = get_highlighted_doc(docs={elem['documentFile']:elem['rawDocumentText'] for elem in instance['documents'] if elem['documentFile']}, 
                                                highlights=instance['set_of_highlights_in_context'],
                                                highlight_start_tkn = highlight_start_tkn,
                                                highlight_end_tkn=highlight_end_tkn,
                                                merge_cross_sents_highlights=merge_cross_sents_highlights,
                                                doc_sents={elem['documentFile']:elem['documentText'] for elem in instance['documents'] if elem['documentFile']})
        
        if cut_surplus:
            if no_highlights: # tasks without given highlights - remove prct_surplus% of the top sentences (unless it's less than 5 sentences, and then keep the first 5 sentences - to avoid cases when short documents completely disappear)
                highlighted_texts = {elem['documentFile']:"".join(elem['documentText'][:max(int(len(elem['documentText'])*(1-prct_surplus)), 5)]) for elem in instance['documents'] if elem['documentFile']}
            else: # tasks with given highligts (e.g., clustering, FiC) - remove everything after last highlight
                highlighted_texts = {doc_name:rmv_txt_after_last_highlight(doc_text, highlight_end_tkn) for doc_name,doc_text in highlighted_texts.items()}
                highlighted_texts = {key:value for key,value in highlighted_texts.items() if value} # remove empty texts

        docs_order = [{"doc_name":doc_name, "doc_text":doc_text} for doc_name,doc_text in highlighted_texts.items()]
        eval_item = {"docs":[{'text':dct["doc_text"]} for dct in docs_order]}

        if "query" in instance.keys(): # for LFQA
            eval_item["question"] = instance["query"]

        curr_prompt, curr_highlight_list = make_demo(
            item=eval_item, prompt=specific_prompt_details["demo_prompt"], doc_prompt=prompt_dict["doc_prompt"], 
            instruction=specific_prompt_details["instruction_prompt"], answer_related_prompts=specific_prompt_details["answer_related_prompts"],
            highlight_start_tkn=prompt_dict["highlight_start_tkn"], highlight_end_tkn=prompt_dict["highlight_end_tkn"],
            content_selection=no_highlights,
            test=True
        )

        return curr_prompt, curr_highlight_list, topic_name, docs_order

def construct_prompts(prompt_dict : Dict, alignments_dict : List[Dict], n_demos : int, debugging : bool, merge_cross_sents_highlights : bool, specific_prompt_details : Dict, tkn_counter: Dict, no_highlights : bool = False, cut_surplus : bool = False, prct_surplus: float = None):
    # Generate the demonstration part
    head_prompt, head_prompt_shorter = "", "" # the shorter is for cases when the instance with the demos is too long
    train_ids = np.random.choice(len(prompt_dict["demos"]), n_demos, replace=False) if not debugging else [0,2]
    used_demos = []
    for train_id in train_ids:
        train_item = prompt_dict["demos"][train_id]
        used_demos.append(train_item)

        curr_prompt_demo, _ = make_demo(
            item=train_item, prompt=specific_prompt_details["demo_prompt"], doc_prompt=prompt_dict["doc_prompt"], 
            instruction=specific_prompt_details["instruction_prompt"], answer_related_prompts=specific_prompt_details["answer_related_prompts"],
            highlight_start_tkn=prompt_dict["highlight_start_tkn"], highlight_end_tkn=prompt_dict["highlight_end_tkn"],
            content_selection=no_highlights
        )
        head_prompt += curr_prompt_demo
        head_prompt += prompt_dict["demo_sep"]

        # get shorter version
        train_item_shorter = {key:[{doc_key:elem['shorter_text'] if doc_key=="text" else doc_value 
                                    for doc_key,doc_value in elem.items()} for elem in value] if key=="docs" else value for key,value in train_item.items()}
        
        curr_prompt_demo_shorter, _ = make_demo(
            item=train_item_shorter, prompt=specific_prompt_details["demo_prompt"], doc_prompt=prompt_dict["doc_prompt"], 
            instruction=specific_prompt_details["instruction_prompt"], answer_related_prompts=specific_prompt_details["answer_related_prompts"],
            highlight_start_tkn=prompt_dict["highlight_start_tkn"], highlight_end_tkn=prompt_dict["highlight_end_tkn"],
            content_selection=no_highlights
        )
        head_prompt_shorter += curr_prompt_demo_shorter
        head_prompt_shorter += prompt_dict["demo_sep"]

    if debugging:
        alignments_dict = alignments_dict[:3]

    final_prompts, additional_data = {}, {}
    for instance in alignments_dict:
        curr_prompt, curr_highlight_list, topic_name, docs_order = construct_non_demo_part(instance, merge_cross_sents_highlights, specific_prompt_details, prompt_dict, no_highlights)        
        
        if cut_surplus: # cut text
            curr_prompt, curr_highlight_list_shorter, topic_name, docs_order_shorter = construct_non_demo_part(instance, merge_cross_sents_highlights, specific_prompt_details, prompt_dict, no_highlights, cut_surplus=True, prct_surplus=prct_surplus)
            final_prompts[topic_name] = head_prompt_shorter + curr_prompt
        elif tkn_counter["tkn_counter"].token_count(head_prompt + curr_prompt)>=tkn_counter["tkn_max_limit"]: # the current prompt is too long - need to "cut" the surplus texts
            # check several such cut-off thr (until the first/smallest one catches) - rage from 0.05 and 0.5
            prct_surplus_lst = np.linspace(0.05, 0.5, int((0.5 - 0.05) / 0.05) + 1)
            for curr_prct_surplus in prct_surplus_lst:
                curr_prompt, curr_highlight_list_shorter, topic_name, docs_order_shorter = construct_non_demo_part(instance, merge_cross_sents_highlights, specific_prompt_details, prompt_dict, no_highlights, cut_surplus=True, prct_surplus=curr_prct_surplus)
                if tkn_counter["tkn_counter"].token_count(head_prompt_shorter + curr_prompt)<tkn_counter["tkn_max_limit"]:
                    break
            final_prompts[topic_name] = head_prompt_shorter + curr_prompt
        else:
            curr_highlight_list_shorter, docs_order_shorter = [], []
            final_prompts[topic_name] = head_prompt + curr_prompt
        highlighted_docs = [{"doc_name":elem["doc_name"],
                             "doc_text":elem["doc_text"].replace("{HS}", prompt_dict["highlight_start_tkn"]).replace("{HE}", prompt_dict["highlight_end_tkn"])} for elem in docs_order]
        non_highlighted_docs = [{"doc_name":elem["doc_name"],
                                 "doc_text":elem["doc_text"].replace("{HS}", "").replace("{HE}", "")} for elem in docs_order]
        highlighted_docs_shorter = [{"doc_name":elem["doc_name"],
                                     "doc_text":elem["doc_text"].replace("{HS}", prompt_dict["highlight_start_tkn"]).replace("{HE}", prompt_dict["highlight_end_tkn"])} for elem in docs_order_shorter]
        non_highlighted_docs_shorter = [{"doc_name":elem["doc_name"],
                                         "doc_text":elem["doc_text"].replace("{HS}", "").replace("{HE}", "")} for elem in docs_order_shorter]
        

        no_highlights_prfx = "gold_" if no_highlights else ""

        additional_data[topic_name] = {"non_highlighted_docs":non_highlighted_docs,
                                       f"{no_highlights_prfx}highlighted_docs":highlighted_docs,
                                       f"{no_highlights_prfx}highlights":curr_highlight_list,
                                       f"non_highlighted_docs_shorter":non_highlighted_docs_shorter,
                                       f"{no_highlights_prfx}highlighted_docs_shorter":highlighted_docs_shorter,
                                       f"{no_highlights_prfx}highlights_shorter":curr_highlight_list_shorter}
        
        if "query" in instance.keys(): # for LFQA
            additional_data[topic_name].update({"question":instance["query"]})
    return used_demos, final_prompts, additional_data

def adapt_highlights_to_doc_alignments(doc_texts_dict, salience_dict):
    salience_dict_adapted = deepcopy(salience_dict)
    for doc_name, salience_list in salience_dict.items():
        for salient_span in salience_list:
            if rmv_spaces_and_punct(salient_span) in rmv_spaces_and_punct(doc_texts_dict[doc_name]):
                continue
            alternative_docs = [key for key,value in doc_texts_dict.items() if rmv_spaces_and_punct(salient_span) in rmv_spaces_and_punct(value)]
            assert len(alternative_docs)>0, "not all selected content is traceable" # make sure the highlight appears in any of the docs

            alternative_doc = alternative_docs[0]
            if not alternative_doc in salience_dict_adapted.keys(): # if curr document wasn't assigned highlights in the model's response (even though the current highlight actually belongs to it)
                salience_dict_adapted[alternative_doc] = []
            
            # add to correct list
            salience_dict_adapted[alternative_doc].append(salient_span)

            # remove from incorrect list
            salience_dict_adapted[doc_name] = [h for h in salience_dict_adapted[doc_name] if h != salient_span]

    return salience_dict_adapted

def parse_FiC_response(response, prompt):
    if "where each time you cluster highlights to build the next sentence." in prompt: # the CoT variant
        assert "So the final summary is:" in response or "So the final answer is:" in response, "cannot find the final summary in the model's response"
        if "So the final summary is:" in response: #MDS
            final_summary = response[response.index("So the final summary is:"):].replace("So the final summary is:", "").strip()
            planning_response = response[:response.index("So the final summary is:")]
        else: #LFQA
            final_summary = response[response.index("So the final answer is:"):].replace("So the final answer is:", "").strip()
            planning_response = response[:response.index("So the final answer is:")]
        pattern = r"highlight(?:s)? ([\d, and]+) (?:is|are) combined to form sentence (\d+):[\n ]+(.*?)(?=[\n]+|\Z)"
        alignment_matches = re.findall(pattern, planning_response, re.IGNORECASE)
        alignments = [{"sent_id" : int(elem[1]),
                    "highlights" : sorted([int(highlight_i.replace("and", "")) for highlight_i in elem[0].split(",")]),
                    "sent_text" : elem[2]} for elem in alignment_matches]
        alignments = sorted(alignments, key=lambda x: x['sent_id'])
        if not "".join([elem['sent_text'] for elem in alignments]).replace(" ", "").replace("\n", "")==final_summary.replace(" ", "").replace("\n", ""):
            logging.info(f"separate sentences don't match final summary. Separate sentences combined:\n {' '.join([elem['sent_text'] for elem in alignments])}.\n\n Final summary:\n{final_summary}")
        return {"alignments":alignments,
                "final_output":final_summary,
                "full_model_response":response}
    else: # the non-CoT variant
        # if added the "So the summary is:" - remove it
        if response.startswith("So the summary is:"): 
            response = response[len("So the summary is:"):]

        return {"final_output":response,
                "full_model_response":response}

def parse_content_selection_response(response, prompt):
    # split into "Document [<INT>]:" and the ensuing text
    doc_spans_pairs = [(match.group(1), match.group(2).strip()) for match in re.finditer(r"(Document \[\d+\]):(.*?)(?=Document \[|\Z)", response, re.DOTALL)]
    if not doc_spans_pairs:
        raise Exception("no content was found")
    salience_dict = {tpl[0]:tpl[1].split("<SPAN_DELIM>") for tpl in doc_spans_pairs}

    # remove salient spans consisting of only spaces and punctuations
    salience_dict = {key:[s for s in value if rmv_spaces_and_punct(s)] for key,value in salience_dict.items()}
    # check that only relevant documents were included
    curr_instance_part = prompt.split("IMPORTANT: Each salient content must be a single consecutive verbatim span from the corresponding passages.")[-1] # remove the ICL demonstrations

    docs_names = re.findall(r'(Document \[\d+\]):', curr_instance_part) # find all 'Document [<INT]:' 
    assert all(key in docs_names for key in salience_dict.keys()), "not only relevant documents are included"

    # check that all selected content is actually traceable in the relevant docs (when all in lowercap letters and while ignoring spaces and new lines and punctuation)
    docs_texts = re.findall(r"(Document \[\d+\]):\s+([^\n]*(?:\n(?!Document \[\d+\]:)[^\n]*)*)", curr_instance_part)
    docs_texts = {name:text for name,text in docs_texts}

    salience_dict = adapt_highlights_to_doc_alignments(docs_texts, salience_dict) # in cases the highlight wasn't properly matched with the correct document, but does appear in another document

    assert all(all(rmv_spaces_and_punct(salient_span) in rmv_spaces_and_punct(docs_texts[doc_name]) for salient_span in salient_spans) for doc_name,salient_spans in salience_dict.items()), "not all selected content is traceable"
    # [[rmv_spaces_and_punct(salient_span) in rmv_spaces_and_punct(docs_texts[doc_name]) for salient_span in salient_spans] for doc_name,salient_spans in salience_dict.items()] # for debugging
    return {"final_output":salience_dict,
            "full_model_response":response}

def parse_clustering_response(response, prompt):
    original_response = deepcopy(response)
    if "So, the highlighted spans are clustered as follows:" in response: # for the CoT cases (leave out only the final output)
        response = response.split("So, the highlighted spans are clustered as follows:")[-1].strip()
    clusters = json.loads(response)
    assert all(type(elem)==dict for elem in clusters), "not all elems in jsoned response are dictionaries" # make sure all are dictionaries
    assert all(len(elem)==1 for elem in clusters), "not all dicts in jsoned response are of length 1" # make sure all dicts have only one key,value pair
    assert all("cluster" in elem.keys() for elem in clusters), "not all keys are named \"cluster\"" # make sure the key is "cluster"
    assert all(type(list(elem.values())[0])==list for elem in clusters), "not all values are lists" # make sure all values are lists
    assert all(all(type(elem)==int for elem in list(cluster_elem.values())[0]) for cluster_elem in clusters), "not all elems of lists are integers" # make sure all elems in the lists are integers
    
    # check that only highlights were clustered
    curr_instance_part = prompt.split("The highlighted spans are:")[-1] # remove the ICL demonstrations
    highlights_indices = re.findall(r'\n\d+\. ', curr_instance_part) # find all '\n<INT>. ' 
    prompt_indices = [int(elem.replace("\n", "").replace(".", "")) for elem in highlights_indices] # get only the <INT>
    response_indices = [i for elem in clusters for i in elem['cluster']]
    assert all(i in prompt_indices for i in response_indices), "not all indices in the lists are actual highlights" # make sure all indices in the response are actual highlights


    return {"final_output":clusters,
            "full_model_response":original_response}

def parse_e2e_only_setting_response(response, *args, **kwargs):
    original_response = deepcopy(response)
    if response.strip().lower().startswith("answer:"):
        response = response[len("answer:"):].strip()
    
    return {"final_output":response,
            "full_model_response":original_response}
    
def parse_ALCE_response(response, prompt):
    # if response start with "Answer:" - remove it"
    original_response = deepcopy(response)
    if response.lower().startswith("answer:"):
        response = response[len("answer:"):].strip()
    
    # find citations and their start and end char index
    all_citations = []
    for match in re.finditer(r"\[\d+\]", response):
        start = match.start()
        end = match.end()
        matched_text = match.group()
        all_citations.append((matched_text, start, end))
    assert all(tpl[0]==response[tpl[1]:tpl[2]] for tpl in all_citations), "misalignment in the citation extraction"

    curr_instance_part = prompt.split("If multiple passages support the sentence, only cite a minimum sufficient subset of the passages.")[-1] # remove the ICL demonstrations
    docs_names = re.findall(r'Document (\[\d+\]):', curr_instance_part) # find all 'Document [<INT]:' (and leave only the [<INT>])
    assert all(key in docs_names for key in [tpl[0] for tpl in all_citations]), "not only relevant documents are included"    

    response_no_citations = re.sub(r"\[\d+\]", "", response)
    # separate into sentences but leave only 
    response_sents = [sent.text.strip() for sent in nlp(response_no_citations).sents]

    if any(len(find_substring_indices(response, sent))>1 and rmv_spaces_and_punct(sent) for sent in response_sents):
        logging.warning(f"at least one sentence was found in more than one mention was found in the generated text so ensure citations are properly paired: {sent}")

    # find idxs of sents in original response (with citations)
    sent_to_citation_mapping = []
    prev_sent_end_idx = -1
    for sent in response_sents:
        if not rmv_spaces_and_punct(sent): # sentence comprising solely of punctuation and spacings (their spans should be irrelevant)
            sent_to_citation_mapping.append({"sent":sent,
                                             "spans":None,
                                             "cited_docs":[]}) 
            continue
        sent_start_idx = find_substring_indices(response, sent)
        if len(sent_start_idx)>1: # for cases when a sentence might appear in more than one place - the take the closest idx to the previous sentence
            sent_start_idx = min([elem for elem in sent_start_idx if elem>prev_sent_end_idx])
        else:
            sent_start_idx = sent_start_idx[0]
        sent_end_idx = sent_start_idx + len(sent)
        sent_to_citation_mapping.append({"sent":sent,
                                         "spans":(sent_start_idx, sent_end_idx),
                                         "cited_docs":[]})
        
        prev_sent_end_idx = sent_end_idx
    
    # for each citation - find closest sent whose end_idx is lower than the citation's start_idx
    for citation in all_citations:
        curr_doc_num = int(citation[0].replace("[", "").replace("]", ""))
        citation_start_idx = citation[1]
        citation_end_idx = citation[2]
        # find the all sentences (that don't consist only of spaces and punctuation) that end before the current citation
        relevant_sents_i = [i for i,elem in enumerate(sent_to_citation_mapping) if elem['spans'] and elem['spans'][1]<citation_start_idx]
        if not relevant_sents_i: # citation comes before any sentence - ignore
            logging.info(f"in the following model's response the citation {citation[0]} came before any sentence - so it is ignored:\n{response}")
            continue
        # take the "latest" sentence
        cited_sent_i = max(relevant_sents_i)
        sent_to_citation_mapping[cited_sent_i]['cited_docs'].append(curr_doc_num)
    
    final_response = [{'sent':elem['sent'],
                       'cited_docs':sorted(list(set(elem['cited_docs'])))} for elem in sent_to_citation_mapping]


    return {"final_output":final_response,
            "response_with_citations" : response,
            "response_without_citations" : response_no_citations,
            "full_model_response" : original_response}

def get_set_of_highlights_in_context_content_selection(doc_name, doc_text, highlights, nlp, doc_sents, *args, **kwargs):
    if not doc_sents: # when in the gold data a document didn't have highlights, the corresponding doc_sents are None
        doc_sents = [sent.text for sent in nlp(doc_text).sents]
    sents_idx_limits = [find_substring(doc_text, sent) for sent in doc_sents]

    highlights_in_context_list = []
    for h in highlights:
        if not rmv_spaces_and_punct(h): # skip highlights consisting only of spaces and punctuations
            continue
        h = h.strip()
        h_idx_limits = find_substring(doc_text, h)

        assert h_idx_limits[0]!=-1, "didn't find highlight"
        
        # find indices of sentences whose idx spans overlap with those of h (and also ignore sents that only consist of space, new lines and punctuations)
        relevant_sents_i = sorted([i for i,lims in enumerate(sents_idx_limits) if set(range(lims[0],lims[1])).intersection(set(range(h_idx_limits[0], h_idx_limits[1]))) and rmv_spaces_and_punct(doc_sents[i])])
        
        # remove found relevant sentences that are a substring of another found sentence
        relevant_sents_i = sorted([i for i in relevant_sents_i if not any(rmv_spaces_and_punct(doc_sents[i]) in rmv_spaces_and_punct(doc_sents[j]) and rmv_spaces_and_punct(doc_sents[i])!=rmv_spaces_and_punct(doc_sents[j]) for j in relevant_sents_i)])
        
        if len(relevant_sents_i)>1:
            # separate span into its separate sentences
            for sentence_index in relevant_sents_i:
                curr_sents_idx_limits = sents_idx_limits[sentence_index]
                curr_h_idx_limits = get_consecutive_subspans(sorted(list(set(range(h_idx_limits[0],h_idx_limits[1]+1)).intersection(range(curr_sents_idx_limits[0], curr_sents_idx_limits[1]+1)))))
                docSentCharIdx = str(curr_sents_idx_limits[0])
                docSentText = doc_text[curr_sents_idx_limits[0]:curr_sents_idx_limits[1]]
                docSpanOffsets = [list(subspan) for subspan in curr_h_idx_limits] #";".join([", ".join([str(elem[0]), str(elem[1])]) for elem in curr_h_idx_limits])
                docSpanText = SPAN_SEP.join([doc_text[subspan[0]:subspan[1]] for subspan in curr_h_idx_limits])
                
                highlights_in_context_list.append({"documentFile" : doc_name,
                                                "scuSentCharIdx" : None,
                                                "scuSentence" : None,
                                                "docSentCharIdx" : docSentCharIdx,
                                                "docSentText" : docSentText,
                                                "docSpanText" : docSpanText,
                                                "docSpanOffsets" : docSpanOffsets,
                                                "sent_idx" : sentence_index})
        
        elif len(relevant_sents_i)==0: # sometimes there are duplicate sentences in the document and the sents_idx_limits missed the spans of where h_idx_limits was identified
            potentially_containing_sents_i = [i for i,sent in enumerate(doc_sents) if rmv_spaces_and_punct(h) in rmv_spaces_and_punct(sent)]
            assert potentially_containing_sents_i, "highlight wasn't found"
            # find the LCS of the chars between h and one of the potential sentences (here - first)
            relevant_sents_i = potentially_containing_sents_i[0]
            docSentCharIdx = str(sents_idx_limits[relevant_sents_i][0])
            docSentText = doc_text[sents_idx_limits[relevant_sents_i][0]:sents_idx_limits[relevant_sents_i][1]]
            
            # find LCS between sentence and highlight - and adapt highlights indices relative to the sent's docSentCharIdx
            lcs_details = longest_common_subsequence(doc_sents[relevant_sents_i], h)
            in_sent_spans = get_consecutive_subspans(sorted(lcs_details[1]))
            docSpanOffsets = [[int(docSentCharIdx)+elem for elem in span] for span in in_sent_spans]
            docSpanText = SPAN_SEP.join([doc_text[subspan[0]:subspan[1]] for subspan in docSpanOffsets])
            
            highlights_in_context_list.append({"documentFile" : doc_name,
                                            "scuSentCharIdx" : None,
                                            "scuSentence" : None,
                                            "docSentCharIdx" : docSentCharIdx,
                                            "docSentText" : docSentText,
                                            "docSpanText" : docSpanText,
                                            "docSpanOffsets" : docSpanOffsets,
                                            "sent_idx" : relevant_sents_i})
        else:
            relevant_sents_i = relevant_sents_i[0]
            docSentCharIdx = [str(sent[0]) for sent_i,sent in enumerate(sents_idx_limits) if sent_i==relevant_sents_i][0]
            docSentText = [doc_text[sent_span[0]:sent_span[1]] for sent_i,sent_span in enumerate(sents_idx_limits) if sent_i==relevant_sents_i][0]
            docSpanOffsets = [list(h_idx_limits)] #", ".join([str(elem) for elem in h_idx_limits])
            docSpanText = doc_text[h_idx_limits[0]:h_idx_limits[1]]
            highlights_in_context_list.append({"documentFile" : doc_name,
                                            "scuSentCharIdx" : None,
                                            "scuSentence" : None,
                                            "docSentCharIdx" : docSentCharIdx,
                                            "docSentText" : docSentText,
                                            "docSpanText" : docSpanText,
                                            "docSpanOffsets" : docSpanOffsets,
                                            "sent_idx" : relevant_sents_i})
        


    # remove duplicates
    json_highlights_in_context_list = set([json.dumps(elem) for elem in highlights_in_context_list])
    highlights_in_context_list = [json.loads(elem) for elem in json_highlights_in_context_list]
    return highlights_in_context_list

def convert_content_selection_results_to_pipeline_format(results, alignments_dict, *args, **kwargs):
    nlp = spacy.load("en_core_web_sm")
    pipeline_style_data = []
    for key,value in results.items():
        curr_original_inst = deepcopy([elem for elem in alignments_dict if elem["unique_id"]==key][0])
        
        curr_documents = curr_original_inst["documents"]
        # extract all the highlights of the doc if the doc appears in the extracted content
        highlights_in_context = [get_set_of_highlights_in_context_content_selection(doc_name=doc['doc_name'], 
                                                                  doc_text=[elem['rawDocumentText'] for elem in curr_documents if elem['documentFile']==doc['doc_name']][0],
                                                                  highlights=value["final_output"][f"Document [{str(i+1)}]"], 
                                                                  nlp=nlp, 
                                                                  doc_sents=next(iter([elem['documentText'] for elem in curr_documents if elem['documentFile']==doc['doc_name']]), None) if "documentText" in curr_documents[0].keys() else None) for i,doc in enumerate(value['non_highlighted_docs']) if f"Document [{str(i+1)}]" in value["final_output"].keys()]
        highlights_in_context = [elem for doc_elems in highlights_in_context for elem in doc_elems]
        
        curr_original_inst["set_of_highlights_in_context"] = highlights_in_context # update the set of highlights in context
        
        
        pipeline_style_data.append(curr_original_inst)
    return pipeline_style_data

def get_set_of_highlights_in_context_clustering(curr_instance, nlp, doc_sents, *args, **kwargs):
    highlights_in_context_list = []
    highlight_global_index = 1 # highlights enuemration starts from 1 in the prompts
    for doc_i,doc_highlights in enumerate(curr_instance['highlights']):
        curr_doc_name = curr_instance['highlighted_docs'][doc_i]['doc_name']
        curr_doc_text = next(iter([elem['rawDocumentText'] for elem in doc_sents if elem['documentFile']==curr_doc_name]), None)
        curr_doc_sents = next(iter([elem['documentText'] for elem in doc_sents if elem['documentFile']==curr_doc_name]), None)

        for highlight in doc_highlights:
            curr_highlights_in_context = get_set_of_highlights_in_context_content_selection(doc_name=curr_doc_name, 
                                                                                            doc_text=curr_doc_text, 
                                                                                            highlights=[highlight], 
                                                                                            nlp=nlp, 
                                                                                            doc_sents=curr_doc_sents)
            
            relevant_clusters = [cluster_i for cluster_i,elem in enumerate(curr_instance['final_output']) if highlight_global_index in elem['cluster']]

            for cluster_index in relevant_clusters:
                # change the scuSentCharIdx to cluster_index and add to highlights_in_context_list
                highlights_in_context_list+=[{k: cluster_index if k == 'scuSentCharIdx' else v for k, v in d.items()} for d in curr_highlights_in_context]

            # increment the highlight_global_index
            highlight_global_index+=1

    return highlights_in_context_list

def get_set_of_highlights_in_context_ALCE(curr_instance):
    highlights_in_context_list = []
    final_output = ""
    curr_scuSentCharIdx = 0
    for sentwise_results in curr_instance["final_output"]:
        if sentwise_results['cited_docs']: # found evidence
            # add each of the cited docs
            for doc_i in sentwise_results['cited_docs']:
                highlights_in_context_list.append({"documentFile" : curr_instance['non_highlighted_docs'][doc_i-1]['doc_name'],
                                                   "scuSentCharIdx" : curr_scuSentCharIdx,
                                                   "scuSentence" : sentwise_results['sent'],
                                                   "docSentCharIdx" : None,
                                                   "docSentText" : None,
                                                   "docSpanText" : None,
                                                   "docSpanOffsets" : None,
                                                   "sent_idx" : None})
        else: # no evidence found
                highlights_in_context_list.append({"documentFile" : None,
                                                   "scuSentCharIdx" : curr_scuSentCharIdx,
                                                   "scuSentence" : sentwise_results['sent'],
                                                   "docSentCharIdx" : None,
                                                   "docSentText" : None,
                                                   "docSpanText" : None,
                                                   "docSpanOffsets" : None,
                                                   "sent_idx" : None})
                
        curr_scuSentCharIdx = curr_scuSentCharIdx + len(sentwise_results['sent']) + 1 # add the lengths of the curr sent plus one space
        final_output = final_output + sentwise_results['sent'] + " "
    
    assert all(final_output[elem['scuSentCharIdx']:].startswith(elem['scuSentence']) for elem in highlights_in_context_list), "scuSentence doesn't match scuSentCharIdx"
    return highlights_in_context_list, final_output.strip()
    
def convert_clustering_results_to_pipeline_format(results, alignments_dict, *args, **kwargs):
    nlp = spacy.load("en_core_web_sm")
    pipeline_style_data = []
    for key,value in results.items():
        curr_pipeline_style_data = [elem for elem in alignments_dict if elem["unique_id"]==key][0]

        curr_documents = curr_pipeline_style_data["documents"]
        highlights_in_context = get_set_of_highlights_in_context_clustering(curr_instance=value,
                                                                            nlp=nlp, 
                                                                            doc_sents=curr_documents)
        # update the response and set_of_highlights_in_context
        curr_pipeline_style_data.update({"set_of_highlights_in_context":highlights_in_context,
                                       "response" : value["gold_summary"]})
        pipeline_style_data.append(curr_pipeline_style_data)
    return pipeline_style_data

def convert_e2e_only_setting_to_pipeline_format(results, alignments_dict, *args, **kwargs):
        pipeline_style_data = []
        for key,value in results.items():
            original_alignments_dict = deepcopy([elem for elem in alignments_dict if elem['unique_id']==key][0])
            original_alignments_dict.update({"set_of_highlights_in_context":[],
                                             "response" : value["final_output"],
                                             "gold_summary" : value["gold_summary"]})
            pipeline_style_data.append(original_alignments_dict)
        return pipeline_style_data

def convert_ALCE_to_pipeline_format(results, alignments_dict, *args, **kwargs):
        pipeline_style_data = []
        for key,value in results.items():
            curr_documents = [elem["documents"] for elem in alignments_dict if elem["unique_id"]==key][0]
            highlights_in_context, final_output = get_set_of_highlights_in_context_ALCE(curr_instance=value)
            original_alignments_dict = deepcopy([elem for elem in alignments_dict if elem['unique_id']==key][0])
            original_alignments_dict.update({"set_of_highlights_in_context" : highlights_in_context,
                                             "response" : final_output,
                                             "gold_summary" : value["gold_summary"]})
            pipeline_style_data.append(original_alignments_dict)
        return pipeline_style_data

def get_set_of_highlights_in_context_FiC_CoT(curr_instance, nlp, doc_sents, *args, **kwargs):
    # find the clustered highlights, and then pair then with the corresponding sentences (According to their assigned scuSentCharIdx)
    clustering_style_instance_format = {"highlights":curr_instance["highlights"],
                                        "highlighted_docs":curr_instance["highlighted_docs"],
                                        "final_output":[{"cluster":elem['highlights']} for elem in curr_instance['alignments']]}
    clustered_set_of_highlights = get_set_of_highlights_in_context_clustering(curr_instance=clustering_style_instance_format, 
                                                                              nlp=nlp, 
                                                                              doc_sents=doc_sents)
    highlights_in_context_list = []
    alignments_sents_embedding = None
    for sent in nlp(curr_instance['final_output']).sents:
        # first find identical sentences in the "CoT planning"
        relevant_sent_id = [elem['sent_id'] for elem in curr_instance['alignments'] if rmv_spaces_and_punct(sent.text) and rmv_spaces_and_punct(sent.text)==rmv_spaces_and_punct(elem['sent_text'])]
        if not relevant_sent_id: # if not found - find sentences that one contains the other
            relevant_sent_id = [elem['sent_id'] for elem in curr_instance['alignments'] if (rmv_spaces_and_punct(sent.text) and rmv_spaces_and_punct(sent.text) in rmv_spaces_and_punct(elem['sent_text'])) or (rmv_spaces_and_punct(elem['sent_text']) and rmv_spaces_and_punct(elem['sent_text']) in rmv_spaces_and_punct(sent.text))]

        if not relevant_sent_id: # no evidence found - as a last resort find sentences from the planning whose embedding is close to curr sentence (ocassionally - the model paraphrases in the final output a little too much + also the spacy tokenizer didn't properly separate the sentences)
            curr_sent_embedding = get_sentence_embedding(sent.text, sent_transformer_model, sent_transformer_tokenizer)
            if not alignments_sents_embedding: # if curr_alignments_sents haven't been converted to embeddings (to avoid converting to embedding multiple times)
                alignments_sents_embedding = [get_sentence_embedding(elem["sent_text"], sent_transformer_model, sent_transformer_tokenizer) for elem in curr_instance['alignments']]
            similarities = [1 - cosine(curr_sent_embedding , s_embedding) for s_embedding in alignments_sents_embedding]
            relevant_sent_id = [curr_instance['alignments'][i]['sent_id'] for i,scr in enumerate(similarities) if scr>=COSINE_SIMILARITY_THR]
        
        if not relevant_sent_id: # if still not found - then current sentence will be without citation
            highlights_in_context_list.append({"documentFile" : None,
                                               "scuSentCharIdx" : sent.start_char,
                                               "scuSentence" : sent.text,
                                               "docSentCharIdx" : None,
                                               "docSentText" : None,
                                               "docSpanText" : None,
                                               "docSpanOffsets" : None,
                                               "sent_idx" : None})
        else:
            # take elem['scuSentCharIdx']+1 because in get_set_of_highlights_in_context_clustering the sentence counting starts from 0, while in the FiC-CoT prompt, it starts from 1
            # "in" and not "==" because sometimes more than one sentence fits
            relevant_clustered_highlights_in_context = deepcopy([elem for elem in clustered_set_of_highlights if elem['scuSentCharIdx']+1 in relevant_sent_id])
            # adapt scuSentCharIdx and scuSentence
            relevant_clustered_highlights_in_context = [{key:value if key!="scuSentence" else sent.text for key,value in elem.items()} for elem in relevant_clustered_highlights_in_context]
            relevant_clustered_highlights_in_context = [{key:value if key!="scuSentCharIdx" else sent.start_char for key,value in elem.items()} for elem in relevant_clustered_highlights_in_context]
            # [elem.update({"scuSentence":sent.text, "scuSentCharIdx":sent.start_char}) for elem in relevant_clustered_highlights_in_context]
            highlights_in_context_list += relevant_clustered_highlights_in_context
    return highlights_in_context_list

def convert_FiC_CoT_results_to_pipeline_format(results, alignments_dict, *args, **kwargs):
    nlp = spacy.load("en_core_web_sm")
    pipeline_style_data = []
    for key,value in results.items():
        curr_pipeline_style_data = [elem for elem in alignments_dict if elem["unique_id"]==key][0]

        curr_documents = curr_pipeline_style_data["documents"]
        highlights_in_context = get_set_of_highlights_in_context_FiC_CoT(curr_instance=value,
                                                                         nlp=nlp, 
                                                                         doc_sents=curr_documents)
        # update the response and set_of_highlights_in_context
        curr_pipeline_style_data.update({"set_of_highlights_in_context":highlights_in_context,
                                       "response" : value["final_output"]})
        pipeline_style_data.append(curr_pipeline_style_data)
    return pipeline_style_data