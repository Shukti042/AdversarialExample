import json
from prompts import get_classification_prompt, get_adv_prompt, get_iterative_feedback, get_similarity_score_prompt, get_reasoning_prompt, get_filtering_prompt
similarity_threshold=6
class Interact:
    def __init__(self, dataset, pipe, label_list):
        self.label_list = label_list
        self.dataset = dataset
        self.pipe = pipe

    def get_single_pred(self, sentence):
        classification_prompt = get_classification_prompt(self.dataset)
        messages = [
                        {"role": "system", "content": classification_prompt},
                        {"role": "user", "content": f"{sentence}"},
                    ]
       
        answer =  self.pipe(messages, max_new_tokens=256)[0]['generated_text'][-1]['content']
                # print(answer)
        counts = [answer.lower().count(label) - answer.lower().count(f"_{label}") for label in self.label_list]

        max_value = max(counts)
        max_indices = [i for i, value in enumerate(counts) if value == max_value]
        result = max_indices[0] if len(max_indices) == 1 else None
        return result

    def get_pred(self, loader):
        results = []
        classification_prompt = get_classification_prompt(self.dataset)
        for sample in loader:
            for x,y in sample:
                messages = [
                        {"role": "system", "content": classification_prompt},
                        {"role": "user", "content": f"{x}"},
                    ]
                    
                answer = self.pipe(messages, max_new_tokens=256)[0]['generated_text'][-1]['content']
                counts = [answer.lower().count(label) - answer.lower().count(f"_{label}") for label in self.label_list]

                max_value = max(counts)
                max_indices = [i for i, value in enumerate(counts) if value == max_value]
                # Note that answer can be "I think entailment and not_entailment are both wrong!" or max_value can be 0
                results.append(max_indices[0] if len(max_indices) == 1 else None)
                messages.append({"role": "assistant", "content": answer})
        
        return results

    def get_adv_prompt_from_red_LLM(self, orig_sentence, pred):
        reason = self.ask_reason(orig_sentence,pred)
        system_prompt = get_adv_prompt(self.dataset, orig_sentence, pred, reason)
        messages = [
                        {"role": "system", "content": system_prompt},
                    ]
        
        
        answer = self.pipe(messages, max_new_tokens=256)[0]['generated_text'][-1]['content']
        messages.append({"role": "assistant", "content": answer})
        return answer, messages

    def get_iterative_feedback_from_red_LLM(self, adv_pred, label_list,pred,score,adv_sentence,red_LLM_chats):
        reason = self.ask_reason(adv_sentence,adv_pred)
        iterative_instruction = get_iterative_feedback(self.dataset, adv_pred, label_list, pred, score, adv_sentence, reason)
        red_LLM_chats.append({"role": "user", "content": iterative_instruction})
        
        answer = self.pipe(red_LLM_chats, max_new_tokens=256)[0]['generated_text'][-1]['content']
        red_LLM_chats.append({"role": "assistant", "content": answer})
        return answer, red_LLM_chats

    def get_adversarial_example(self,orig_sentence,pred):
        answers=[]
        scores=[]
        adv_prompts=[]
        adv_preds=[]
        adv_prompt, red_LLM_chats = self.get_adv_prompt_from_red_LLM(orig_sentence, pred)
        messages = [
                        {"role": "system", "content": adv_prompt},
                    ]
        num_iter = 9
        for i in range(num_iter):
            answer = self.pipe(messages, max_new_tokens=256)[0]['generated_text'][-1]['content']
            answer =self.filter_response(answer)
            messages.append({"role": "assistant", "content": answer})
            adv_pred = self.get_single_pred(answer)
          
            score = self.get_similarity_score(orig_sentence, answer)
            answers.append(answer)
            scores.append(score)
            adv_prompts.append(adv_prompt)
            adv_preds.append(adv_pred)
            if (adv_pred is None or self.label_list[adv_pred]!=pred) and score >= similarity_threshold:
                json_messages = json.dumps(messages)
                with open(f"log_{self.dataset}.txt","a") as f:
                    f.write("Conversations:\n")
                    f.write(f"{json_messages} \n")
                    f.write("Red LLM Conversations:\n")
                    f.write(f"{json.dumps(red_LLM_chats)} \n")
                while(len(scores)!=num_iter):
                    answers.append(answer)
                    scores.append(score)
                    adv_prompts.append(adv_prompt)
                    adv_preds.append(adv_pred)
                return answers, scores, adv_preds
            new_message, red_LLM_chats = self.get_iterative_feedback_from_red_LLM(adv_pred, self.label_list,pred,score,answer,red_LLM_chats)
            messages.append({"role": "user", "content": new_message})
            adv_prompt = new_message
        
        json_messages = json.dumps(messages)
        with open(f"log_{self.dataset}.txt","a") as f:
            f.write("Conversations:\n")
            f.write(f"{json_messages} \n")
            f.write("Red LLM Conversations:\n")
            f.write(f"{json.dumps(red_LLM_chats)} \n")
        return answers, scores, adv_preds
    
    def get_similarity_score(self,orig,adv):
        similarity_check_prompt = get_similarity_score_prompt(orig,adv)
        messages = [
                        {"role": "user", "content": similarity_check_prompt},
                    ]
        score = -1
        attempt=0
        while(True):
            attempt+=1
            answer = self.pipe(messages, max_new_tokens=256)[0]['generated_text'][-1]['content']
            try:
                score = float(answer)
                break
            except ValueError:
                print("Not a Number")
                print(answer)
                if attempt>10:
                    return 1

        
        return score
    
    def ask_reason(self, sentence, pred):
        if pred is None:
            return "Nothing"
        reasoning_prompt = get_reasoning_prompt(self.dataset,pred, sentence)
        messages = [
                        {"role": "user", "content": reasoning_prompt},
                    ]
            
        reason = self.pipe(messages, max_new_tokens=256)[0]['generated_text'][-1]['content']
        return reason
    
    def filter_response(self, sentence):
        filtering_prompt = get_filtering_prompt(self.dataset)
        messages = [
                        {"role": "system", "content": filtering_prompt},
                        {"role": "user", "content": sentence},
                    ]
            
        reason = self.pipe(messages, max_new_tokens=256)[0]['generated_text'][-1]['content']
        return reason
