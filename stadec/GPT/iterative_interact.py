from openai import OpenAI
import time
import json
from prompts import get_classification_prompt, get_adv_prompt, get_iterative_feedback, get_similarity_score_prompt
similarity_threshold=7
class Interact:
    def __init__(self, dataset, API_key, API_base, version, label_list):
        self.label_list = label_list
        self.dataset = dataset
        self.client = OpenAI(api_key=API_key, base_url=API_base)
        self.version = version

    def get_single_pred(self, sentence):
        classification_prompt = get_classification_prompt(self.dataset)
        messages = [
                        {"role": "system", "content": classification_prompt},
                        {"role": "user", "content": f"{sentence}"},
                    ]
        response = None
        while response is None:
            try:
                response = self.client.chat.completions.create(
                    model=self.version,
                    temperature=0.1,
                    messages=messages,
                )
            except Exception as e:
                print(e)
                time.sleep(10)
            
        answer = response.choices[0].message.content
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

                response = None
                while response is None:
                    try:
                        response = self.client.chat.completions.create(
                            model=self.version,
                            temperature=0.1,
                            messages=messages,
                        )
                    except Exception as e:
                        print(e)
                        time.sleep(10)
                    
                answer = response.choices[0].message.content
                counts = [answer.lower().count(label) - answer.lower().count(f"_{label}") for label in self.label_list]

                max_value = max(counts)
                max_indices = [i for i, value in enumerate(counts) if value == max_value]
                results.append(max_indices[0] if len(max_indices) == 1 else None)
                messages.append({"role": "assistant", "content": answer})
        
        return results

    def get_adv_prompt_from_red_LLM(self, orig_sentence, pred):
        prompt = get_adv_prompt(self.dataset, orig_sentence)
        return prompt

    def get_iterative_feedback_from_red_LLM(self, adv_pred, label_list,pred,score):
        iterative_instruction = get_iterative_feedback(self.dataset, adv_pred, label_list, pred, score)
        return iterative_instruction

    def get_adversarial_example(self,orig_sentence,pred):
        answers=[]
        scores=[]
        adv_preds=[]
        adv_prompt = self.get_adv_prompt_from_red_LLM(orig_sentence, pred)
        messages = [
                        {"role": "system", "content": adv_prompt},
                    ]
        num_iter = 9
        for i in range(num_iter):
            response = None
            while response is None:
                try:
                    response = self.client.chat.completions.create(
                        model=self.version,
                        temperature=0.1,
                        messages=messages,
                    )
                except Exception as e:
                    print(e)
                    time.sleep(10)
            
            answer = response.choices[0].message.content
            messages.append({"role": "assistant", "content": answer})
            adv_pred = self.get_single_pred(answer)
          
            score = self.get_similarity_score(orig_sentence, answer)
            answers.append(answer)
            scores.append(score)
            adv_preds.append(adv_pred)
            if (adv_pred is None or self.label_list[adv_pred]!=pred) and score >= similarity_threshold:
                json_messages = json.dumps(messages)
                with open(f"log_{self.dataset}.txt","a") as f:
                    f.write("Conversations:\n")
                    f.write(f"{json_messages} \n")
                while(len(scores)!=num_iter):
                    answers.append(answer)
                    scores.append(score)
                    adv_preds.append(adv_pred)
                return answers, scores, adv_preds
            new_message = self.get_iterative_feedback_from_red_LLM(adv_pred, self.label_list,pred,score)
            messages.append({"role": "user", "content": new_message})
        
        json_messages = json.dumps(messages)
        with open(f"log_{self.dataset}.txt","a") as f:
            f.write("Conversations:\n")
            f.write(f"{json_messages} \n")
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
            response = None
            while response is None:
                try:
                    response = self.client.chat.completions.create(
                        model=self.version,
                        temperature=0.1,
                        messages=messages,
                    )
                except Exception as e:
                    print(e)
                    time.sleep(10)
                
            answer = response.choices[0].message.content
            try:
                score = float(answer)
                break
            except ValueError:
                print("Not a Number")
                print(answer)
                if attempt>10:
                    return 1

        
        return score
    
    