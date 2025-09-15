import pickle
from iterative_interact import Interact
from transformers import pipeline
import torch

batch_size = 1
similarity_threshold=6
dataset_name = ""

with open("dataset", "r") as f:
    dataset_name = f.read().strip()

def get_dataset(dataset_name):
    test_loader = None
    if dataset_name == "spam":
        dataset_path = 'data/spam_data_test.pkl'
        with open(dataset_path, 'rb') as f:
                dataset_ = pickle.load(f)
                test_loader = dataset_['data']
    elif dataset_name == "spam3":
        dataset_path = 'data/spam_data3.pkl'
        with open(dataset_path, 'rb') as f:
                dataset_ = pickle.load(f)
                test_loader = dataset_['data']
    elif dataset_name == "hate":
        dataset_path = 'data/hate_data_test.pkl'
        with open(dataset_path, 'rb') as f:
                dataset_ = pickle.load(f)
                test_loader = dataset_['data']
                test_loader = test_loader[:100]
    elif dataset_name == "toxic":
        dataset_path = 'data/severe_toxic_comments.pkl'
        with open(dataset_path, 'rb') as f:
                dataset_ = pickle.load(f)
                test_loader = dataset_['data']
    elif dataset_name == "liar":
        dataset_path = 'data/sampled_fake_news.pkl'
        with open(dataset_path, 'rb') as f:
                dataset_ = pickle.load(f)
                test_loader = dataset_['data']
                test_loader = test_loader[:100]
    test_loader = [
        test_loader[i : i + batch_size]
        for i in range(0, len(test_loader), batch_size)
    ]

    if dataset_name == "spam" or dataset_name == "spam3" or dataset_name == "hate" or dataset_name == "toxic" or dataset_name == "liar":
        label_list = list(map(dataset_['labels'].get, sorted(dataset_['labels'])))
    
    return test_loader, label_list

def save_data_pickle(original_text, adverarial_text, label, filename=f"adversarial_examples_{dataset_name}.pkl"):
    with open(filename, 'ab') as file:
        pickle.dump({'original_text': original_text, 'adverarial_text': adverarial_text, 'label': label}, file)

def get_accuracy(pred, label):
    assert len(pred) == len(label)
    correct = [i == j for (i, j) in zip(pred, label)]
    return sum(correct) / len(label)


def get_ASR(pred, adv_pred, label, similarity_scores):
    assert len(pred) == len(label) and len(pred) == len(adv_pred)
    correct = [i == j for (i, j) in zip(pred, label)]
    adv_wrong = [
        (pred[i] == label[i] and adv_pred[i] != label[i] and similarity_scores[i]>=similarity_threshold) for i in range(len(label))
    ]
    return sum(adv_wrong) / sum(correct)


test_loader, label_list = get_dataset(dataset_name)
pipe = pipeline("text-generation", model="meta-llama/Meta-Llama-3-70B-Instruct", torch_dtype=torch.float16, device_map="auto")
interact = Interact(dataset_name, pipe,label_list)
pred = interact.get_pred(test_loader)
label = [y for batch in test_loader for x, y in batch]
sentences = [x for batch in test_loader for x, y in batch]
adv_preds=[]
similarities=[]
for i in range(len(sentences)):
    if(label[i]!=pred[i]):
        adv_preds.append([pred[i]]*9)
        similarities.append([0]*9)
        continue
    
    with open(f"log_{dataset_name}.txt","a") as f:
        f.write("Original Sentence:\n")
        f.write(sentences[i]+"\n")
        f.write("Original Prediction:\n")
        f.write(f"{pred[i]}\n")
    adv_sample, similarity, adv_pred = interact.get_adversarial_example(sentences[i],label_list[pred[i]])
    adv_preds.append(adv_pred)
    similarities.append(similarity)
    if (pred[i] == label[i] and adv_pred[-1] != label[i] and similarity[-1]>=similarity_threshold):
        save_data_pickle(sentences[i],adv_sample,pred[i])
    with open(f"output_{dataset_name}.txt","a") as f:
        f.write("Original Sentence:\n")
        f.write(sentences[i]+"\n")
        f.write("Original Prediction:\n")
        f.write(f"{pred[i]}\n")
        f.write("Adv Sentence:\n")
        f.write(f"{adv_sample[-1]}\n")
        f.write("New Prediction:\n")
        f.write(f"{adv_pred[-1]}\n")
        f.write("Similarity Score:\n")
        f.write(f"{similarity[-1]}\n")
    with open(f"log_{dataset_name}.txt","a") as f:
        f.write("----------------------\n")
        f.write("Final Outcome:\n")
        f.write("Adv Sentence:\n")
        f.write(f"{adv_sample[-1]}\n")
        f.write("New Prediction:\n")
        f.write(f"{adv_pred[-1]}\n")
        f.write("Similarity Score:\n")
        f.write(f"{similarity[-1]}\n")
        f.write("-----------------------\n")
    


for i in range(len(similarities[0])):
    with open(f"out_{dataset_name}{i}.txt","a") as f:
        f.write("Task Accuracy: ")
        f.write(f"{get_accuracy(pred,label)}\n")     
        f.write("Attack Success Rate: ")
        adv_pred = [x[i] for x in adv_preds]
        similarity =[x[i] for x in similarities]
        f.write(f"{get_ASR(pred, adv_pred, label, similarity)}\n")
