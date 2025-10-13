from transformers import pipeline
from .utility import split_sections
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class ExtractQualifications:
    def __init__(self,model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path,model_max_len=512)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.classifier = pipeline('text-classification', model=self.model, tokenizer=self.tokenizer,device=device)


    def extract_qualifications(self,job_desc_html):
        data_sections = split_sections(job_desc_html)
        results = self.classifier(data_sections,padding=True,truncation=True)
        filtered_texts = [text for text, result in zip(data_sections, results) if result['label'] == "True"]
        qualifications_text = " ".join(filtered_texts)
        return qualifications_text


