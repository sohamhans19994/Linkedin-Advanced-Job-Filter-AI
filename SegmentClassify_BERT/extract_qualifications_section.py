from transformers import pipeline
from .utility import split_sections
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class ExtractQualifications:
    def __init__(self,model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.classifier = pipeline('text-classification', model=self.model, tokenizer=self.tokenizer)
        
    def extract_qualifications(self,job_desc_html):
        data_sections = split_sections(job_desc_html)
        results = self.classifier(data_sections)
        filtered_texts = [text for text, result in zip(data_sections, results) if result['label'] == "True"]
        qualifications_text = " ".join(filtered_texts)
        return qualifications_text


