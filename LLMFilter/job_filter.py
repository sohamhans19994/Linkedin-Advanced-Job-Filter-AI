from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.llms.llama_cpp.llama_utils import (
    messages_to_prompt,
    completion_to_prompt,
)

class JobFilter:
    def __init__(self,model_path):
        self.llm = LlamaCPP(
            model_path=model_path,
            temperature=0.1,
            max_new_tokens=1,
            context_window=3900,
            generate_kwargs={},
            model_kwargs={"n_gpu_layers": 50},
        )

        self.prompt_template = "Q: Based on the following description of job qualifications:\n {qualifications}\n Answer in true or false: {question}"

    def parse_boolean(self,text):
        """
        Convert a text string into a boolean value. Handles various representations and extra spaces.
        
        Args:
            text (str): The input text to be converted to boolean.
            
        Returns:
            bool: The parsed boolean value.
            
        Raises:
            ValueError: If the text cannot be interpreted as a boolean.
        """
        # Strip leading/trailing spaces and convert to lowercase
        cleaned_text = text.strip().lower()
        
        # If the text is too long, take the first word
        if len(cleaned_text) > 0 and ' ' in cleaned_text:
            cleaned_text = cleaned_text.split()[0]
        
        # Define accepted boolean representations
        true_values = {'true', 'yes', '1', 't', 'y'}
        false_values = {'false', 'no', '0', 'f', 'n'}
        
        # Check if the cleaned text is in true_values or false_values
        if cleaned_text in true_values:
            return True
        elif cleaned_text in false_values:
            return False
        else:
            raise ValueError(f"Cannot interpret '{text}' as a boolean.")
    
    def score_job(self,qualifications_text,questions_list):
        score = 0
        max_score = 0
        for question,q_score in questions_list:
            if q_score > 0:
                max_score += q_score
            try:
                answer = self.llm.complete(self.prompt_template.format(qualifications=qualifications_text, question=question))
                if self.parse_boolean(answer):
                    score += q_score
            except ValueError:
                score += 0
        return score/max_score

