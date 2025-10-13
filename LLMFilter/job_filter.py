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
            verbose=False
        )

        self.prompt_template = "Q: Based on the following description of job qualifications:\n {qualifications}\n Answer in true or false: {question}\nA: "

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
            max_score += abs(q_score)
            try:
                answer = self.llm.complete(self.prompt_template.format(qualifications=qualifications_text, question=question))
                if self.parse_boolean(answer.text):
                    score += q_score
                else:
                    score -= q_score
            except ValueError:
                score += 0
        return score/max_score

# if __name__ == "__main__":
#     jobFilter = JobFilter("/home/sohans/Meta-Llama-3-70B-Instruct-IQ2_S.gguf")
#     qualifications_text = """
#     Amazon Business SDE II, Amazon
#     Basic Qualifications

#     3+ years of non-internship professional software development experience
#     2+ years of non-internship design or architecture (design patterns, reliability and scaling) of new and existing systems experience
#     Experience programming with at least one software programming language
#     Experience in machine learning, data mining, information retrieval, statistics or natural language processing

#     Preferred Qualifications

#     3+ years of full software development life cycle, including coding standards, code reviews, source control management, build processes, testing, and operations experience
#     Bachelor's degree in computer science or equivalent

#     """
#     questions_list = [
#         ["Is the job in the AI or ML or a related field?",2],
#         ["Is the job in the software engineering field?",0.5],
#         ["Does the job need 3 or more years of AI/ML experience?",-0.5],
#         ["Does the job need 5 or more years of AI/ML experience?",-0.5],
#         ["Does the job need 7 or more years of AI/ML experience?",-1],
#         ["Is the company famous?",1],
#         ["Does the job require any specific security clearance?",-2],
#         ["Does the job strongly require a PhD?",-1],
#         ["Does the job have a preference for PhD but no strong necessity?",-0.5]
#     ]
#     jobFilter.score_job(qualifications_text,questions_list)