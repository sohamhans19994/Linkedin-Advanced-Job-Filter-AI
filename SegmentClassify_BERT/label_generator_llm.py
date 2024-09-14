from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field, constr
from langchain_openai import ChatOpenAI
# from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
import os
from langchain_community.callbacks import get_openai_callback
import csv
import json
from tqdm import tqdm


def label_data(text_blocks_list, save_data_path):
    llm = ChatOpenAI(model="gpt-4o-mini")
    
    class ClassificationResult(BaseModel):
        is_qualifications: bool

    parser = JsonOutputParser(pydantic_object=ClassificationResult)

    prompt = PromptTemplate(
        template="Classify the following text into whether or not it is the qualifications section of a job description. \n{text}\n {format_instructions} Strictly follow the output format don't add any additional words or comments",
        input_variables=["text"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | llm | parser

    labelled_data = {"text":[],"label":[]}
    for text_block in tqdm(text_blocks_list):
        try:
            out = chain.invoke({"text": text_block})
            labelled_data["text"].append(text_block)
            labelled_data["label"].append(out['is_qualifications'])
        except Exception as e:
            print(f"LLM error: {e}")
    
    print("Data labelled... saving...")

    # Open the file in write mode
    with open(save_data_path, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=labelled_data.keys())
        writer.writeheader()
        rows = zip(*labelled_data.values())
        for row in rows:
            writer.writerow(dict(zip(labelled_data.keys(), row)))

    print(f"Data has been written to {save_data_path}")

if __name__ == "__main__":
    os.environ['OPENAI_API_KEY'] = "OPENAI_API_KEY"
    with open('../data/bert_finetune/sections_list.json', 'r') as file:
        text_blocks_list = json.load(file)
    label_data(text_blocks_list,"../data/bert_finetune/classification_training_data.csv")
