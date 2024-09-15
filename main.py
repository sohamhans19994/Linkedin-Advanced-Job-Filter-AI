from JobsScraper.scrape import Scrape
from SegmentClassify_BERT.extract_qualifications_section import ExtractQualifications
from LLMFilter.job_filter import JobFilter
import pandas as pd
import os
import json
import yaml
from datetime import datetime
import shutil
import argparse


with open("settings.yaml", 'r') as file:
    config = yaml.safe_load(file)

def scrape():
    date = datetime.now().strftime('%Y-%m-%d')
    save_path = os.path.join(config["DATA_FOLDER"],date)
    if os.path.exists(save_path):
        delete = input(f"Delete {save_path}?")
        if delete:
            shutil.rmtree(save_path)
            os.makedirs(save_path)
    else:
        os.makedirs(save_path)
    
    scraper = Scrape(save_path)
    return save_path


def score(save_path,extractor,jobFilter):
    with open('LLM_filter/filter_questions.json', 'r') as file:
        questions = json.load(file)

    def score_row(job_row):
        qualifications = extractor.extract_qualifications(job_row['Description HTML'])
        return jobFilter.score_job(qualifications,questions)
    
    jobs_df = pd.read_csv(os.path.join(save_path,"jobs.csv"))
    jobs_df['Match_Score'] = jobs_df.apply(score_row, axis=1)
    jobs_df_sorted = jobs_df.sort_values(by='Match_Score', ascending=False)
    jobs_df_sorted.to_csv(os.path.join(save_path,"jobs.csv"))
    print(f"Jobs with scores saved at {os.path.join(save_path,'jobs.csv')}")
    return jobs_df_sorted



def main(args):
    qualifications_extractor = ExtractQualifications(config['BERT_MODEL_PATH'])
    jobFilter = JobFilter(config['LLAMA_MODEL_PATH'])
    if args.scrape:
        save_path = scrape()
        print(f"Scraped results saved to {save_path}")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to scrape data and score jobs.")
    parser.add_argument(
        '--scrape', 
        action='store_true', 
        help="Run the scrape process."
    )
    parser.add_argument(
        '--score', 
        action='store_true', 
        help="Run the scoring process."
    )
    