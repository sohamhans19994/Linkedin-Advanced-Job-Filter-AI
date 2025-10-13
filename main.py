from JobsScraper.scrape import Scrape
from SegmentClassify_BERT.extract_qualifications_section import ExtractQualifications
import pandas as pd
import os
import json
import yaml
from datetime import datetime
import shutil
import argparse
from tqdm import tqdm

script_path = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(script_path, 'settings.yaml'), 'r') as file:
    config = yaml.safe_load(file)

def scrape(scraper, save_path, use_drive=False):
    # date = datetime.now().strftime('%Y-%m-%d')
    # save_path = os.path.join(config["DATA_DIR"],date)
    if os.path.exists(save_path):
        delete = input(f"Delete {save_path}?")
        if delete:
            shutil.rmtree(save_path)
            os.makedirs(save_path)
    else:
        os.makedirs(save_path,exist_ok=True)
    
    scraper.linkedin_jobs()
    if use_drive:
        sorted_save_name = os.path.basename(os.path.dirname(save_path)) + "_jobsSorted.csv"
        scraper.add_to_cloud(local_path=save_path,cloud_save_name=sorted_save_name)
    return save_path



def score(save_path,extractor,jobFilter):
    with open(os.path.join(script_path,'LLMFilter/filter_questions.json'), 'r') as file:
        questions = json.load(file)

    def score_row(job_row):
        qualifications = extractor.extract_qualifications(job_row['Description HTML'])
        qualifications = job_row['Title'] + "\n" + job_row['Company'] + '\n' + qualifications
        return jobFilter.score_job(qualifications,questions)
    
    tqdm.pandas()

    jobs_df = pd.read_csv(os.path.join(save_path,"jobs.csv"))
    jobs_df['Match_Score'] = jobs_df.progress_apply(score_row, axis=1)
    jobs_df_sorted = jobs_df.sort_values(by='Match_Score', ascending=False)
    jobs_df_sorted.to_csv(os.path.join(save_path,"jobs.csv"))
    print(f"Jobs with scores saved at {os.path.join(save_path,'jobs.csv')}")
    return jobs_df_sorted



def main(args):
    # date = datetime.now().strftime('%Y-%m-%d')
    date = args.date
    save_path = os.path.join(config["DATA_DIR"],date)
    # os.environ['LI_AT_COOKIE'] = config["LI_AT_COOKIE"]
    scraper = Scrape(save_path,config)
    if args.scrape:
        save_path = scrape(scraper, save_path, args.use_drive)

    if args.score:
        from LLMFilter.job_filter import JobFilter
        qualifications_extractor = ExtractQualifications(config['BERT_MODEL_PATH'])
        jobFilter = JobFilter(config['LLAMA_MODEL_PATH'])
        if args.use_drive:
            if os.path.exists(os.path.join(save_path,"jobs.csv")):
                print("Jobs file exists")
            else:
                os.makedirs(save_path, exist_ok=True)
                scraper.get_from_cloud(save_path)
        score(save_path,qualifications_extractor,jobFilter) 
        if args.use_drive:
            cloud_save_name = os.path.basename(os.path.dirname(save_path)) + "_jobsSorted.csv"
            scraper.add_to_cloud(os.path.join(save_path,"jobs.csv"),cloud_save_name=cloud_save_name)

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
    parser.add_argument(
        '--use-drive', 
        action='store_true', 
        help="Run the scoring process."
    )
    parser.add_argument(
        '--date',
        type=str,
        default=datetime.now().strftime('%Y-%m-%d'),
        help='Enter date in yyyy-mm-dd format (default: %(default)s)'
    )
    
    args = parser.parse_args()
    main(args)