import logging
from linkedin_jobs_scraper import LinkedinScraper
from linkedin_jobs_scraper.events import Events, EventData, EventMetrics
from linkedin_jobs_scraper.query import Query, QueryOptions, QueryFilters
from linkedin_jobs_scraper.filters import RelevanceFilters, TimeFilters, TypeFilters, ExperienceLevelFilters, \
    OnSiteOrRemoteFilters
import pandas as pd
import os
import re
import threading
import time
from datetime import datetime
from tqdm import tqdm

class Scrape:
    def __init__(self,save_folder_path):
        self.jobs_save_path = os.path.join(save_folder_path,"jobs.csv")
        self.logs_save_path = os.path.join(save_folder_path,"logs.txt")
        logging.basicConfig(
            filename=self.logs_save_path,  # Log messages to this file
            level=logging.INFO,  # Capture all messages of DEBUG level and above
            filemode='a'  # Append to the log file (use 'w' to overwrite)
        )
    
    def linkedin_jobs(self):
        list_lock = threading.Lock()
        self.jobs_list = []
        self.just_saved = False
        def on_data(data: EventData):
            row = [data.title, data.company, data.company_link, data.date, data.link, data.insights,data.description,data.description_html,None]
            with list_lock:
                self.jobs_list.append(row)
        def on_error(error):
            logging.error(error)
        
        def on_metrics(metrics: EventMetrics):
            logging.info('[ON_METRICS]'+ str(metrics))

        def on_end():
            jobs_df = pd.DataFrame(self.jobs_list, columns=["Title", "Company", "Company Link", "Date", "Link", "Insights", "Description", "Description HTML","Match Score"])
            print(f"Found {len(self.jobs_list)} jobs")
            jobs_df.to_csv(self.jobs_save_path,index=False)
            logging.info('[END]')
        
        self.scraper = LinkedinScraper(
            chrome_executable_path='/usr/bin/chromedriver',
            chrome_binary_location=None, 
            chrome_options=None, 
            headless=True,  
            max_workers=10,  
            slow_mo=2,  
            page_load_timeout=40      
        )
        self.scraper.on(Events.DATA, on_data)
        self.scraper.on(Events.ERROR, on_error)
        self.scraper.on(Events.END, on_end)
        self.scraper.on(Events.METRICS,on_metrics)
    
        queries = [
            Query(
                query='Machine Learning OR AI OR Software Engineering',
                options=QueryOptions(
                    limit=10000,
                    locations=['United States'],
                    apply_link=False,  
                    skip_promoted_jobs=False,  
                    filters=QueryFilters(
                        relevance=RelevanceFilters.RECENT,
                        time=TimeFilters.DAY
                    )
                )
            )
        ]

        self.scraper.run(queries)
    
    def add_to_cloud(self):
        pass

    def get_from_cloud(self):
        pass